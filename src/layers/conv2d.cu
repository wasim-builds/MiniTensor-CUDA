/**
 * @file conv2d.cu
 * @brief Conv2d layer: im2col → GEMM → col2im for backward.
 */

#include "minitensor/layers/conv2d.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <sstream>

namespace minitensor {
namespace layers {

Conv2d::Conv2d(int in_ch, int out_ch,
               int kH, int kW, int sH, int sW,
               int pH, int pW, int dH, int dW,
               bool bias, Device device)
    : in_ch_(in_ch), out_ch_(out_ch),
      kH_(kH), kW_(kW),
      stride_h_(sH), stride_w_(sW),
      pad_h_(pH), pad_w_(pW),
      dilation_h_(dH), dilation_w_(dW),
      use_bias_(bias), device_(device)
{
    weight_ = Tensor({out_ch, in_ch, kH, kW}, DType::Float32, device);
    if (use_bias_)
        bias_ = Tensor({out_ch}, DType::Float32, device);
    reset_parameters();
}

void Conv2d::reset_parameters() {
    int fan_in = in_ch_ * kH_ * kW_;
    float std  = std::sqrt(2.0f / static_cast<float>(fan_in));
    Tensor w = Tensor::randn({out_ch_, in_ch_, kH_, kW_},
                             DType::Float32, Device::CPU, 0.0f, std);
    weight_ = w.to(device_);
    if (use_bias_)
        bias_ = Tensor::zeros({out_ch_}, DType::Float32, device_);
}

// ─── Forward ──────────────────────────────────────────────────────────────────
Tensor Conv2d::forward(const Tensor& x) {
    NVTX_RANGE("Conv2d::forward");

    N_cache_ = static_cast<int>(x.shape()[0]);
    C_cache_ = static_cast<int>(x.shape()[1]);
    H_cache_ = static_cast<int>(x.shape()[2]);
    W_cache_ = static_cast<int>(x.shape()[3]);
    input_cache_ = x;

    int outH = output_h(H_cache_);
    int outW = output_w(W_cache_);

    // col matrix: (C*kH*kW) × (N*outH*outW)
    int col_rows = in_ch_ * kH_ * kW_;
    int col_cols = N_cache_ * outH * outW;
    col_cache_ = Tensor({col_rows, col_cols}, DType::Float32, device_);

    kernels::launch_im2col(
        x.data_ptr<float>(), col_cache_.data_ptr<float>(),
        N_cache_, C_cache_, H_cache_, W_cache_,
        kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_,
        dilation_h_, dilation_w_);

    // GEMM: weight (out_ch, col_rows) × col (col_rows, col_cols)
    //     → out_mat (out_ch, col_cols)
    Tensor out_mat({out_ch_, col_cols}, DType::Float32, device_);
    // Reshape weight to (out_ch, col_rows)
    Tensor w2d = weight_.reshape({out_ch_, col_rows});
    kernels::launch_matmul(
        w2d.data_ptr<float>(), col_cache_.data_ptr<float>(),
        out_mat.data_ptr<float>(),
        out_ch_, col_cols, col_rows);

    if (use_bias_)
        kernels::launch_add_bias(out_mat.data_ptr<float>(),
                                 bias_.data_ptr<float>(), col_cols, out_ch_);

    // Reshape to (N, out_ch, outH, outW)
    return out_mat.reshape({N_cache_, out_ch_, outH, outW});
}

// ─── Backward ─────────────────────────────────────────────────────────────────
Tensor Conv2d::backward(const Tensor& grad_out) {
    NVTX_RANGE("Conv2d::backward");

    int outH = output_h(H_cache_);
    int outW = output_w(W_cache_);
    int col_rows = in_ch_ * kH_ * kW_;
    int col_cols = N_cache_ * outH * outW;

    // Reshape grad_out: (N, out_ch, outH, outW) → (out_ch, col_cols)
    Tensor go2d = grad_out.reshape({out_ch_, col_cols});

    // grad_weight: (out_ch, col_rows) = go2d (out_ch, col_cols) × col^T (col_cols, col_rows)
    // Transpose col_cache (col_rows, col_cols) → col_T (col_cols, col_rows)
    Tensor col_cpu = col_cache_.to(Device::CPU);
    Tensor colT({col_cols, col_rows}, DType::Float32, Device::CPU);
    float* sc = col_cpu.data_ptr<float>();
    float* dc = colT.data_ptr<float>();
    for (int r = 0; r < col_rows; ++r)
        for (int c = 0; c < col_cols; ++c)
            dc[c * col_rows + r] = sc[r * col_cols + c];
    Tensor colT_dev = colT.to(device_);

    Tensor grad_w({out_ch_, col_rows}, DType::Float32, device_);
    kernels::launch_matmul(
        go2d.data_ptr<float>(), colT_dev.data_ptr<float>(),
        grad_w.data_ptr<float>(), out_ch_, col_rows, col_cols);
    // grad_w reshaped back to (out_ch, in_ch, kH, kW) — stored externally

    // grad_col: (col_rows, col_cols) = W^T (col_rows, out_ch) × go2d (out_ch, col_cols)
    Tensor w2d = weight_.reshape({out_ch_, col_rows});
    // Transpose w2d
    Tensor wT_cpu = w2d.to(Device::CPU);
    Tensor wT({col_rows, out_ch_}, DType::Float32, Device::CPU);
    float* sw = wT_cpu.data_ptr<float>();
    float* dw = wT.data_ptr<float>();
    for (int o = 0; o < out_ch_; ++o)
        for (int r = 0; r < col_rows; ++r)
            dw[r * out_ch_ + o] = sw[o * col_rows + r];
    Tensor wT_dev = wT.to(device_);

    Tensor grad_col({col_rows, col_cols}, DType::Float32, device_);
    kernels::launch_matmul(
        wT_dev.data_ptr<float>(), go2d.data_ptr<float>(),
        grad_col.data_ptr<float>(), col_rows, col_cols, out_ch_);

    // col2im → grad_x
    Tensor grad_x = Tensor::zeros({N_cache_, C_cache_, H_cache_, W_cache_},
                                  DType::Float32, device_);
    kernels::launch_col2im(
        grad_col.data_ptr<float>(), grad_x.data_ptr<float>(),
        N_cache_, C_cache_, H_cache_, W_cache_,
        kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_,
        dilation_h_, dilation_w_);

    return grad_x;
}

std::string Conv2d::name() const {
    std::ostringstream oss;
    oss << "Conv2d(" << in_ch_ << "→" << out_ch_
        << ", k=" << kH_ << "×" << kW_ << ")";
    return oss.str();
}

} // namespace layers
} // namespace minitensor
