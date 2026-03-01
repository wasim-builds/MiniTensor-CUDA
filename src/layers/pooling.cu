/**
 * @file pooling.cu
 * @brief MaxPool2D and AvgPool2D kernel implementations + layer wrappers.
 */

#include "minitensor/layers/pooling.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <cfloat>

namespace minitensor {
namespace kernels {

// ─── MaxPool2D forward ────────────────────────────────────────────────────────
__global__ void maxpool2d_fwd_kernel(
    const float* __restrict__ x, float* __restrict__ out, int32_t* __restrict__ mask,
    int N, int C, int H, int W, int kH, int kW,
    int sH, int sW, int pH, int pW, int outH, int outW)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // outH×outW index
    int nc  = blockIdx.y;                            // N×C index

    if (nc >= N * C || col >= outH * outW) return;

    int oh = col / outW;
    int ow = col % outW;
    int n  = nc / C;
    int c  = nc % C;

    float best_val = -FLT_MAX;
    int   best_idx = -1;

    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh * sH - pH + kh;
            int iw = ow * sW - pW + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                int src = ((n * C + c) * H + ih) * W + iw;
                if (x[src] > best_val) { best_val = x[src]; best_idx = src; }
            }
        }
    }

    int dst = ((n * C + c) * outH + oh) * outW + ow;
    out[dst]  = best_val;
    mask[dst] = best_idx;
}

// ─── MaxPool2D backward ───────────────────────────────────────────────────────
__global__ void maxpool2d_bwd_kernel(
    const float* __restrict__ grad_out, const int32_t* __restrict__ mask,
    float* __restrict__ grad_in, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    atomicAdd(&grad_in[mask[i]], grad_out[i]);
}

// ─── AvgPool2D forward ────────────────────────────────────────────────────────
__global__ void avgpool2d_fwd_kernel(
    const float* __restrict__ x, float* __restrict__ out,
    int N, int C, int H, int W, int kH, int kW,
    int sH, int sW, int pH, int pW, int outH, int outW)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int nc  = blockIdx.y;
    if (nc >= N * C || col >= outH * outW) return;

    int oh = col / outW, ow = col % outW;
    int n = nc / C, c = nc % C;
    float sum = 0.0f; int cnt = 0;

    for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh * sH - pH + kh, iw = ow * sW - pW + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += x[((n * C + c) * H + ih) * W + iw];
                ++cnt;
            }
        }
    out[((n * C + c) * outH + oh) * outW + ow] = sum / static_cast<float>(cnt);
}

// ─── AvgPool2D backward ──────────────────────────────────────────────────────
__global__ void avgpool2d_bwd_kernel(
    const float* __restrict__ grad_out, float* __restrict__ grad_in,
    int N, int C, int H, int W, int outH, int outW,
    int kH, int kW, int sH, int sW, int pH, int pW)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int nc  = blockIdx.y;
    if (nc >= N * C || col >= outH * outW) return;

    int oh = col / outW, ow = col % outW;
    int n = nc / C, c = nc % C;
    float g = grad_out[((n * C + c) * outH + oh) * outW + ow];
    float cnt = static_cast<float>(kH * kW);

    for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh * sH - pH + kh, iw = ow * sW - pW + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                atomicAdd(&grad_in[((n * C + c) * H + ih) * W + iw], g / cnt);
        }
}

// ─── Host launchers ───────────────────────────────────────────────────────────
void launch_maxpool2d_forward(const float* x, float* out, int32_t* mask,
                              int N, int C, int H, int W,
                              int kH, int kW, int sH, int sW, int pH, int pW)
{
    NVTX_RANGE("maxpool2d_fwd");
    int outH = (H + 2*pH - kH) / sH + 1;
    int outW = (W + 2*pW - kW) / sW + 1;
    dim3 block(256);
    dim3 grid((outH*outW + 255) / 256, N * C);
    maxpool2d_fwd_kernel<<<grid, block>>>(x, out, mask, N, C, H, W, kH, kW, sH, sW, pH, pW, outH, outW);
    CUDA_CHECK(cudaGetLastError());
}

void launch_maxpool2d_backward(const float* grad_out, const int32_t* mask,
                               float* grad_in, int N, int C, int H, int W,
                               int outH, int outW)
{
    NVTX_RANGE("maxpool2d_bwd");
    int total = N * C * outH * outW;
    int block = 256, grid = (total + 255) / 256;
    maxpool2d_bwd_kernel<<<grid, block>>>(grad_out, mask, grad_in, total);
    CUDA_CHECK(cudaGetLastError());
}

void launch_avgpool2d_forward(const float* x, float* out,
                              int N, int C, int H, int W,
                              int kH, int kW, int sH, int sW, int pH, int pW)
{
    NVTX_RANGE("avgpool2d_fwd");
    int outH = (H + 2*pH - kH) / sH + 1;
    int outW = (W + 2*pW - kW) / sW + 1;
    dim3 block(256), grid((outH*outW + 255) / 256, N*C);
    avgpool2d_fwd_kernel<<<grid, block>>>(x, out, N, C, H, W, kH, kW, sH, sW, pH, pW, outH, outW);
    CUDA_CHECK(cudaGetLastError());
}

void launch_avgpool2d_backward(const float* grad_out, float* grad_in,
                               int N, int C, int H, int W,
                               int outH, int outW,
                               int kH, int kW, int sH, int sW, int pH, int pW)
{
    NVTX_RANGE("avgpool2d_bwd");
    dim3 block(256), grid((outH*outW + 255) / 256, N*C);
    avgpool2d_bwd_kernel<<<grid, block>>>(grad_out, grad_in, N, C, H, W, outH, outW, kH, kW, sH, sW, pH, pW);
    CUDA_CHECK(cudaGetLastError());
}

// ─── add_bias helpers ─────────────────────────────────────────────────────────
__global__ void add_bias_kernel(float* out, const float* bias, int M, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i < M && j < N) out[i * N + j] += bias[j];
}

__global__ void add_bias_nchw_kernel(float* out, const float* bias,
                                     int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int c = (idx / (H * W)) % C;
    out[idx] += bias[c];
}

void launch_add_bias(float* out, const float* bias, int M, int N) {
    dim3 block(256), grid((N + 255) / 256, M);
    add_bias_kernel<<<grid, block>>>(out, bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_bias_nchw(float* out, const float* bias,
                          int N, int C, int H, int W) {
    int total = N * C * H * W;
    add_bias_nchw_kernel<<<(total + 255) / 256, 256>>>(out, bias, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

// ─── MaxPool2d layer ──────────────────────────────────────────────────────────
namespace layers {

MaxPool2d::MaxPool2d(int kH, int kW, int sH, int sW,
                     int pH, int pW, Device device)
    : kH_(kH), kW_(kW),
      stride_h_(sH < 0 ? kH : sH), stride_w_(sW < 0 ? kW : sW),
      pad_h_(pH), pad_w_(pW), device_(device) {}

Tensor MaxPool2d::forward(const Tensor& x) {
    NVTX_RANGE("MaxPool2d::forward");
    N_ = (int)x.shape()[0]; C_ = (int)x.shape()[1];
    H_ = (int)x.shape()[2]; W_ = (int)x.shape()[3];
    outH_ = output_h(H_); outW_ = output_w(W_);

    Tensor out  = Tensor::zeros({N_, C_, outH_, outW_}, DType::Float32, device_);
    mask_       = Tensor({N_, C_, outH_, outW_}, DType::Int32,   device_);

    kernels::launch_maxpool2d_forward(
        x.data_ptr<float>(), out.data_ptr<float>(), mask_.data_ptr<int32_t>(),
        N_, C_, H_, W_, kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_);
    return out;
}

Tensor MaxPool2d::backward(const Tensor& grad_out) {
    NVTX_RANGE("MaxPool2d::backward");
    Tensor grad_in = Tensor::zeros({N_, C_, H_, W_}, DType::Float32, device_);
    kernels::launch_maxpool2d_backward(
        grad_out.data_ptr<float>(), mask_.data_ptr<int32_t>(),
        grad_in.data_ptr<float>(), N_, C_, H_, W_, outH_, outW_);
    return grad_in;
}

// ─── AvgPool2d layer ──────────────────────────────────────────────────────────
AvgPool2d::AvgPool2d(int kH, int kW, int sH, int sW,
                     int pH, int pW, Device device)
    : kH_(kH), kW_(kW),
      stride_h_(sH < 0 ? kH : sH), stride_w_(sW < 0 ? kW : sW),
      pad_h_(pH), pad_w_(pW), device_(device) {}

Tensor AvgPool2d::forward(const Tensor& x) {
    N_=(int)x.shape()[0]; C_=(int)x.shape()[1];
    H_=(int)x.shape()[2]; W_=(int)x.shape()[3];
    outH_ = output_h(H_); outW_ = output_w(W_);
    Tensor out = Tensor::zeros({N_, C_, outH_, outW_}, DType::Float32, device_);
    kernels::launch_avgpool2d_forward(
        x.data_ptr<float>(), out.data_ptr<float>(),
        N_, C_, H_, W_, kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_);
    return out;
}

Tensor AvgPool2d::backward(const Tensor& grad_out) {
    Tensor grad_in = Tensor::zeros({N_, C_, H_, W_}, DType::Float32, device_);
    kernels::launch_avgpool2d_backward(
        grad_out.data_ptr<float>(), grad_in.data_ptr<float>(),
        N_, C_, H_, W_, outH_, outW_, kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_);
    return grad_in;
}

} // namespace layers
} // namespace minitensor
