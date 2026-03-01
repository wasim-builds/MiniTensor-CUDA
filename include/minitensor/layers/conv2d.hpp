/**
 * @file conv2d.hpp
 * @brief 2D Convolutional layer using im2col + GEMM strategy.
 *
 * Input format: NCHW  (batch, channels, height, width)
 * Weight format: (out_channels, in_channels/groups, kH, kW)
 */
#pragma once

#include "minitensor/tensor.hpp"
#include <string>

namespace minitensor {
namespace layers {

class Conv2d {
public:
    /**
     * @param in_channels   Number of input channels
     * @param out_channels  Number of output (filter) channels
     * @param kH            Kernel height
     * @param kW            Kernel width
     * @param stride_h      Vertical stride (default 1)
     * @param stride_w      Horizontal stride (default 1)
     * @param pad_h         Vertical padding (default 0)
     * @param pad_w         Horizontal padding (default 0)
     * @param dilation_h    Vertical dilation (default 1)
     * @param dilation_w    Horizontal dilation (default 1)
     * @param bias          Include bias term (default true)
     * @param device        Compute device
     */
    Conv2d(int in_channels, int out_channels,
           int kH, int kW,
           int stride_h = 1, int stride_w = 1,
           int pad_h    = 0, int pad_w    = 0,
           int dilation_h = 1, int dilation_w = 1,
           bool bias = true,
           Device device = Device::CUDA);

    /**
     * @brief Forward pass.
     * @param x  Input tensor (N, C_in, H, W)
     * @return   Output tensor (N, C_out, H_out, W_out) where:
     *           H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1
     */
    Tensor forward(const Tensor& x);

    /**
     * @brief Backward pass.
     * @param grad_out  Gradient from above of shape (N, C_out, H_out, W_out)
     * @return          Gradient w.r.t. input x of shape (N, C_in, H, W)
     *
     * Also stores grad_weight into weight_.grad and grad_bias into bias_.grad.
     */
    Tensor backward(const Tensor& grad_out);

    // ── Parameters ───────────────────────────────────────────────────────
    Tensor& weight()      { return weight_; }
    Tensor& bias_tensor() { return bias_; }
    bool    has_bias()    const { return use_bias_; }

    /** @brief Kaiming uniform initialization. */
    void reset_parameters();

    // ── Shape helpers ─────────────────────────────────────────────────────
    int output_h(int H) const {
        return (H + 2*pad_h_ - dilation_h_*(kH_-1) - 1) / stride_h_ + 1;
    }
    int output_w(int W) const {
        return (W + 2*pad_w_ - dilation_w_*(kW_-1) - 1) / stride_w_ + 1;
    }

    std::string name() const;

private:
    int in_ch_, out_ch_;
    int kH_,  kW_;
    int stride_h_, stride_w_;
    int pad_h_,    pad_w_;
    int dilation_h_, dilation_w_;
    bool use_bias_;
    Device device_;

    Tensor weight_;  // (out_ch, in_ch, kH, kW)
    Tensor bias_;    // (out_ch,)

    // Saved for backward
    Tensor input_cache_;  // (N, C, H, W)
    Tensor col_cache_;    // im2col output
    int    N_cache_, C_cache_, H_cache_, W_cache_;
};

} // namespace layers
} // namespace minitensor
