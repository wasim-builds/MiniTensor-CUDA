/**
 * @file pooling.hpp
 * @brief MaxPool2D and AvgPool2D layers.
 */
#pragma once

#include "minitensor/tensor.hpp"
#include <cstdint>

namespace minitensor {
namespace layers {

// ─── MaxPool2D ────────────────────────────────────────────────────────────────
class MaxPool2d {
public:
    MaxPool2d(int kH, int kW, int stride_h = -1, int stride_w = -1,
              int pad_h = 0, int pad_w = 0, Device device = Device::CUDA);

    /** @param x  Input (N, C, H, W)  @return Output (N, C, H_out, W_out) */
    Tensor forward(const Tensor& x);

    /** @param grad_out Upstream gradient  @return Gradient w.r.t. x */
    Tensor backward(const Tensor& grad_out);

    int output_h(int H) const { return (H + 2*pad_h_ - kH_) / stride_h_ + 1; }
    int output_w(int W) const { return (W + 2*pad_w_ - kW_) / stride_w_ + 1; }

private:
    int kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_;
    Device device_;

    Tensor    mask_;  // argmax indices (int32) for backward
    int N_, C_, H_, W_, outH_, outW_;
};

// ─── AvgPool2D ────────────────────────────────────────────────────────────────
class AvgPool2d {
public:
    AvgPool2d(int kH, int kW, int stride_h = -1, int stride_w = -1,
              int pad_h = 0, int pad_w = 0, Device device = Device::CUDA);

    Tensor forward(const Tensor& x);
    Tensor backward(const Tensor& grad_out);

    int output_h(int H) const { return (H + 2*pad_h_ - kH_) / stride_h_ + 1; }
    int output_w(int W) const { return (W + 2*pad_w_ - kW_) / stride_w_ + 1; }

private:
    int kH_, kW_, stride_h_, stride_w_, pad_h_, pad_w_;
    Device device_;
    int N_, C_, H_, W_, outH_, outW_;
};

} // namespace layers
} // namespace minitensor
