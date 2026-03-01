/**
 * @file linear.hpp
 * @brief Fully-connected (dense) linear layer.
 *
 * Implements: out = x * W^T + b
 *   x:   (N, in_features)
 *   W:   (out_features, in_features)
 *   b:   (out_features,)
 *   out: (N, out_features)
 */
#pragma once

#include "minitensor/tensor.hpp"
#include <string>

namespace minitensor {
namespace layers {

class Linear {
public:
    /**
     * @param in_features   Input feature dimension
     * @param out_features  Output feature dimension
     * @param bias          Whether to include a bias term (default true)
     * @param device        Target compute device
     */
    Linear(int in_features, int out_features,
           bool bias = true, Device device = Device::CUDA);

    // ── Forward ──────────────────────────────────────────────────────────
    /**
     * @brief Forward pass: out = x*W^T + b
     * @param x  Input tensor of shape (N, in_features)
     * @return   Output tensor of shape (N, out_features)
     */
    Tensor forward(const Tensor& x);

    // ── Backward ─────────────────────────────────────────────────────────
    /**
     * @brief Backward pass — computes grad w.r.t. inputs, weights, and bias.
     * @param grad_output  Upstream gradient of shape (N, out_features)
     * @return             Gradient w.r.t. input x of shape (N, in_features)
     *
     * Stores grad_weight (out_features, in_features) into weight_.grad
     * and grad_bias (out_features,) into bias_.grad.
     */
    Tensor backward(const Tensor& grad_output);

    // ── Parameters ───────────────────────────────────────────────────────
    Tensor& weight()       { return weight_; }
    Tensor& bias_tensor()  { return bias_; }
    bool    has_bias()     const { return use_bias_; }

    /** @brief Kaiming uniform weight initialization (for ReLU networks). */
    void reset_parameters();

    int in_features()  const { return in_features_; }
    int out_features() const { return out_features_; }

    std::string name() const { return "Linear(" +
        std::to_string(in_features_) + "→" +
        std::to_string(out_features_) + ")"; }

private:
    int     in_features_;
    int     out_features_;
    bool    use_bias_;
    Device  device_;

    Tensor  weight_;   // (out_features, in_features)
    Tensor  bias_;     // (out_features,)

    // Saved for backward
    Tensor  input_cache_;
};

} // namespace layers
} // namespace minitensor
