/**
 * @file batchnorm.hpp
 * @brief Batch Normalization 2D layer (NCHW inputs).
 *
 * Normalizes per-channel across (N, H, W) dimensions.
 * Maintains running mean/variance for inference (eval) mode.
 *
 * Forward (train):
 *   mean[c]   = mean over (N,H,W)
 *   var[c]    = variance over (N,H,W)
 *   x_hat     = (x - mean) / sqrt(var + eps)
 *   out       = gamma * x_hat + beta
 *   running_mean = (1-m)*running_mean + m*mean
 *   running_var  = (1-m)*running_var  + m*var
 *
 * Forward (eval):
 *   x_hat = (x - running_mean) / sqrt(running_var + eps)
 *   out   = gamma * x_hat + beta
 */
#pragma once

#include "minitensor/tensor.hpp"

namespace minitensor {
namespace layers {

class BatchNorm2d {
public:
    /**
     * @param num_features  Number of channels C
     * @param eps           Numerical stability constant (default 1e-5)
     * @param momentum      EMA momentum for running stats (default 0.1)
     * @param affine        Learnable gamma/beta (default true)
     * @param device        Compute device
     */
    BatchNorm2d(int num_features,
                float eps      = 1e-5f,
                float momentum = 0.1f,
                bool  affine   = true,
                Device device  = Device::CUDA);

    /**
     * @brief Forward pass (train or eval mode).
     * @param x  Input of shape (N, C, H, W)
     * @return   Normalized output of same shape
     */
    Tensor forward(const Tensor& x);

    /**
     * @brief Backward pass.
     * @param grad_out  Upstream gradient (N, C, H, W)
     * @return          Gradient w.r.t. input x
     */
    Tensor backward(const Tensor& grad_out);

    // ── Mode ──────────────────────────────────────────────────────────────
    void train() { training_ = true;  }
    void eval()  { training_ = false; }
    bool is_training() const { return training_; }

    // ── Parameters ────────────────────────────────────────────────────────
    Tensor& gamma()       { return gamma_; }
    Tensor& beta()        { return beta_;  }
    const Tensor& running_mean() const { return running_mean_; }
    const Tensor& running_var()  const { return running_var_;  }

private:
    int    num_features_;
    float  eps_;
    float  momentum_;
    bool   affine_;
    bool   training_ = true;
    Device device_;

    Tensor gamma_;        // (C,) scale — requires_grad=true
    Tensor beta_;         // (C,) shift — requires_grad=true
    Tensor running_mean_; // (C,) — buffer, no grad
    Tensor running_var_;  // (C,) — buffer, no grad

    // Saved for backward
    Tensor input_cache_;  // original x
    Tensor mean_cache_;   // per-channel mean
    Tensor var_cache_;    // per-channel variance
    int    N_, H_, W_;    // spatial dims from last forward
};

} // namespace layers
} // namespace minitensor
