/**
 * @file activation.hpp
 * @brief Element-wise activation function layers.
 */
#pragma once

#include "minitensor/tensor.hpp"

namespace minitensor {
namespace layers {

// ─── Base ────────────────────────────────────────────────────────────────────
class ActivationLayer {
public:
    virtual ~ActivationLayer() = default;
    virtual Tensor forward(const Tensor& x)              = 0;
    virtual Tensor backward(const Tensor& grad_output)   = 0;
};

// ─── ReLU ────────────────────────────────────────────────────────────────────
class ReLU : public ActivationLayer {
public:
    Tensor forward(const Tensor& x)            override;
    Tensor backward(const Tensor& grad_output) override;
private:
    Tensor input_cache_;
};

// ─── Leaky ReLU ──────────────────────────────────────────────────────────────
class LeakyReLU : public ActivationLayer {
public:
    explicit LeakyReLU(float negative_slope = 0.01f)
        : negative_slope_(negative_slope) {}

    Tensor forward(const Tensor& x)            override;
    Tensor backward(const Tensor& grad_output) override;
private:
    float  negative_slope_;
    Tensor input_cache_;
};

// ─── Sigmoid ─────────────────────────────────────────────────────────────────
class Sigmoid : public ActivationLayer {
public:
    Tensor forward(const Tensor& x)            override;
    Tensor backward(const Tensor& grad_output) override;
private:
    Tensor output_cache_;  // σ(x) saved for backward
};

// ─── Tanh ────────────────────────────────────────────────────────────────────
class Tanh : public ActivationLayer {
public:
    Tensor forward(const Tensor& x)            override;
    Tensor backward(const Tensor& grad_output) override;
private:
    Tensor output_cache_;
};

} // namespace layers
} // namespace minitensor
