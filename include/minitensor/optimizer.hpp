/**
 * @file optimizer.hpp
 * @brief Optimizer base class and parameter registration.
 */
#pragma once

#include "minitensor/tensor.hpp"
#include <vector>
#include <memory>

namespace minitensor {

// ─── Parameter group ─────────────────────────────────────────────────────────
struct ParamGroup {
    Tensor* param  = nullptr;  ///< Parameter tensor (device)
    Tensor* grad   = nullptr;  ///< Gradient tensor (device)
    std::string name;          ///< Optional debug label
};

// ─── Optimizer base ───────────────────────────────────────────────────────────
class Optimizer {
public:
    explicit Optimizer(std::vector<ParamGroup> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;

    /** @brief Performs one optimization step using current gradients. */
    virtual void step() = 0;

    /** @brief Zeros all gradients. */
    void zero_grad();

    int step_count() const { return step_; }

protected:
    std::vector<ParamGroup> params_;
    int step_ = 0;
};

// ─── SGD ─────────────────────────────────────────────────────────────────────
/**
 * @brief Stochastic Gradient Descent with optional momentum and L2 regularization.
 *
 * Update rule:
 *   velocity[t] = momentum * velocity[t-1] + grad[t] + weight_decay * param
 *   param[t]   -= lr * velocity[t]
 */
class SGD : public Optimizer {
public:
    /**
     * @param params       List of parameter groups
     * @param lr           Learning rate
     * @param momentum     Momentum factor (0 = vanilla SGD)
     * @param weight_decay L2 regularization coefficient
     * @param nesterov     Enable Nesterov momentum
     */
    SGD(std::vector<ParamGroup> params,
        float lr           = 0.01f,
        float momentum     = 0.9f,
        float weight_decay = 0.0f,
        bool  nesterov     = false);

    void step() override;

private:
    float lr_;
    float momentum_;
    float weight_decay_;
    bool  nesterov_;
    std::vector<Tensor> velocities_;  ///< Per-parameter velocity buffers
};

// ─── Adam ─────────────────────────────────────────────────────────────────────
/**
 * @brief Adam optimizer — Adaptive Moment Estimation.
 *
 * Update rule (Kingma & Ba 2015):
 *   m[t] = beta1 * m[t-1] + (1-beta1) * grad
 *   v[t] = beta2 * v[t-1] + (1-beta2) * grad^2
 *   m_hat = m[t] / (1 - beta1^t)
 *   v_hat = v[t] / (1 - beta2^t)
 *   param -= lr * m_hat / (sqrt(v_hat) + eps)
 */
class Adam : public Optimizer {
public:
    /**
     * @param params       List of parameter groups
     * @param lr           Learning rate (default 1e-3)
     * @param beta1        First moment decay (default 0.9)
     * @param beta2        Second moment decay (default 0.999)
     * @param eps          Numerical stability (default 1e-8)
     * @param weight_decay AdamW-style L2 regularization
     */
    Adam(std::vector<ParamGroup> params,
         float lr           = 1e-3f,
         float beta1        = 0.9f,
         float beta2        = 0.999f,
         float eps          = 1e-8f,
         float weight_decay = 0.0f);

    void step() override;

private:
    float lr_, beta1_, beta2_, eps_, weight_decay_;
    std::vector<Tensor> m_;  ///< First moment
    std::vector<Tensor> v_;  ///< Second moment
};

} // namespace minitensor
