/**
 * @file adam.cu
 * @brief Adam optimizer CUDA update kernel.
 */

#include "minitensor/optimizer.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace minitensor {

// ─── Adam update kernel ───────────────────────────────────────────────────────
__global__ void adam_update_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    int n,
    float lr, float beta1, float beta2,
    float eps, float weight_decay,
    float bias_corr1, float bias_corr2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i] + weight_decay * param[i];

    // Update biased moment estimates
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

    // Bias-corrected estimates
    float m_hat = m[i] / bias_corr1;
    float v_hat = v[i] / bias_corr2;

    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

namespace kernels {

void launch_adam_update(float* param,
                        float* m, float* v,
                        const float* grad,
                        int n,
                        float lr, float beta1, float beta2,
                        float eps, float weight_decay,
                        int step)
{
    NVTX_RANGE("adam_update");
    float bias_corr1 = 1.0f - powf(beta1, static_cast<float>(step));
    float bias_corr2 = 1.0f - powf(beta2, static_cast<float>(step));

    int block = 256;
    int grid  = (n + block - 1) / block;
    adam_update_kernel<<<grid, block>>>(
        param, m, v, grad, n,
        lr, beta1, beta2, eps, weight_decay,
        bias_corr1, bias_corr2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

// ─── Adam class ───────────────────────────────────────────────────────────────
Adam::Adam(std::vector<ParamGroup> params,
           float lr, float beta1, float beta2, float eps, float weight_decay)
    : Optimizer(std::move(params)), lr_(lr), beta1_(beta1),
      beta2_(beta2), eps_(eps), weight_decay_(weight_decay)
{
    for (auto& pg : params_) {
        m_.push_back(Tensor::zeros(pg.param->shape(), pg.param->dtype(), pg.param->device()));
        v_.push_back(Tensor::zeros(pg.param->shape(), pg.param->dtype(), pg.param->device()));
    }
}

void Adam::step() {
    NVTX_RANGE("Adam::step");
    ++step_;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& pg = params_[i];
        int   n  = static_cast<int>(pg.param->numel());

        kernels::launch_adam_update(
            pg.param->data_ptr<float>(),
            m_[i].data_ptr<float>(),
            v_[i].data_ptr<float>(),
            pg.grad->data_ptr<float>(),
            n, lr_, beta1_, beta2_, eps_, weight_decay_, step_);
    }
}

} // namespace minitensor
