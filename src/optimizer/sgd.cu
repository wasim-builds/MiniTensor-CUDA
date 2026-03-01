/**
 * @file sgd.cu
 * @brief SGD with momentum CUDA update kernel and host optimizer.
 */

#include "minitensor/optimizer.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <cstring>

namespace minitensor {

// ─── SGD update kernel ───────────────────────────────────────────────────────
__global__ void sgd_update_kernel(
    float* __restrict__ param,
    float* __restrict__ vel,
    const float* __restrict__ grad,
    int n, float lr, float momentum, float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i] + weight_decay * param[i];
    vel[i]  = momentum * vel[i] + g;
    param[i] -= lr * vel[i];
}

namespace kernels {

void launch_sgd_update(float* param, float* velocity, const float* grad,
                       int n, float lr, float momentum, float weight_decay)
{
    NVTX_RANGE("sgd_update");
    int block = 256;
    int grid  = (n + block - 1) / block;
    sgd_update_kernel<<<grid, block>>>(param, velocity, grad, n, lr, momentum, weight_decay);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels

// ─── SGD class ────────────────────────────────────────────────────────────────
void Optimizer::zero_grad() {
    for (auto& pg : params_) {
        if (!pg.grad) continue;
        int64_t n = pg.param->numel();
        if (pg.param->device() == Device::CUDA)
            CUDA_CHECK(cudaMemset(pg.grad->data_ptr(), 0, n * sizeof(float)));
        else
            std::memset(pg.grad->data_ptr(), 0, n * sizeof(float));
    }
}

SGD::SGD(std::vector<ParamGroup> params, float lr, float momentum,
         float weight_decay, bool nesterov)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum),
      weight_decay_(weight_decay), nesterov_(nesterov)
{
    for (auto& pg : params_) {
        velocities_.push_back(Tensor::zeros(pg.param->shape(),
                                            pg.param->dtype(),
                                            pg.param->device()));
    }
}

void SGD::step() {
    NVTX_RANGE("SGD::step");
    ++step_;
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& pg  = params_[i];
        auto& vel = velocities_[i];
        int   n   = static_cast<int>(pg.param->numel());

        kernels::launch_sgd_update(
            pg.param->data_ptr<float>(),
            vel.data_ptr<float>(),
            pg.grad->data_ptr<float>(),
            n, lr_, momentum_, weight_decay_);
    }
}

} // namespace minitensor
