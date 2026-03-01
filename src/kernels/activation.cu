/**
 * @file activation.cu
 * @brief Element-wise activation CUDA kernels (forward + backward).
 *
 * Kernel design: 1D thread layout, each thread processes one element.
 * Grid/block sized to maximize occupancy for small element-wise ops.
 */

#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace minitensor {
namespace kernels {

static constexpr int BLOCK_SIZE = 256;

// ─── ReLU ────────────────────────────────────────────────────────────────────
__global__ void relu_fwd_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void relu_bwd_kernel(const float* x, const float* gout,
                                float* gin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gin[i] = x[i] > 0.0f ? gout[i] : 0.0f;
}

// ─── Leaky ReLU ──────────────────────────────────────────────────────────────
__global__ void leaky_relu_fwd_kernel(const float* x, float* out,
                                      int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] > 0.0f ? x[i] : alpha * x[i];
}

__global__ void leaky_relu_bwd_kernel(const float* x, const float* gout,
                                      float* gin, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gin[i] = x[i] > 0.0f ? gout[i] : alpha * gout[i];
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────
__global__ void sigmoid_fwd_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f / (1.0f + expf(-x[i]));
}

// Backward uses saved output σ(x): grad_in = grad_out * σ(x) * (1 - σ(x))
__global__ void sigmoid_bwd_kernel(const float* out, const float* gout,
                                   float* gin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gin[i] = gout[i] * out[i] * (1.0f - out[i]);
}

// ─── Tanh ────────────────────────────────────────────────────────────────────
__global__ void tanh_fwd_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(x[i]);
}

// Backward: grad_in = grad_out * (1 - tanh²(x)) = grad_out * (1 - out²)
__global__ void tanh_bwd_kernel(const float* out, const float* gout,
                                float* gin, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gin[i] = gout[i] * (1.0f - out[i] * out[i]);
}

// ─── Host launchers ───────────────────────────────────────────────────────────
static inline int grid(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

void launch_relu_forward(const float* x, float* out, int n) {
    NVTX_RANGE("relu_fwd");
    relu_fwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, out, n);
    CUDA_CHECK(cudaGetLastError());
}
void launch_relu_backward(const float* x, const float* gout,
                          float* gin, int n) {
    NVTX_RANGE("relu_bwd");
    relu_bwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, gout, gin, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_leaky_relu_forward(const float* x, float* out,
                               int n, float alpha) {
    NVTX_RANGE("leaky_relu_fwd");
    leaky_relu_fwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, out, n, alpha);
    CUDA_CHECK(cudaGetLastError());
}
void launch_leaky_relu_backward(const float* x, const float* gout,
                                float* gin, int n, float alpha) {
    NVTX_RANGE("leaky_relu_bwd");
    leaky_relu_bwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, gout, gin, n, alpha);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sigmoid_forward(const float* x, float* out, int n) {
    NVTX_RANGE("sigmoid_fwd");
    sigmoid_fwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, out, n);
    CUDA_CHECK(cudaGetLastError());
}
void launch_sigmoid_backward(const float* out, const float* gout,
                             float* gin, int n) {
    NVTX_RANGE("sigmoid_bwd");
    sigmoid_bwd_kernel<<<grid(n), BLOCK_SIZE>>>(out, gout, gin, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tanh_forward(const float* x, float* out, int n) {
    NVTX_RANGE("tanh_fwd");
    tanh_fwd_kernel<<<grid(n), BLOCK_SIZE>>>(x, out, n);
    CUDA_CHECK(cudaGetLastError());
}
void launch_tanh_backward(const float* out, const float* gout,
                          float* gin, int n) {
    NVTX_RANGE("tanh_bwd");
    tanh_bwd_kernel<<<grid(n), BLOCK_SIZE>>>(out, gout, gin, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace minitensor
