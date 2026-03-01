/**
 * @file batchnorm.cu
 * @brief Batch Normalization 2D forward and backward CUDA kernels.
 *
 * Two-pass algorithm:
 *   Pass 1: compute per-channel mean and variance over (N,H,W)
 *   Pass 2: normalize, scale, shift; update running stats
 *
 * Backward:
 *   Computes grad_x, grad_gamma, grad_beta via the BN Jacobian.
 *   Reference: Ioffe & Szegedy (2015), Algorithm 1 backward.
 */

#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <math.h>

namespace minitensor {
namespace kernels {

static constexpr int BN_BLOCK = 256;

// ─── Pass 1: per-channel mean ─────────────────────────────────────────────────
__global__ void bn_mean_kernel(const float* __restrict__ x,
                               float* __restrict__ mean,
                               int N, int C, int H, int W) {
    int c = blockIdx.x;  // one block per channel
    if (c >= C) return;

    int spatial = N * H * W;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        int n = i / (H * W);
        int hw = i % (H * W);
        sum += x[((n * C + c) * H + hw / W) * W + hw % W];
    }

    // Block reduce
    __shared__ float smem[BN_BLOCK];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) mean[c] = smem[0] / static_cast<float>(spatial);
}

// ─── Pass 1b: per-channel variance ───────────────────────────────────────────
__global__ void bn_var_kernel(const float* __restrict__ x,
                              const float* __restrict__ mean,
                              float* __restrict__ var,
                              int N, int C, int H, int W) {
    int c = blockIdx.x;
    if (c >= C) return;

    int spatial = N * H * W;
    float m  = mean[c];
    float sq = 0.0f;
    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        int n = i / (H * W);
        int hw = i % (H * W);
        float d = x[((n * C + c) * H + hw / W) * W + hw % W] - m;
        sq += d * d;
    }

    __shared__ float smem[BN_BLOCK];
    smem[threadIdx.x] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) var[c] = smem[0] / static_cast<float>(spatial);
}

// ─── Pass 2: normalize + affine + running stats update ───────────────────────
__global__ void bn_normalize_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ run_mean,
    float* __restrict__ run_var,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float eps, float momentum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Decode NCHW index
    int w = idx % W; int tmp = idx / W;
    int h = tmp % H;     tmp /= H;
    int c = tmp % C;

    float m   = mean[c];
    float v   = var[c];
    float inv = rsqrtf(v + eps);
    float xhat = (x[idx] - m) * inv;
    out[idx] = gamma[c] * xhat + beta[c];

    // Update running stats from first thread per channel
    if (h == 0 && w == 0 && (idx / (H * W)) % N == 0) {
        run_mean[c] = (1.0f - momentum) * run_mean[c] + momentum * m;
        run_var[c]  = (1.0f - momentum) * run_var[c]  + momentum * v;
    }
}

// ─── Backward: grad_gamma, grad_beta ─────────────────────────────────────────
__global__ void bn_grad_gamma_beta_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int N, int C, int H, int W, float eps)
{
    int c = blockIdx.x;
    if (c >= C) return;

    int spatial = N * H * W;
    float gg = 0.0f, gb = 0.0f;
    float m = mean[c], inv = rsqrtf(var[c] + eps);

    for (int i = threadIdx.x; i < spatial; i += blockDim.x) {
        int n  = i / (H * W);
        int hw = i % (H * W);
        int full_idx = ((n * C + c) * H + hw / W) * W + hw % W;
        float xhat = (x[full_idx] - m) * inv;
        gg += grad_out[full_idx] * xhat;
        gb += grad_out[full_idx];
    }

    __shared__ float sgg[BN_BLOCK], sgb[BN_BLOCK];
    sgg[threadIdx.x] = gg; sgb[threadIdx.x] = gb;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sgg[threadIdx.x] += sgg[threadIdx.x + s];
            sgb[threadIdx.x] += sgb[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { grad_gamma[c] = sgg[0]; grad_beta[c] = sgb[0]; }
}

// ─── Backward: grad_x ────────────────────────────────────────────────────────
__global__ void bn_grad_x_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ grad_gamma,
    const float* __restrict__ grad_beta,
    float* __restrict__ grad_x,
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W; int tmp = idx / W;
    int h = tmp % H;     tmp /= H;
    int c = tmp % C;
    (void)h; (void)w;

    float m   = mean[c];
    float inv = rsqrtf(var[c] + eps);
    float xhat = (x[idx] - m) * inv;
    int spatial = N * H * W;

    // BN backward formula (simplified):
    // grad_x = gamma/sqrt(var+eps) * [grad_out - grad_beta/spatial - xhat*grad_gamma/spatial]
    grad_x[idx] = (gamma[c] * inv / spatial) *
                  (spatial * grad_out[idx]
                   - grad_beta[c]
                   - xhat * grad_gamma[c]);
}

// ─── Host launchers ───────────────────────────────────────────────────────────
void launch_batchnorm_forward(const float* x,
                              const float* gamma, const float* beta,
                              float* mean, float* var,
                              float* run_mean, float* run_var,
                              float* out,
                              int N, int C, int H, int W,
                              float eps, float momentum)
{
    NVTX_RANGE("batchnorm_fwd");

    // Pass 1a: mean
    bn_mean_kernel<<<C, BN_BLOCK>>>(x, mean, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());

    // Pass 1b: var
    bn_var_kernel<<<C, BN_BLOCK>>>(x, mean, var, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: normalize
    int total = N * C * H * W;
    int grid  = (total + BN_BLOCK - 1) / BN_BLOCK;
    bn_normalize_kernel<<<grid, BN_BLOCK>>>(
        x, gamma, beta, mean, var, run_mean, run_var, out,
        N, C, H, W, eps, momentum);
    CUDA_CHECK(cudaGetLastError());
}

void launch_batchnorm_backward(const float* grad_out,
                               const float* x,
                               const float* gamma,
                               const float* mean, const float* var,
                               float* grad_x,
                               float* grad_gamma, float* grad_beta,
                               int N, int C, int H, int W, float eps)
{
    NVTX_RANGE("batchnorm_bwd");

    bn_grad_gamma_beta_kernel<<<C, BN_BLOCK>>>(
        grad_out, x, mean, var, grad_gamma, grad_beta, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());

    int total = N * C * H * W;
    int grid  = (total + BN_BLOCK - 1) / BN_BLOCK;
    bn_grad_x_kernel<<<grid, BN_BLOCK>>>(
        grad_out, x, gamma, mean, var, grad_gamma, grad_beta,
        grad_x, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace minitensor
