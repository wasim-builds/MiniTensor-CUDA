/**
 * @file reduce.cu
 * @brief Warp-level and block-level reduction kernels.
 *
 * Implements sum, mean, max reductions using cooperative groups
 * (warp shuffle + shared-memory tree reduction).
 *
 * Warp shuffle intrinsics (__shfl_down_sync) eliminate the need for
 * shared memory within a warp (32 threads), reducing latency.
 */

#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cfloat>

namespace cg = cooperative_groups;
namespace minitensor {
namespace kernels {

static constexpr int REDUCE_BLOCK = 256;

// ─── Warp reduce helpers ──────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    return val;
}

// ─── Block-level sum reduction → single per-block value ──────────────────────
__global__ void reduce_sum_kernel(const float* __restrict__ x,
                                  float* __restrict__ out, int n) {
    __shared__ float smem[32]; // one slot per warp

    cg::thread_block block = cg::this_thread_block();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (tid < n) ? x[tid] : 0.0f;

    // Warp reduce
    val = warp_reduce_sum(val);

    // Write each warp's result to shared mem
    if (threadIdx.x % 32 == 0)
        smem[threadIdx.x / 32] = val;
    block.sync();

    // Final reduction among warps (first warp only)
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    if (threadIdx.x / 32 == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0)
        atomicAdd(out, val);
}

// ─── Axis-reduce for 2D matrix ────────────────────────────────────────────────
// axis=0: sum each column (output shape: N)
// axis=1: sum each row   (output shape: M)
__global__ void sum_axis_kernel(const float* __restrict__ x, float* __restrict__ out,
                                int M, int N, int axis) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (axis == 0) {
        // Sum over rows → output[j] = sum_i x[i][j]
        if (idx < N) {
            float s = 0.0f;
            for (int i = 0; i < M; ++i) s += x[i * N + idx];
            out[idx] = s;
        }
    } else {
        // Sum over cols → output[i] = sum_j x[i][j]
        if (idx < M) {
            float s = 0.0f;
            for (int j = 0; j < N; ++j) s += x[idx * N + j];
            out[idx] = s;
        }
    }
}

// ─── Host launchers ───────────────────────────────────────────────────────────
float launch_sum(const float* x, int n) {
    NVTX_RANGE("reduce_sum");

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    int grid_size = (n + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    reduce_sum_kernel<<<grid_size, REDUCE_BLOCK>>>(x, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
    return result;
}

float launch_mean(const float* x, int n) {
    return launch_sum(x, n) / static_cast<float>(n);
}

void launch_sum_axis(const float* x, float* out, int M, int N, int axis) {
    NVTX_RANGE("sum_axis");
    int len = (axis == 0) ? N : M;
    int grid_size = (len + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    sum_axis_kernel<<<grid_size, REDUCE_BLOCK>>>(x, out, M, N, axis);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace minitensor
