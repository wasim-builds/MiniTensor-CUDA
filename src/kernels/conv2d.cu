/**
 * @file conv2d.cu
 * @brief im2col + GEMM convolution kernel.
 *
 * Strategy:
 *   1. im2col: each input patch (C × kH × kW) becomes a column.
 *      Output col matrix is (C*kH*kW) × (N * outH * outW).
 *   2. GEMM: weight matrix (out_ch × C*kH*kW) × col → output.
 *
 * Backward:
 *   - grad_weight = grad_out_reshaped × col^T
 *   - grad_input  = weight^T × grad_out_reshaped → col2im
 *   - grad_bias   = sum over (N,H,W) of grad_out
 */

#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>

namespace minitensor {
namespace kernels {

// ─────────────────────────────────────────────────────────────────────────────
// im2col kernel
// Each thread maps one (c, kh, kw, oh, ow, n) element to output column matrix
// ─────────────────────────────────────────────────────────────────────────────
__global__ void im2col_kernel(
    const float* __restrict__ input,   // (N, C, H, W)
          float* __restrict__ col,     // (C*kH*kW, N*outH*outW)
    int N, int C, int H, int W,
    int kH, int kW,
    int outH, int outW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW)
{
    // Each thread handles one element of the col matrix
    const int col_size = N * outH * outW;
    const int row_size = C * kH * kW;

    int col_idx = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (row_idx >= row_size || col_idx >= col_size) return;

    // Decode row: (c, kh, kw)
    int tmp    = row_idx;
    int kw_idx = tmp % kW; tmp /= kW;
    int kh_idx = tmp % kH; tmp /= kH;
    int c_idx  = tmp;

    // Decode col: (n, oh, ow)
    tmp        = col_idx;
    int ow_idx = tmp % outW; tmp /= outW;
    int oh_idx = tmp % outH; tmp /= outH;
    int n_idx  = tmp;

    // Compute source position in input
    int h_in = oh_idx * sH - pH + kh_idx * dH;
    int w_in = ow_idx * sW - pW + kw_idx * dW;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((n_idx * C + c_idx) * H + h_in) * W + w_in];
    }

    col[row_idx * col_size + col_idx] = val;
}

// ─────────────────────────────────────────────────────────────────────────────
// col2im kernel — accumulates gradients back into input tensor
// ─────────────────────────────────────────────────────────────────────────────
__global__ void col2im_kernel(
    const float* __restrict__ col,    // (C*kH*kW, N*outH*outW)
          float* __restrict__ input,  // (N, C, H, W)
    int N, int C, int H, int W,
    int kH, int kW,
    int outH, int outW,
    int sH, int sW,
    int pH, int pW,
    int dH, int dW)
{
    const int col_size = N * outH * outW;
    const int row_size = C * kH * kW;

    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (row_idx >= row_size || col_idx >= col_size) return;

    int tmp    = row_idx;
    int kw_idx = tmp % kW; tmp /= kW;
    int kh_idx = tmp % kH; tmp /= kH;
    int c_idx  = tmp;

    tmp        = col_idx;
    int ow_idx = tmp % outW; tmp /= outW;
    int oh_idx = tmp % outH; tmp /= outH;
    int n_idx  = tmp;

    int h_in = oh_idx * sH - pH + kh_idx * dH;
    int w_in = ow_idx * sW - pW + kw_idx * dW;

    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        atomicAdd(
            &input[((n_idx * C + c_idx) * H + h_in) * W + w_in],
            col[row_idx * col_size + col_idx]
        );
    }
}

// ─── Host launchers ───────────────────────────────────────────────────────────
void launch_im2col(const float* input, float* col,
                   int N, int C, int H, int W,
                   int kH, int kW,
                   int sH, int sW,
                   int pH, int pW,
                   int dH, int dW)
{
    NVTX_RANGE("im2col");

    int outH = (H + 2*pH - dH*(kH-1) - 1) / sH + 1;
    int outW = (W + 2*pW - dW*(kW-1) - 1) / sW + 1;

    int col_size = N * outH * outW;
    int row_size = C * kH * kW;

    // 2D grid: x=col dimension, y=row dimension
    dim3 block(16, 16);
    dim3 grid((col_size + block.x - 1) / block.x,
              (row_size + block.y - 1) / block.y);

    im2col_kernel<<<grid, block>>>(
        input, col, N, C, H, W, kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
    CUDA_CHECK(cudaGetLastError());
}

void launch_col2im(const float* col, float* input,
                   int N, int C, int H, int W,
                   int kH, int kW,
                   int sH, int sW,
                   int pH, int pW,
                   int dH, int dW)
{
    NVTX_RANGE("col2im");

    int outH = (H + 2*pH - dH*(kH-1) - 1) / sH + 1;
    int outW = (W + 2*pW - dW*(kW-1) - 1) / sW + 1;

    int col_size = N * outH * outW;
    int row_size = C * kH * kW;

    dim3 block(16, 16);
    dim3 grid((col_size + block.x - 1) / block.x,
              (row_size + block.y - 1) / block.y);

    col2im_kernel<<<grid, block>>>(
        col, input, N, C, H, W, kH, kW, outH, outW, sH, sW, pH, pW, dH, dW);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernels
} // namespace minitensor
