/**
 * @file kernels.hpp
 * @brief CUDA kernel launcher declarations.
 *
 * All functions are host-callable wrappers around __global__ kernels.
 * They handle grid/block configuration automatically based on problem size.
 *
 * Convention:
 *  - Matrices are stored row-major (C-style)
 *  - A is (M×K), B is (K×N), C is (M×N)
 *  - Convolution: NCHW format
 */
#pragma once

#include <cstdint>

namespace minitensor {
namespace kernels {

// ─── GEMM ────────────────────────────────────────────────────────────────────
/**
 * @brief Tiled shared-memory GEMM: C = alpha*A*B + beta*C
 *
 * Uses TILE_SIZE×TILE_SIZE thread blocks with shared memory tiling for
 * optimal memory coalescing and L1 reuse.
 *
 * @param A      Device pointer to input matrix (M×K)
 * @param B      Device pointer to input matrix (K×N)
 * @param C      Device pointer to output matrix (M×N)
 * @param M      Rows of A / C
 * @param N      Cols of B / C
 * @param K      Cols of A / Rows of B
 * @param alpha  Scalar multiplier for A*B
 * @param beta   Scalar multiplier for existing C (0.0 to overwrite)
 */
void launch_matmul(const float* A, const float* B, float* C,
                   int M, int N, int K,
                   float alpha = 1.0f, float beta = 0.0f);

/**
 * @brief Batched GEMM: C[b] = A[b]*B[b] for b in [0, batch).
 * Useful for grouped convolution and attention.
 */
void launch_batched_matmul(const float* A, const float* B, float* C,
                           int batch, int M, int N, int K);

// ─── Convolution helpers ──────────────────────────────────────────────────────
/**
 * @brief im2col: unfolds input tensor into column matrix for GEMM-based conv.
 *
 * For an input (N, C, H, W) with kernel (kH, kW), stride (sH, sW),
 * padding (pH, pW), produces a matrix of shape:
 *   (C * kH * kW) × (N * outH * outW)
 *
 * @param input    Device pointer to NCHW input (N,C,H,W)
 * @param col      Device pointer to output columns
 * @param N        Batch size
 * @param C        Input channels
 * @param H, W     Spatial dimensions
 * @param kH, kW   Kernel height/width
 * @param sH, sW   Stride height/width
 * @param pH, pW   Padding height/width
 * @param dH, dW   Dilation height/width
 */
void launch_im2col(const float* input, float* col,
                   int N, int C, int H, int W,
                   int kH, int kW,
                   int sH, int sW,
                   int pH, int pW,
                   int dH, int dW);

/**
 * @brief col2im: reverses im2col for backward pass.
 * Accumulates (adds) into output — caller must zero it first.
 */
void launch_col2im(const float* col, float* input,
                   int N, int C, int H, int W,
                   int kH, int kW,
                   int sH, int sW,
                   int pH, int pW,
                   int dH, int dW);

// ─── Activations ─────────────────────────────────────────────────────────────
/** @brief Element-wise ReLU: out = max(0, x). */
void launch_relu_forward(const float* x, float* out, int n);

/** @brief ReLU backward: grad_in = grad_out * (x > 0). */
void launch_relu_backward(const float* x, const float* grad_out,
                          float* grad_in, int n);

/** @brief Leaky-ReLU forward: out = x > 0 ? x : alpha*x. */
void launch_leaky_relu_forward(const float* x, float* out, int n, float alpha = 0.01f);
void launch_leaky_relu_backward(const float* x, const float* grad_out,
                                float* grad_in, int n, float alpha = 0.01f);

/** @brief Sigmoid forward/backward. */
void launch_sigmoid_forward(const float* x, float* out, int n);
void launch_sigmoid_backward(const float* out, const float* grad_out,
                             float* grad_in, int n);

/** @brief Tanh forward/backward. */
void launch_tanh_forward(const float* x, float* out, int n);
void launch_tanh_backward(const float* out, const float* grad_out,
                          float* grad_in, int n);

// ─── Reductions ───────────────────────────────────────────────────────────────
/** @brief Sum all elements → scalar on device. */
float launch_sum(const float* x, int n);

/** @brief Mean of all elements → scalar. */
float launch_mean(const float* x, int n);

/** @brief Sum along axis for a 2D matrix (M×N). axis: 0=row-wise, 1=col-wise. */
void  launch_sum_axis(const float* x, float* out, int M, int N, int axis);

// ─── Batch Normalization ──────────────────────────────────────────────────────
/**
 * @brief BatchNorm2D forward (training mode).
 *
 * @param x        Input (N, C, H, W) on device
 * @param gamma    Scale parameter (C,)
 * @param beta     Shift parameter (C,)
 * @param mean     Per-channel mean output (C,)
 * @param var      Per-channel variance output (C,)
 * @param run_mean Running mean (updated in-place)
 * @param run_var  Running variance (updated in-place)
 * @param out      Output (N, C, H, W)
 * @param eps      Numerical stability epsilon
 * @param momentum EMA momentum for running stats
 */
void launch_batchnorm_forward(const float* x,
                              const float* gamma, const float* beta,
                              float* mean, float* var,
                              float* run_mean, float* run_var,
                              float* out,
                              int N, int C, int H, int W,
                              float eps = 1e-5f, float momentum = 0.1f);

/** @brief BatchNorm2D backward: computes grad_x, grad_gamma, grad_beta. */
void launch_batchnorm_backward(const float* grad_out,
                               const float* x,
                               const float* gamma,
                               const float* mean, const float* var,
                               float* grad_x,
                               float* grad_gamma, float* grad_beta,
                               int N, int C, int H, int W,
                               float eps = 1e-5f);

// ─── Pooling ───────────────────────────────────────────────────────────────────
/** @brief MaxPool2D forward; mask stores argmax indices for backward. */
void launch_maxpool2d_forward(const float* x, float* out, int32_t* mask,
                              int N, int C, int H, int W,
                              int kH, int kW, int sH, int sW, int pH, int pW);

/** @brief MaxPool2D backward using stored mask. */
void launch_maxpool2d_backward(const float* grad_out, const int32_t* mask,
                               float* grad_in,
                               int N, int C, int H, int W,
                               int outH, int outW);

/** @brief AvgPool2D forward and backward. */
void launch_avgpool2d_forward(const float* x, float* out,
                              int N, int C, int H, int W,
                              int kH, int kW, int sH, int sW, int pH, int pW);
void launch_avgpool2d_backward(const float* grad_out, float* grad_in,
                               int N, int C, int H, int W,
                               int outH, int outW,
                               int kH, int kW, int sH, int sW, int pH, int pW);

// ─── Element-wise SGD / Adam updates ─────────────────────────────────────────
/** @brief In-place SGD with momentum: v = m*v + g; p -= lr*v */
void launch_sgd_update(float* param, float* velocity, const float* grad,
                       int n, float lr, float momentum, float weight_decay);

/** @brief In-place Adam update. */
void launch_adam_update(float* param,
                        float* m, float* v,
                        const float* grad,
                        int n,
                        float lr, float beta1, float beta2,
                        float eps, float weight_decay,
                        int step);

// ─── Add bias (broadcast) ─────────────────────────────────────────────────────
/** @brief Adds bias vector to each row of a 2D matrix: out[i][j] += bias[j]. */
void launch_add_bias(float* out, const float* bias, int M, int N);

/** @brief Adds bias with NCHW broadcast: out[n][c][h][w] += bias[c]. */
void launch_add_bias_nchw(float* out, const float* bias,
                          int N, int C, int H, int W);

} // namespace kernels
} // namespace minitensor
