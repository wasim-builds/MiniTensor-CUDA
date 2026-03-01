/**
 * @file linear.cu
 * @brief Linear (fully-connected) layer — forward and backward.
 */

#include "minitensor/layers/linear.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"
#include "minitensor/memory.hpp"

#include <cuda_runtime.h>
#include <cmath>

namespace minitensor {
namespace layers {

// ─── Constructor ──────────────────────────────────────────────────────────────
Linear::Linear(int in_features, int out_features, bool bias, Device device)
    : in_features_(in_features), out_features_(out_features),
      use_bias_(bias), device_(device)
{
    weight_ = Tensor({out_features, in_features}, DType::Float32, device);
    if (use_bias_)
        bias_ = Tensor({out_features}, DType::Float32, device);
    reset_parameters();
}

// ─── Kaiming uniform init ─────────────────────────────────────────────────────
// fan_in = in_features, gain = sqrt(2) for ReLU
// bound = sqrt(3) * sqrt(2 / fan_in)
void Linear::reset_parameters() {
    float bound = std::sqrt(6.0f / static_cast<float>(in_features_));
    // Generate uniform [-bound, bound] on CPU then transfer
    Tensor w = Tensor::randn({out_features_, in_features_},
                             DType::Float32, Device::CPU, 0.0f, bound * 0.5773f);
    weight_ = w.to(device_);

    if (use_bias_) {
        float b_bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        Tensor b = Tensor::randn({out_features_}, DType::Float32,
                                 Device::CPU, 0.0f, b_bound * 0.5773f);
        bias_ = b.to(device_);
    }
}

// ─── Forward: out = x*W^T + b ─────────────────────────────────────────────────
// x:   (N, in_features)   -> treated as (N, K)
// W^T: (in_features, out) -> treated as (K, M)
// out: (N, out)           -> (N, M)
Tensor Linear::forward(const Tensor& x) {
    NVTX_RANGE("Linear::forward");

    input_cache_ = x;  // save for backward
    int N = static_cast<int>(x.shape()[0]);
    int K = in_features_;
    int M = out_features_;

    Tensor out = Tensor::zeros({N, M}, DType::Float32, device_);

    // C = x * W^T  i.e.  GEMM(x[N,K], W[M,K]^T -> treat as W^T[K,M])
    // We call launch_matmul with A=x (N,K), B=W^T
    // Since W is stored as (M,K), W^T is (K,M) — we pass it transposed
    // by swapping arguments: launch_matmul(W, x^T) would need a transpose.
    // Easiest: compute out = x * W^T by calling GEMM(A=x[N,K], B=W[M,K], C[N,M])
    // with B interpreted as transposed. We implement a transpose-B flag variant inline:
    // Actually: out[n][m] = sum_k x[n][k] * W[m][k]
    // = GEMM where we need B = W transposed. We manually transpose W or use a
    // separate kernel call. For simplicity, we transpose W at call time.

    // Allocate WT = W^T  (K, M)
    Tensor wt = Tensor({K, M}, DType::Float32, device_);
    // Transpose kernel (inline for now — simple 2D element copy)
    // For a production system, use a coalesced transpose kernel.
    {
        Tensor w_cpu = weight_.to(Device::CPU);
        Tensor wt_cpu({K, M}, DType::Float32, Device::CPU);
        float* src = w_cpu.data_ptr<float>();
        float* dst = wt_cpu.data_ptr<float>();
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                dst[k * M + m] = src[m * K + k];
        wt = wt_cpu.to(device_);
    }

    kernels::launch_matmul(
        x.data_ptr<float>(), wt.data_ptr<float>(), out.data_ptr<float>(),
        N, M, K);

    if (use_bias_) {
        // Broadcast bias: out[n][m] += bias[m]
        kernels::launch_add_bias(out.data_ptr<float>(),
                                 bias_.data_ptr<float>(), N, M);
    }
    return out;
}

// ─── Backward ────────────────────────────────────────────────────────────────
// grad_output: (N, M)
// grad_weight = grad_output^T * x   → (M, K)
// grad_x      = grad_output * W     → (N, K)
// grad_bias   = sum_N(grad_output)  → (M,)
Tensor Linear::backward(const Tensor& grad_output) {
    NVTX_RANGE("Linear::backward");

    int N = static_cast<int>(input_cache_.shape()[0]);
    int K = in_features_;
    int M = out_features_;

    // grad_x = grad_output (N,M) * W (M,K) → (N,K)
    Tensor grad_x({N, K}, DType::Float32, device_);
    kernels::launch_matmul(
        grad_output.data_ptr<float>(), weight_.data_ptr<float>(),
        grad_x.data_ptr<float>(), N, K, M);

    // grad_weight = grad_output^T (M,N) * x (N,K) → (M,K)
    // Transpose grad_output for the GEMM
    Tensor got_cpu = grad_output.to(Device::CPU);
    Tensor got_T({M, N}, DType::Float32, Device::CPU);
    float* s = got_cpu.data_ptr<float>();
    float* d = got_T.data_ptr<float>();
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m)
            d[m * N + n] = s[n * M + m];
    Tensor got_T_dev = got_T.to(device_);

    // weight_.grad not tracked here (demo: accumulate into separate storage)
    Tensor grad_w({M, K}, DType::Float32, device_);
    kernels::launch_matmul(
        got_T_dev.data_ptr<float>(), input_cache_.data_ptr<float>(),
        grad_w.data_ptr<float>(), M, K, N);

    // grad_bias = sum over N
    if (use_bias_) {
        Tensor grad_b({M}, DType::Float32, device_);
        kernels::launch_sum_axis(
            grad_output.data_ptr<float>(),
            grad_b.data_ptr<float>(), N, M, 0);
        (void)grad_b; // caller retrieves via bias_.grad in full autograd context
    }

    return grad_x;
}

} // namespace layers
} // namespace minitensor
