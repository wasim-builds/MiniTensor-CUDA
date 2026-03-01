/**
 * @file bench_matmul.cu
 * @brief CPU vs GPU matrix multiplication benchmark.
 *
 * Measures wall-clock time for square GEMM at sizes {512, 1024, 2048, 4096}.
 * Reports milliseconds, GFLOPS, and speedup ratio.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace minitensor;
using Clock = std::chrono::high_resolution_clock;

// ─── CPU naive GEMM ───────────────────────────────────────────────────────────
static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            float a = A[i*K+k];
            for (int j = 0; j < N; ++j)
                C[i*N+j] += a * B[k*N+j];
        }
}

int main() {
    std::printf("\n=== Matrix Multiplication Benchmark (CUDA vs CPU) ===\n");
    std::printf("%-6s | %-10s | %-10s | %-9s | %-13s\n",
                "N", "CPU (ms)", "GPU (ms)", "Speedup", "GFLOPS (GPU)");
    std::printf("%s\n", std::string(60, '-').c_str());

    const int WARMUP = 5, REPS = 20;

    for (int N : {512, 1024, 2048, 4096}) {
        int M = N, K = N;
        double flops = 2.0 * M * N * K;

        // ── CPU timing ────────────────────────────────────────────────────
        std::vector<float> A(M*K), B(K*N), C_cpu(M*N, 0.f);
        for (auto& v : A) v = static_cast<float>(rand()) / RAND_MAX;
        for (auto& v : B) v = static_cast<float>(rand()) / RAND_MAX;

        double cpu_ms = 1e30;
        if (N <= 2048) {
            auto t0 = Clock::now();
            cpu_gemm(A.data(), B.data(), C_cpu.data(), M, N, K);
            auto t1 = Clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        }

        // ── GPU timing ────────────────────────────────────────────────────
        Tensor gA = Tensor::randn({M, K}, DType::Float32, Device::CUDA);
        Tensor gB = Tensor::randn({K, N}, DType::Float32, Device::CUDA);
        Tensor gC = Tensor::zeros({M, N}, DType::Float32, Device::CUDA);

        // Warmup
        for (int w = 0; w < WARMUP; ++w)
            kernels::launch_matmul(gA.data_ptr<float>(), gB.data_ptr<float>(),
                                   gC.data_ptr<float>(), M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        NVTX_MARK("bench_matmul_start");

        for (int r = 0; r < REPS; ++r)
            kernels::launch_matmul(gA.data_ptr<float>(), gB.data_ptr<float>(),
                                   gC.data_ptr<float>(), M, N, K);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        NVTX_MARK("bench_matmul_end");

        float gpu_ms_total;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms_total, start, stop));
        double gpu_ms = gpu_ms_total / REPS;

        double speedup   = (N <= 2048) ? (cpu_ms / gpu_ms) : -1;
        double gflops    = flops / (gpu_ms * 1e-3) / 1e9;

        if (N <= 2048)
            std::printf("%-6d | %-10.1f | %-10.2f | %-9.1fx | %-13.1f\n",
                        N, cpu_ms, gpu_ms, speedup, gflops);
        else
            std::printf("%-6d | %-10s | %-10.2f | %-9s | %-13.1f\n",
                        N, "N/A", gpu_ms, "N/A", gflops);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::printf("\n");
    return 0;
}
