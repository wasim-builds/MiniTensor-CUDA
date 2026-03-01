/**
 * @file test_matmul.cu
 * @brief GEMM correctness: compares CUDA tiled GEMM vs CPU naive.
 * Acceptance: max absolute error < 1e-3 for N up to 512.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace minitensor;

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

int main() {
    bool all_pass = true;

    for (int N : {32, 64, 128, 256, 512}) {
        int M = N, K = N;

        Tensor A = Tensor::randn({M, K}, DType::Float32, Device::CUDA);
        Tensor B = Tensor::randn({K, N}, DType::Float32, Device::CUDA);
        Tensor C = Tensor::zeros({M, N}, DType::Float32, Device::CUDA);

        kernels::launch_matmul(
            A.data_ptr<float>(), B.data_ptr<float>(),
            C.data_ptr<float>(), M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // CPU reference
        Tensor A_cpu = A.cpu(), B_cpu = B.cpu();
        std::vector<float> C_ref(M * N, 0.f);
        cpu_gemm(A_cpu.data_ptr<float>(), B_cpu.data_ptr<float>(),
                 C_ref.data(), M, N, K);

        Tensor C_cpu = C.cpu();
        float max_err = 0.f;
        for (int i = 0; i < M*N; ++i) {
            float err = std::abs(C_cpu.data_ptr<float>()[i] - C_ref[i]);
            if (err > max_err) max_err = err;
        }

        bool pass = max_err < 1e-3f;
        std::printf("[test_matmul] N=%4d  max_err=%.2e  %s\n",
                    N, max_err, pass ? "PASS" : "FAIL");
        if (!pass) all_pass = false;
    }

    return all_pass ? 0 : 1;
}
