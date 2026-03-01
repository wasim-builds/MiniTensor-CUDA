/**
 * @file test_tensor.cu
 * @brief Tensor creation, transfer, reshape, and scalar extraction tests.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace minitensor;

static bool approx(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

int main() {
    int failures = 0;

    // ── zeros / ones ──────────────────────────────────────────────────────
    {
        Tensor z = Tensor::zeros({4, 4}, DType::Float32, Device::CUDA);
        Tensor z_cpu = z.cpu();
        for (int i = 0; i < 16; ++i)
            if (!approx(z_cpu.data_ptr<float>()[i], 0.f)) { ++failures; break; }
        std::printf("[test_tensor] zeros:  %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    {
        int f0 = failures;
        Tensor o = Tensor::ones({3, 5}, DType::Float32, Device::CUDA);
        Tensor o_cpu = o.cpu();
        for (int i = 0; i < 15; ++i)
            if (!approx(o_cpu.data_ptr<float>()[i], 1.f)) { ++failures; break; }
        std::printf("[test_tensor] ones:   %s\n", failures == f0 ? "PASS" : "FAIL");
    }

    // ── reshape ───────────────────────────────────────────────────────────
    {
        int f0 = failures;
        Tensor t = Tensor::ones({2, 6}, DType::Float32, Device::CPU);
        Tensor r = t.reshape({3, 4});
        if (r.shape()[0] != 3 || r.shape()[1] != 4) ++failures;
        std::printf("[test_tensor] reshape: %s\n", failures == f0 ? "PASS" : "FAIL");
    }

    // ── device transfer ───────────────────────────────────────────────────
    {
        int f0 = failures;
        Tensor cpu_t = Tensor::randn({8, 8}, DType::Float32, Device::CPU);
        Tensor gpu_t = cpu_t.cuda();
        Tensor back  = gpu_t.cpu();
        float* s = cpu_t.data_ptr<float>();
        float* d = back.data_ptr<float>();
        for (int i = 0; i < 64; ++i)
            if (!approx(s[i], d[i])) { ++failures; break; }
        std::printf("[test_tensor] H→D→H:  %s\n", failures == f0 ? "PASS" : "FAIL");
    }

    // ── numel ─────────────────────────────────────────────────────────────
    {
        int f0 = failures;
        Tensor t = Tensor::zeros({2, 3, 4});
        if (t.numel() != 24) ++failures;
        std::printf("[test_tensor] numel:  %s\n", failures == f0 ? "PASS" : "FAIL");
    }

    std::printf("\n[test_tensor] %s (%d failure(s))\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
