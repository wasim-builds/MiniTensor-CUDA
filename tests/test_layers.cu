/**
 * @file test_layers.cu
 * @brief Layer forward/backward shape sanity and NaN checks.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/layers/linear.hpp"
#include "minitensor/layers/conv2d.hpp"
#include "minitensor/layers/activation.hpp"
#include "minitensor/layers/batchnorm.hpp"
#include "minitensor/layers/pooling.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace minitensor;
using namespace minitensor::layers;

static bool has_nan(const Tensor& t) {
    Tensor cpu = t.cpu();
    float* p = cpu.data_ptr<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i)
        if (std::isnan(p[i]) || std::isinf(p[i])) return true;
    return false;
}

static bool shape_eq(const Tensor& t, std::vector<int64_t> expected) {
    return t.shape() == expected;
}

int main() {
    int failures = 0;

    // ── Linear ────────────────────────────────────────────────────────────
    {
        Linear fc(32, 16, true, Device::CUDA);
        Tensor x = Tensor::randn({4, 32}, DType::Float32, Device::CUDA);
        Tensor y = fc.forward(x);
        bool ok  = shape_eq(y, {4, 16}) && !has_nan(y);
        std::printf("[test_layers] Linear fwd:      %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;

        Tensor g = Tensor::ones({4, 16}, DType::Float32, Device::CUDA);
        Tensor gx = fc.backward(g);
        ok = shape_eq(gx, {4, 32}) && !has_nan(gx);
        std::printf("[test_layers] Linear bwd:      %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── Conv2d ────────────────────────────────────────────────────────────
    {
        Conv2d conv(3, 8, 3, 3, 1, 1, 1, 1, 1, 1, true, Device::CUDA);
        Tensor x = Tensor::randn({2,3,16,16}, DType::Float32, Device::CUDA);
        Tensor y = conv.forward(x);
        bool ok  = shape_eq(y, {2, 8, 16, 16}) && !has_nan(y);
        std::printf("[test_layers] Conv2d fwd:      %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── ReLU ─────────────────────────────────────────────────────────────
    {
        ReLU relu;
        Tensor x = Tensor::randn({4, 8}, DType::Float32, Device::CUDA);
        Tensor y = relu.forward(x);
        // Verify no negative outputs
        Tensor y_cpu = y.cpu();
        bool ok = !has_nan(y);
        float* p = y_cpu.data_ptr<float>();
        for (int64_t i = 0; i < y_cpu.numel(); ++i)
            if (p[i] < 0.f) { ok = false; break; }
        std::printf("[test_layers] ReLU fwd:        %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── BatchNorm2d ───────────────────────────────────────────────────────
    {
        BatchNorm2d bn(8, 1e-5f, 0.1f, true, Device::CUDA);
        Tensor x = Tensor::randn({2, 8, 4, 4}, DType::Float32, Device::CUDA);
        Tensor y = bn.forward(x);
        bool ok  = shape_eq(y, {2, 8, 4, 4}) && !has_nan(y);
        std::printf("[test_layers] BatchNorm2d fwd: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // ── MaxPool2d ─────────────────────────────────────────────────────────
    {
        MaxPool2d pool(2, 2, 2, 2, 0, 0, Device::CUDA);
        Tensor x = Tensor::randn({2, 4, 8, 8}, DType::Float32, Device::CUDA);
        Tensor y = pool.forward(x);
        bool ok  = shape_eq(y, {2, 4, 4, 4}) && !has_nan(y);
        std::printf("[test_layers] MaxPool2d fwd:   %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    std::printf("\n[test_layers] %s (%d failure(s))\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
