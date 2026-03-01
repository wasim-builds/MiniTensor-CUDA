/**
 * @file bench_e2e.cu
 * @brief End-to-end forward pass benchmark: 5-layer CNN (similar to LeNet-5).
 *
 * Architecture:
 *   Conv(1→6, 5×5) → ReLU → AvgPool(2×2)
 *   Conv(6→16, 5×5) → ReLU → AvgPool(2×2)
 *   Linear(16×4×4, 120) → ReLU
 *   Linear(120, 84) → ReLU
 *   Linear(84, 10)
 */

#include "minitensor/tensor.hpp"
#include "minitensor/layers/linear.hpp"
#include "minitensor/layers/conv2d.hpp"
#include "minitensor/layers/activation.hpp"
#include "minitensor/layers/pooling.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

using namespace minitensor;
using namespace minitensor::layers;
using Clock = std::chrono::high_resolution_clock;

int main() {
    std::printf("\n=== End-to-End CNN Forward Pass Benchmark ===\n");
    std::printf("Architecture: LeNet-5 style (input 32×32×1)\n\n");

    const int BATCH   = 32;
    const int WARMUP  = 5;
    const int REPS    = 50;

    // ── Build network ──────────────────────────────────────────────────────
    Conv2d conv1(1,  6,  5, 5, 1, 1, 0, 0, 1, 1, true, Device::CUDA);
    ReLU   relu1;
    AvgPool2d pool1(2, 2, 2, 2, 0, 0, Device::CUDA);

    Conv2d conv2(6,  16, 5, 5, 1, 1, 0, 0, 1, 1, true, Device::CUDA);
    ReLU   relu2;
    AvgPool2d pool2(2, 2, 2, 2, 0, 0, Device::CUDA);

    Linear fc1(16*4*4, 120, true, Device::CUDA);
    ReLU   relu3;
    Linear fc2(120, 84, true, Device::CUDA);
    ReLU   relu4;
    Linear fc3(84,  10, true, Device::CUDA);

    auto forward = [&](const Tensor& x) {
        NVTX_RANGE("e2e_forward");
        Tensor t = conv1.forward(x);
        t = relu1.forward(t);
        t = pool1.forward(t);
        t = conv2.forward(t);
        t = relu2.forward(t);
        t = pool2.forward(t);
        t = t.reshape({BATCH, 16*4*4});
        t = fc1.forward(t);
        t = relu3.forward(t);
        t = fc2.forward(t);
        t = relu4.forward(t);
        t = fc3.forward(t);
        return t;
    };

    Tensor x = Tensor::randn({BATCH, 1, 32, 32}, DType::Float32, Device::CUDA);

    // Warmup
    for (int w = 0; w < WARMUP; ++w) forward(x);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── GPU timing ─────────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int r = 0; r < REPS; ++r) forward(x);

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float gpu_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_total_ms, ev_start, ev_stop));
    double gpu_ms = gpu_total_ms / REPS;

    // ── CPU timing ─────────────────────────────────────────────────────────
    Conv2d cpu_conv1(1,  6,  5, 5, 1, 1, 0, 0, 1, 1, true, Device::CPU);
    Conv2d cpu_conv2(6, 16,  5, 5, 1, 1, 0, 0, 1, 1, true, Device::CPU);
    Linear cpu_fc1(16*4*4, 120, true, Device::CPU);
    Linear cpu_fc2(120, 84, true, Device::CPU);
    Linear cpu_fc3(84,  10, true, Device::CPU);
    ReLU cpu_r1, cpu_r2, cpu_r3, cpu_r4;
    AvgPool2d cpu_p1(2,2,2,2,0,0,Device::CPU);
    AvgPool2d cpu_p2(2,2,2,2,0,0,Device::CPU);

    Tensor x_cpu = x.cpu();
    auto cpu_forward = [&]() {
        Tensor t = cpu_conv1.forward(x_cpu);
        t = cpu_r1.forward(t); t = cpu_p1.forward(t);
        t = cpu_conv2.forward(t);
        t = cpu_r2.forward(t); t = cpu_p2.forward(t);
        t = t.reshape({BATCH, 16*4*4});
        t = cpu_fc1.forward(t); t = cpu_r3.forward(t);
        t = cpu_fc2.forward(t); t = cpu_r4.forward(t);
        t = cpu_fc3.forward(t);
        return t;
    };

    auto t0 = Clock::now();
    cpu_forward();
    auto t1 = Clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1-t0).count();

    std::printf("GPU forward (batch=%d):  %.2f ms / iter\n", BATCH, gpu_ms);
    std::printf("CPU forward (batch=%d):  %.2f ms / iter\n", BATCH, cpu_ms);
    std::printf("Speedup:                 %.1fx\n", cpu_ms / gpu_ms);
    std::printf("GPU throughput:          %.0f samples/sec\n\n",
                BATCH / (gpu_ms * 1e-3));

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
