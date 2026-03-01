/**
 * @file bench_conv2d.cu
 * @brief Conv2D benchmark using ResNet-50 first-layer dimensions.
 * Input: (8, 3, 224, 224), Filter: (64, 3, 7, 7), stride=2, pad=3
 */

#include "minitensor/tensor.hpp"
#include "minitensor/layers/conv2d.hpp"
#include "minitensor/utils/cuda_check.hpp"
#include "minitensor/utils/nvtx.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>

using namespace minitensor;
using Clock = std::chrono::high_resolution_clock;

// ─── CPU naive conv ───────────────────────────────────────────────────────────
static double cpu_conv2d_ms(int N, int C, int H, int W, int F, int kH, int kW,
                             int sH, int sW, int pH, int pW)
{
    int outH = (H + 2*pH - kH) / sH + 1;
    int outW = (W + 2*pW - kW) / sW + 1;

    std::vector<float> x(N*C*H*W, 1.f);
    std::vector<float> w(F*C*kH*kW, 0.01f);
    std::vector<float> y(N*F*outH*outW, 0.f);

    auto t0 = Clock::now();
    for (int n = 0; n < N; ++n)
    for (int f = 0; f < F; ++f)
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        float s = 0.f;
        for (int c = 0; c < C; ++c)
        for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh*sH-pH+kh, iw = ow*sW-pW+kw;
            if (ih>=0&&ih<H&&iw>=0&&iw<W)
                s += x[((n*C+c)*H+ih)*W+iw]*w[((f*C+c)*kH+kh)*kW+kw];
        }
        y[((n*F+f)*outH+oh)*outW+ow] = s;
    }
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1-t0).count();
}

int main() {
    std::printf("\n=== Conv2D Benchmark (ResNet-50 layer dims) ===\n");

    struct Case { int N, C, H, W, F, kH, kW, sH, sW, pH, pW; const char* label; };
    std::vector<Case> cases = {
        {1,  3, 224, 224, 64, 7, 7, 2, 2, 3, 3, "ResNet-L1  (N=1)"},
        {8,  3, 224, 224, 64, 7, 7, 2, 2, 3, 3, "ResNet-L1  (N=8)"},
        {8, 64,  56,  56, 64, 3, 3, 1, 1, 1, 1, "ResNet-L2  (N=8)"},
    };

    const int WARMUP = 3, REPS = 10;
    std::printf("%-22s | %-10s | %-10s | %-9s\n",
                "Config", "CPU (ms)", "GPU (ms)", "Speedup");
    std::printf("%s\n", std::string(60, '-').c_str());

    for (auto& c : cases) {
        layers::Conv2d conv(c.C, c.F, c.kH, c.kW,
                            c.sH, c.sW, c.pH, c.pW, 1, 1, true, Device::CUDA);
        Tensor x = Tensor::randn({c.N,c.C,c.H,c.W}, DType::Float32, Device::CUDA);

        // Warmup
        for (int w = 0; w < WARMUP; ++w) conv.forward(x);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        NVTX_RANGE("bench_conv2d");
        for (int r = 0; r < REPS; ++r) conv.forward(x);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpu_total;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_total, start, stop));
        double gpu_ms = gpu_total / REPS;

        // CPU (only small cases)
        double cpu_ms = -1;
        if (c.N == 1 && c.H <= 56)
            cpu_ms = cpu_conv2d_ms(c.N,c.C,c.H,c.W,c.F,c.kH,c.kW,c.sH,c.sW,c.pH,c.pW);

        if (cpu_ms > 0)
            std::printf("%-22s | %-10.1f | %-10.2f | %-9.1fx\n",
                        c.label, cpu_ms, gpu_ms, cpu_ms/gpu_ms);
        else
            std::printf("%-22s | %-10s | %-10.2f | %-9s\n",
                        c.label, "N/A", gpu_ms, "N/A");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    std::printf("\n");
    return 0;
}
