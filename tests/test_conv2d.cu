/**
 * @file test_conv2d.cu
 * @brief Conv2D correctness: GPU im2col-GEMM vs naive CPU reference.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/layers/conv2d.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace minitensor;

// ─── CPU naive convolution ────────────────────────────────────────────────────
static void cpu_conv2d(const float* x, const float* w, const float* b, float* y,
                       int N, int C, int H, int W,
                       int F, int kH, int kW, int sH, int sW, int pH, int pW)
{
    int outH = (H + 2*pH - kH) / sH + 1;
    int outW = (W + 2*pW - kW) / sW + 1;
    for (int n = 0; n < N; ++n)
    for (int f = 0; f < F; ++f)
    for (int oh = 0; oh < outH; ++oh)
    for (int ow = 0; ow < outW; ++ow) {
        float s = b ? b[f] : 0.f;
        for (int c = 0; c < C; ++c)
        for (int kh = 0; kh < kH; ++kh)
        for (int kw = 0; kw < kW; ++kw) {
            int ih = oh*sH - pH + kh, iw = ow*sW - pW + kw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                s += x[((n*C+c)*H+ih)*W+iw] * w[((f*C+c)*kH+kh)*kW+kw];
        }
        y[((n*F+f)*outH+oh)*outW+ow] = s;
    }
}

int main() {
    bool all_pass = true;

    struct Case { int N, C, H, W, F, kH, kW, sH, sW, pH, pW; };
    std::vector<Case> cases = {
        {1,  1,  8,  8,  4, 3, 3, 1, 1, 0, 0},
        {2,  3, 16, 16,  8, 3, 3, 1, 1, 1, 1},
        {1,  8, 32, 32, 16, 5, 5, 2, 2, 2, 2},
    };

    for (auto& c : cases) {
        // CPU inputs
        Tensor x_cpu = Tensor::randn({c.N,c.C,c.H,c.W}, DType::Float32, Device::CPU);
        Tensor w_cpu = Tensor::randn({c.F,c.C,c.kH,c.kW}, DType::Float32, Device::CPU);
        Tensor b_cpu = Tensor::zeros({c.F}, DType::Float32, Device::CPU);

        // GPU forward
        layers::Conv2d conv(c.C, c.F, c.kH, c.kW, c.sH, c.sW, c.pH, c.pW);
        // Override weights with the same CPU tensors
        conv.weight() = w_cpu.cuda();
        conv.bias_tensor() = b_cpu.cuda();
        Tensor out_gpu = conv.forward(x_cpu.cuda());
        Tensor out_cpu = out_gpu.cpu();

        // CPU reference
        int outH = (c.H + 2*c.pH - c.kH) / c.sH + 1;
        int outW = (c.W + 2*c.pW - c.kW) / c.sW + 1;
        std::vector<float> ref(c.N * c.F * outH * outW, 0.f);
        cpu_conv2d(x_cpu.data_ptr<float>(), w_cpu.data_ptr<float>(),
                   b_cpu.data_ptr<float>(), ref.data(),
                   c.N, c.C, c.H, c.W, c.F, c.kH, c.kW, c.sH, c.sW, c.pH, c.pW);

        float max_err = 0.f;
        float* gp = out_cpu.data_ptr<float>();
        for (int i = 0; i < (int)ref.size(); ++i) {
            float err = std::abs(gp[i] - ref[i]);
            if (err > max_err) max_err = err;
        }
        bool pass = max_err < 1e-3f;
        std::printf("[test_conv2d] N=%d C=%d H=%d F=%d k=%dx%d  max_err=%.2e  %s\n",
                    c.N, c.C, c.H, c.F, c.kH, c.kW, max_err, pass ? "PASS" : "FAIL");
        if (!pass) all_pass = false;
    }
    return all_pass ? 0 : 1;
}
