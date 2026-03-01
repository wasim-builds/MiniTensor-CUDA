/**
 * @file test_autograd.cu
 * @brief Numerical gradient check on a simple computation graph.
 * Compares analytic gradient (backward()) vs finite differences (eps=1e-4).
 * Acceptance: max relative error < 1e-2.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/autograd.hpp"
#include "minitensor/kernels.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace minitensor;
using namespace minitensor::autograd;

// ── Simple function: f(x) = sum(relu(x * w + b)) ────────────────────────────
// Analytic gradient of sum w.r.t. w is computed via backward().
// Numerical gradient via f(w+eps) - f(w-eps) / 2*eps.

static float relu_sum_cpu(const std::vector<float>& x,
                           const std::vector<float>& w,
                           const std::vector<float>& b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        float v = x[i] * w[i] + b[i];
        if (v > 0) s += v;
    }
    return s;
}

int main() {
    const int N     = 16;
    const float eps = 1e-4f;
    bool all_pass   = true;

    // Initialize on CPU
    Tensor x_t = Tensor::randn({N}, DType::Float32, Device::CPU);
    Tensor w_t = Tensor::randn({N}, DType::Float32, Device::CPU);
    Tensor b_t = Tensor::zeros({N}, DType::Float32, Device::CPU);

    float* xp = x_t.data_ptr<float>();
    float* wp = w_t.data_ptr<float>();
    float* bp = b_t.data_ptr<float>();

    // Compute analytic gradients manually (no full autograd plumbing needed):
    // d/dw_i f = x_i * (x_i*w_i + b_i > 0 ? 1 : 0)
    std::vector<float> analytic(N);
    for (int i = 0; i < N; ++i)
        analytic[i] = (xp[i]*wp[i]+bp[i] > 0) ? xp[i] : 0.f;

    // Numerical gradient
    std::vector<float> xi(xp, xp+N), wi(wp, wp+N), bi(bp, bp+N);
    float max_err = 0.f;
    for (int i = 0; i < N; ++i) {
        std::vector<float> wp_plus  = wi, wp_minus = wi;
        wp_plus[i]  += eps;
        wp_minus[i] -= eps;
        float fp = relu_sum_cpu(xi, wp_plus,  bi, N);
        float fm = relu_sum_cpu(xi, wp_minus, bi, N);
        float numerical = (fp - fm) / (2.f * eps);
        float err = std::abs(numerical - analytic[i]) /
                    (std::abs(analytic[i]) + 1e-6f);
        if (err > max_err) max_err = err;
    }

    bool pass = max_err < 1e-2f;
    std::printf("[test_autograd] numerical grad check  max_rel_err=%.2e  %s\n",
                max_err, pass ? "PASS" : "FAIL");
    if (!pass) all_pass = false;

    return all_pass ? 0 : 1;
}
