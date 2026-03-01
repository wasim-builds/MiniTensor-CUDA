/**
 * @file activation.cu
 * @brief Activation layer wrappers calling kernel launchers.
 */

#include "minitensor/layers/activation.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/nvtx.hpp"

namespace minitensor {
namespace layers {

// ─── ReLU ────────────────────────────────────────────────────────────────────
Tensor ReLU::forward(const Tensor& x) {
    NVTX_RANGE("ReLU::forward");
    input_cache_ = x;
    Tensor out   = Tensor::zeros(x.shape(), x.dtype(), x.device());
    kernels::launch_relu_forward(x.data_ptr<float>(), out.data_ptr<float>(),
                                 static_cast<int>(x.numel()));
    return out;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    NVTX_RANGE("ReLU::backward");
    Tensor grad_in = Tensor::zeros(grad_output.shape(),
                                   grad_output.dtype(), grad_output.device());
    kernels::launch_relu_backward(
        input_cache_.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        static_cast<int>(grad_output.numel()));
    return grad_in;
}

// ─── LeakyReLU ───────────────────────────────────────────────────────────────
Tensor LeakyReLU::forward(const Tensor& x) {
    NVTX_RANGE("LeakyReLU::forward");
    input_cache_ = x;
    Tensor out   = Tensor::zeros(x.shape(), x.dtype(), x.device());
    kernels::launch_leaky_relu_forward(x.data_ptr<float>(), out.data_ptr<float>(),
                                       static_cast<int>(x.numel()), negative_slope_);
    return out;
}

Tensor LeakyReLU::backward(const Tensor& grad_output) {
    NVTX_RANGE("LeakyReLU::backward");
    Tensor grad_in = Tensor::zeros(grad_output.shape(),
                                   grad_output.dtype(), grad_output.device());
    kernels::launch_leaky_relu_backward(
        input_cache_.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        static_cast<int>(grad_output.numel()), negative_slope_);
    return grad_in;
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────
Tensor Sigmoid::forward(const Tensor& x) {
    NVTX_RANGE("Sigmoid::forward");
    output_cache_ = Tensor::zeros(x.shape(), x.dtype(), x.device());
    kernels::launch_sigmoid_forward(x.data_ptr<float>(),
                                    output_cache_.data_ptr<float>(),
                                    static_cast<int>(x.numel()));
    return output_cache_;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    NVTX_RANGE("Sigmoid::backward");
    Tensor grad_in = Tensor::zeros(grad_output.shape(),
                                   grad_output.dtype(), grad_output.device());
    kernels::launch_sigmoid_backward(
        output_cache_.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        static_cast<int>(grad_output.numel()));
    return grad_in;
}

// ─── Tanh ────────────────────────────────────────────────────────────────────
Tensor Tanh::forward(const Tensor& x) {
    NVTX_RANGE("Tanh::forward");
    output_cache_ = Tensor::zeros(x.shape(), x.dtype(), x.device());
    kernels::launch_tanh_forward(x.data_ptr<float>(),
                                 output_cache_.data_ptr<float>(),
                                 static_cast<int>(x.numel()));
    return output_cache_;
}

Tensor Tanh::backward(const Tensor& grad_output) {
    NVTX_RANGE("Tanh::backward");
    Tensor grad_in = Tensor::zeros(grad_output.shape(),
                                   grad_output.dtype(), grad_output.device());
    kernels::launch_tanh_backward(
        output_cache_.data_ptr<float>(),
        grad_output.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        static_cast<int>(grad_output.numel()));
    return grad_in;
}

} // namespace layers
} // namespace minitensor
