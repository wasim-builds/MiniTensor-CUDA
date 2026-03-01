/**
 * @file batchnorm.cu
 * @brief BatchNorm2d layer: delegates to batchnorm kernel launchers.
 */

#include "minitensor/layers/batchnorm.hpp"
#include "minitensor/kernels.hpp"
#include "minitensor/utils/nvtx.hpp"

namespace minitensor {
namespace layers {

BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum,
                         bool affine, Device device)
    : num_features_(num_features), eps_(eps), momentum_(momentum),
      affine_(affine), device_(device)
{
    if (affine_) {
        gamma_        = Tensor::ones ({num_features}, DType::Float32, device);
        beta_         = Tensor::zeros({num_features}, DType::Float32, device);
    }
    running_mean_ = Tensor::zeros({num_features}, DType::Float32, device);
    running_var_  = Tensor::ones ({num_features}, DType::Float32, device);
}

Tensor BatchNorm2d::forward(const Tensor& x) {
    NVTX_RANGE("BatchNorm2d::forward");

    N_ = static_cast<int>(x.shape()[0]);
    int C = static_cast<int>(x.shape()[1]);
    H_ = static_cast<int>(x.shape()[2]);
    W_ = static_cast<int>(x.shape()[3]);

    input_cache_ = x;
    mean_cache_  = Tensor::zeros({C}, DType::Float32, device_);
    var_cache_   = Tensor::zeros({C}, DType::Float32, device_);
    Tensor out   = Tensor::zeros(x.shape(), DType::Float32, device_);

    if (training_) {
        kernels::launch_batchnorm_forward(
            x.data_ptr<float>(),
            gamma_.data_ptr<float>(), beta_.data_ptr<float>(),
            mean_cache_.data_ptr<float>(), var_cache_.data_ptr<float>(),
            running_mean_.data_ptr<float>(), running_var_.data_ptr<float>(),
            out.data_ptr<float>(),
            N_, C, H_, W_, eps_, momentum_);
    } else {
        // Eval mode: use running stats — normalize inline
        kernels::launch_batchnorm_forward(
            x.data_ptr<float>(),
            gamma_.data_ptr<float>(), beta_.data_ptr<float>(),
            running_mean_.data_ptr<float>(), running_var_.data_ptr<float>(),
            running_mean_.data_ptr<float>(), running_var_.data_ptr<float>(),
            out.data_ptr<float>(),
            N_, C, H_, W_, eps_, 0.0f);  // momentum=0 → don't update running
    }
    return out;
}

Tensor BatchNorm2d::backward(const Tensor& grad_out) {
    NVTX_RANGE("BatchNorm2d::backward");

    int C = num_features_;
    Tensor grad_x     = Tensor::zeros(input_cache_.shape(), DType::Float32, device_);
    Tensor grad_gamma = Tensor::zeros({C}, DType::Float32, device_);
    Tensor grad_beta  = Tensor::zeros({C}, DType::Float32, device_);

    kernels::launch_batchnorm_backward(
        grad_out.data_ptr<float>(),
        input_cache_.data_ptr<float>(),
        gamma_.data_ptr<float>(),
        mean_cache_.data_ptr<float>(), var_cache_.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
        N_, C, H_, W_, eps_);

    return grad_x;
}

} // namespace layers
} // namespace minitensor
