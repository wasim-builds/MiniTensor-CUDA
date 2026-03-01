/**
 * @file tensor.cu
 * @brief Tensor class implementation — allocation, transfer, and arithmetic.
 */

#include "minitensor/tensor.hpp"
#include "minitensor/memory.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace minitensor {

// ─── Storage ─────────────────────────────────────────────────────────────────
Storage::Storage(size_t bytes, Device device) : bytes_(bytes), device_(device) {
    if (bytes == 0) return;
    if (device == Device::CUDA) {
        data_ = cuda_malloc(bytes);
    } else {
        data_ = cuda_malloc_host(bytes);  // pinned for fast DMA
    }
}

Storage::~Storage() {
    if (!data_) return;
    if (device_ == Device::CUDA) {
        cuda_free(data_);
    } else {
        cuda_free_host(data_);
    }
}

// ─── Tensor ──────────────────────────────────────────────────────────────────
Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device)
{
    compute_strides();
    size_t bytes = static_cast<size_t>(numel()) * dtype_size(dtype_);
    if (bytes > 0)
        storage_ = std::make_shared<Storage>(bytes, device_);
}

void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    size_t stride = dtype_size(dtype_);
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = static_cast<int64_t>(stride);
        stride *= static_cast<size_t>(shape_[i]);
    }
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(),
                           int64_t(1), std::multiplies<int64_t>());
}

bool Tensor::is_contiguous() const { return true; } // always contiguous for now

// ─── Factory methods ──────────────────────────────────────────────────────────
Tensor Tensor::zeros(std::vector<int64_t> shape, DType dtype, Device device) {
    Tensor t(std::move(shape), dtype, device);
    if (t.numel() == 0) return t;
    if (device == Device::CUDA)
        cuda_memset(t.data_ptr(), 0, t.nbytes());
    else
        std::memset(t.data_ptr(), 0, t.nbytes());
    return t;
}

Tensor Tensor::ones(std::vector<int64_t> shape, DType dtype, Device device) {
    Tensor t = zeros(std::move(shape), dtype, device);
    if (t.numel() == 0) return t;
    // Fill with 1.0f on host then transfer if needed
    Tensor host_t(t.shape(), dtype, Device::CPU);
    float* p = host_t.data_ptr<float>();
    std::fill(p, p + host_t.numel(), 1.0f);
    if (device == Device::CPU) return host_t;
    // Copy to device
    cuda_memcpy(t.data_ptr(), host_t.data_ptr(), t.nbytes(),
                cudaMemcpyHostToDevice);
    return t;
}

Tensor Tensor::randn(std::vector<int64_t> shape, DType dtype, Device device,
                     float mean, float stddev)
{
    int64_t n = std::accumulate(shape.begin(), shape.end(),
                                int64_t(1), std::multiplies<int64_t>());
    // Generate on CPU
    Tensor host_t(shape, dtype, Device::CPU);
    float* p = host_t.data_ptr<float>();
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(mean, stddev);
    for (int64_t i = 0; i < n; ++i) p[i] = dist(gen);

    if (device == Device::CPU) return host_t;

    Tensor dev_t(shape, dtype, Device::CUDA);
    cuda_memcpy(dev_t.data_ptr(), host_t.data_ptr(), dev_t.nbytes(),
                cudaMemcpyHostToDevice);
    return dev_t;
}

Tensor Tensor::from_data(void* data, std::vector<int64_t> shape, DType dtype) {
    // Wraps existing host buffer without ownership (non-owning view)
    Tensor t;
    t.shape_   = std::move(shape);
    t.dtype_   = dtype;
    t.device_  = Device::CPU;
    t.compute_strides();
    size_t bytes = static_cast<size_t>(t.numel()) * dtype_size(dtype);
    // Create storage that does NOT free the pointer
    // We allocate a Storage and write the pointer directly (unsafe but documented)
    t.storage_ = std::shared_ptr<Storage>(new Storage(0, Device::CPU),
        [](Storage* s){ delete s; });
    // NOTE: For production, use a custom deleter; here we copy for safety.
    (void)data; (void)bytes;
    throw std::runtime_error("from_data with non-owning view: copy the data instead.");
    return t;
}

// ─── Device transfer ──────────────────────────────────────────────────────────
Tensor Tensor::to(Device target) const {
    if (device_ == target) return *this;
    Tensor dst(shape_, dtype_, target);
    if (nbytes() == 0) return dst;
    if (target == Device::CUDA) {
        cuda_memcpy(dst.data_ptr(), data_ptr(), nbytes(), cudaMemcpyHostToDevice);
    } else {
        cuda_memcpy(dst.data_ptr(), data_ptr(), nbytes(), cudaMemcpyDeviceToHost);
    }
    return dst;
}

// ─── Reshape ─────────────────────────────────────────────────────────────────
Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    int64_t new_n = std::accumulate(new_shape.begin(), new_shape.end(),
                                    int64_t(1), std::multiplies<int64_t>());
    if (new_n != numel())
        throw std::invalid_argument("Tensor::reshape: numel mismatch");

    Tensor t;
    t.storage_ = storage_;
    t.shape_   = std::move(new_shape);
    t.dtype_   = dtype_;
    t.device_  = device_;
    t.offset_  = offset_;
    t.compute_strides();
    return t;
}

Tensor Tensor::contiguous() const { return *this; }

// ─── Scalar extraction ────────────────────────────────────────────────────────
template<>
float Tensor::item<float>() const {
    if (numel() != 1)
        throw std::runtime_error("item() only valid for 1-element tensor");
    float val;
    if (device_ == Device::CUDA) {
        cuda_memcpy(&val, data_ptr(), sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        val = *data_ptr<float>();
    }
    return val;
}

// ─── Gradient helpers ─────────────────────────────────────────────────────────
void Tensor::zero_grad() {
    if (!grad_) return;
    if (device_ == Device::CUDA)
        cuda_memset(grad_->data_ptr(), 0, grad_->nbytes());
    else
        std::memset(grad_->data_ptr(), 0, grad_->nbytes());
}

// ─── String representation ────────────────────────────────────────────────────
std::string Tensor::to_string(bool /*print_data*/) const {
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) oss << ", ";
    }
    oss << "], dtype=" << dtype_name(dtype_)
        << ", device=" << device_name(device_) << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    return os << t.to_string(false);
}

} // namespace minitensor
