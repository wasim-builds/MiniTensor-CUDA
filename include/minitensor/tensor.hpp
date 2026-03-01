/**
 * @file tensor.hpp
 * @brief Core Tensor class for MiniTensor-CUDA.
 *
 * Provides a multi-dimensional array abstraction that lives on either
 * the CPU (host) or CUDA device.  The Tensor owns its memory and
 * handles allocation/deallocation automatically.
 *
 * Thread-safety: Tensor objects are NOT thread-safe.  Use separate
 * Tensor instances per CUDA stream / host thread.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace minitensor {

// ─── Device enum ─────────────────────────────────────────────────────────────
enum class Device : uint8_t {
    CPU  = 0,
    CUDA = 1,
};

inline const char* device_name(Device d) {
    return d == Device::CPU ? "cpu" : "cuda";
}

// ─── DType enum ──────────────────────────────────────────────────────────────
enum class DType : uint8_t {
    Float32 = 0,
    Float16  = 1,
    Int32    = 2,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int32:   return 4;
    }
    return 4;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::Int32:   return "int32";
    }
    return "unknown";
}

// ─── Storage (ref-counted raw memory) ────────────────────────────────────────
/**
 * @brief Ref-counted raw byte buffer on CPU or CUDA device.
 *
 * Multiple Tensor views can share the same Storage.
 */
class Storage {
public:
    Storage() = default;
    Storage(size_t bytes, Device device);
    ~Storage();

    // Non-copyable; use shared_ptr<Storage>
    Storage(const Storage&)            = delete;
    Storage& operator=(const Storage&) = delete;

    void*  data()   const { return data_; }
    size_t bytes()  const { return bytes_; }
    Device device() const { return device_; }

private:
    void*  data_   = nullptr;
    size_t bytes_  = 0;
    Device device_ = Device::CPU;
};

// ─── Tensor ──────────────────────────────────────────────────────────────────
/**
 * @brief N-dimensional tensor with optional gradient tracking.
 *
 * Layout is always contiguous row-major (C-order).
 *
 * @code
 *   // Create a 3×4 float32 tensor filled with zeros on CUDA
 *   auto t = Tensor::zeros({3, 4}, DType::Float32, Device::CUDA);
 *
 *   // Move to CPU for inspection
 *   auto t_cpu = t.to(Device::CPU);
 *   float val  = t_cpu.item<float>();   // only valid for 1-element tensors
 * @endcode
 */
class Tensor {
public:
    // ── Constructors / destructor ─────────────────────────────────────────
    Tensor() = default;

    /**
     * @brief Constructs a tensor with given shape on the specified device.
     * @param shape  Dimension sizes, e.g. {N, C, H, W}
     * @param dtype  Element type (default: float32)
     * @param device Target device (default: CPU)
     */
    Tensor(std::vector<int64_t> shape,
           DType  dtype  = DType::Float32,
           Device device = Device::CPU);

    // Rule of five — rely on shared_ptr for deep copy semantics
    Tensor(const Tensor&)             = default;
    Tensor& operator=(const Tensor&)  = default;
    Tensor(Tensor&&)      noexcept    = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor()                         = default;

    // ── Factory methods ───────────────────────────────────────────────────
    /** @brief Zero-filled tensor. */
    static Tensor zeros(std::vector<int64_t> shape,
                        DType  dtype  = DType::Float32,
                        Device device = Device::CPU);

    /** @brief One-filled tensor. */
    static Tensor ones(std::vector<int64_t> shape,
                       DType  dtype  = DType::Float32,
                       Device device = Device::CPU);

    /** @brief Standard-normal random tensor (host RNG, then optionally copied to device). */
    static Tensor randn(std::vector<int64_t> shape,
                        DType  dtype  = DType::Float32,
                        Device device = Device::CPU,
                        float  mean   = 0.0f,
                        float  stddev = 1.0f);

    /** @brief Wraps an existing host buffer WITHOUT taking ownership. */
    static Tensor from_data(void* data,
                            std::vector<int64_t> shape,
                            DType dtype = DType::Float32);

    // ── Properties ───────────────────────────────────────────────────────
    const std::vector<int64_t>& shape()   const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    DType    dtype()  const { return dtype_; }
    Device   device() const { return device_; }

    int64_t  ndim()     const { return static_cast<int64_t>(shape_.size()); }
    int64_t  numel()    const;
    size_t   nbytes()   const { return static_cast<size_t>(numel()) * dtype_size(dtype_); }

    /** @brief Returns true if tensor owns contiguous memory. */
    bool is_contiguous() const;

    /** @brief Raw data pointer — caller must know the device. */
    void*       data_ptr()       { return static_cast<char*>(storage_->data()) + offset_; }
    const void* data_ptr() const { return static_cast<const char*>(storage_->data()) + offset_; }

    /** @brief Typed raw pointer (device-appropriate). */
    template<typename T>
    T*       data_ptr()       { return reinterpret_cast<T*>(data_ptr()); }
    template<typename T>
    const T* data_ptr() const { return reinterpret_cast<const T*>(data_ptr()); }

    // ── Shape manipulation ────────────────────────────────────────────────
    /**
     * @brief Returns a view with a new shape (must have same numel).
     *  Tensor must be contiguous.
     */
    Tensor reshape(std::vector<int64_t> new_shape) const;

    /** @brief Alias for reshape. */
    Tensor view(std::vector<int64_t> new_shape) const { return reshape(new_shape); }

    /** @brief Returns a contiguous copy. */
    Tensor contiguous() const;

    // ── Device transfer ───────────────────────────────────────────────────
    /**
     * @brief Returns tensor on the target device.
     * No-op if already on target device.
     */
    Tensor to(Device target_device) const;

    Tensor cuda() const { return to(Device::CUDA); }
    Tensor cpu()  const { return to(Device::CPU);  }

    // ── Scalar access ─────────────────────────────────────────────────────
    /**
     * @brief Extracts a scalar value (only valid for 1-element tensors).
     * Automatically copies from device if needed.
     */
    template<typename T = float>
    T item() const;

    // ── Operator overloads (element-wise, creates new tensor) ─────────────
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // element-wise
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // ── Gradient tracking ─────────────────────────────────────────────────
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool val) { requires_grad_ = val; }

    Tensor& grad()       { return *grad_; }
    const Tensor& grad() const { return *grad_; }
    bool has_grad() const { return grad_ != nullptr; }

    void zero_grad();
    void accumulate_grad(const Tensor& g);

    // ── Utilities ─────────────────────────────────────────────────────────
    std::string to_string(bool print_data = false) const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

    // ── Internals (used by kernels / layers) ──────────────────────────────
    std::shared_ptr<Storage> storage() const { return storage_; }

private:
    std::shared_ptr<Storage>  storage_;
    std::vector<int64_t>      shape_;
    std::vector<int64_t>      strides_;    // in bytes
    size_t                    offset_ = 0; // byte offset into storage
    DType                     dtype_  = DType::Float32;
    Device                    device_ = Device::CPU;
    bool                      requires_grad_ = false;
    std::unique_ptr<Tensor>   grad_;

    void compute_strides();
};

} // namespace minitensor
