/**
 * @file cuda_check.hpp
 * @brief CUDA error-checking macro and helper utilities.
 */
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace minitensor {

inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "[CUDA Error] " << cudaGetErrorString(err)
            << " at " << file << ":" << line;
        throw std::runtime_error(oss.str());
    }
}

} // namespace minitensor

#define CUDA_CHECK(expr) ::minitensor::cuda_check((expr), __FILE__, __LINE__)
