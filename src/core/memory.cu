/**
 * @file memory.cu
 * @brief CUDA memory pool implementation with coalesced allocation.
 */

#include "minitensor/memory.hpp"
#include "minitensor/utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <bit>
#include <stdexcept>

namespace minitensor {

// ─── Raw helpers ──────────────────────────────────────────────────────────────
void* cuda_malloc(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void* cuda_malloc_host(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, bytes));
    return ptr;
}

void cuda_free_host(void* ptr) {
    if (ptr) cudaFreeHost(ptr);
}

void cuda_memcpy(void* dst, const void* src, size_t bytes, int kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes,
               static_cast<cudaMemcpyKind>(kind)));
}

void cuda_memset(void* ptr, int value, size_t bytes) {
    CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

// ─── MemoryPool ───────────────────────────────────────────────────────────────
MemoryPool& MemoryPool::global() {
    static MemoryPool instance;
    return instance;
}

MemoryPool::~MemoryPool() {
    // Free all cached device blocks
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [size, vec] : free_blocks_) {
        for (void* ptr : vec) cudaFree(ptr);
    }
    for (auto& [ptr, size] : allocated_ptrs_) {
        cudaFree(ptr);  // still-allocated blocks
    }
}

/* static */ size_t MemoryPool::round_to_bucket(size_t bytes) {
    // Round up to next power of two (minimum 256 bytes)
    if (bytes <= kDefaultAlignment) return kDefaultAlignment;
    // std::bit_ceil requires C++20; fall back for C++17:
    size_t p = 1;
    while (p < bytes) p <<= 1;
    return p;
}

void* MemoryPool::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    size_t bucket = round_to_bucket(bytes);

    std::lock_guard<std::mutex> lock(mutex_);
    auto& vec = free_blocks_[bucket];
    if (!vec.empty()) {
        void* ptr = vec.back();
        vec.pop_back();
        allocated_ptrs_[ptr] = bucket;
        total_allocated_ += bucket;
        return ptr;
    }

    // No free block — allocate new
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bucket));
    allocated_ptrs_[ptr] = bucket;
    total_reserved_  += bucket;
    total_allocated_ += bucket;
    return ptr;
}

void MemoryPool::free(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_ptrs_.find(ptr);
    if (it == allocated_ptrs_.end())
        throw std::runtime_error("MemoryPool::free: unknown pointer");
    size_t bucket = it->second;
    allocated_ptrs_.erase(it);
    free_blocks_[bucket].push_back(ptr);
    total_allocated_ -= bucket;
}

size_t MemoryPool::total_reserved()   const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_reserved_;
}
size_t MemoryPool::total_allocated()  const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_allocated_;
}

void MemoryPool::release_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [size, vec] : free_blocks_) {
        for (void* ptr : vec) {
            cudaFree(ptr);
            total_reserved_ -= size;
        }
        vec.clear();
    }
}

// ─── PooledMemory ─────────────────────────────────────────────────────────────
PooledMemory::PooledMemory(size_t bytes, MemoryPool& pool)
    : bytes_(bytes), pool_(pool)
{
    ptr_ = pool_.allocate(bytes);
}

PooledMemory::~PooledMemory() {
    pool_.free(ptr_);
}

} // namespace minitensor
