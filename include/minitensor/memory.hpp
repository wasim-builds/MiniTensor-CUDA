/**
 * @file memory.hpp
 * @brief CUDA memory pool and coalesced allocation utilities.
 *
 * Goals:
 *  - Reduce cudaMalloc call overhead by recycling freed blocks
 *  - Ensure 256-byte alignment for coalesced memory access
 *  - Provide pinned-memory helpers for fast CPU↔GPU transfers
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace minitensor {

// ─── Alignment constant ───────────────────────────────────────────────────────
static constexpr size_t kDefaultAlignment = 256; // bytes — matches CUDA L2 cache line

// ─── Raw allocation helpers ──────────────────────────────────────────────────
/**
 * @brief Allocates @p bytes on the CUDA device with kDefaultAlignment.
 * @throws std::runtime_error on cudaMalloc failure.
 */
void* cuda_malloc(size_t bytes);

/**
 * @brief Frees a CUDA device pointer. No-op if ptr == nullptr.
 */
void  cuda_free(void* ptr);

/**
 * @brief Allocates page-locked (pinned) host memory for fast DMA transfers.
 */
void* cuda_malloc_host(size_t bytes);

/**
 * @brief Frees pinned host memory.
 */
void  cuda_free_host(void* ptr);

/**
 * @brief Copies @p bytes between host and device.
 * @param dst  Destination pointer (host or device)
 * @param src  Source pointer (host or device)
 * @param kind cudaMemcpyKind (H2D, D2H, D2D, H2H)
 */
void  cuda_memcpy(void* dst, const void* src, size_t bytes, int kind);

/**
 * @brief Zero-fills a device buffer.
 */
void  cuda_memset(void* ptr, int value, size_t bytes);

// ─── MemoryBlock ─────────────────────────────────────────────────────────────
struct MemoryBlock {
    void*  ptr   = nullptr;
    size_t bytes = 0;
    bool   free  = true;
};

// ─── MemoryPool ──────────────────────────────────────────────────────────────
/**
 * @brief Thread-safe CUDA memory pool that recycles allocations.
 *
 * Strategy: bucket free-list keyed by allocation size (rounded to next power
 * of two, minimum 256 B).  A block is returned if an exact-or-larger free
 * block exists in the bucket; otherwise a fresh cudaMalloc is issued.
 *
 * @code
 *   auto& pool = MemoryPool::global();
 *   void* ptr  = pool.allocate(1024 * 1024);   // 1 MiB
 *   pool.free(ptr);
 * @endcode
 */
class MemoryPool {
public:
    MemoryPool() = default;
    ~MemoryPool();

    // Non-copyable / non-movable singleton pattern
    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /** @brief Returns the process-wide global pool. */
    static MemoryPool& global();

    /**
     * @brief Allocates at least @p bytes from the pool.
     * Rounds up to the next power-of-two bucket for reuse efficiency.
     */
    void* allocate(size_t bytes);

    /**
     * @brief Returns @p ptr to the pool for future reuse.
     * @note  @p ptr must have been allocated via this pool.
     */
    void  free(void* ptr);

    /** @brief Returns total bytes currently held by the pool (free + used). */
    size_t total_reserved() const;

    /** @brief Returns bytes currently allocated (in use). */
    size_t total_allocated() const;

    /** @brief Releases all free blocks back to CUDA. */
    void   release_cache();

private:
    /// Map from bucket_size → list of free device pointers
    std::map<size_t, std::vector<void*>> free_blocks_;
    /// Map from ptr → allocated size (for free routing)
    std::map<void*, size_t>              allocated_ptrs_;
    size_t total_reserved_   = 0;
    size_t total_allocated_  = 0;
    mutable std::mutex       mutex_;

    static size_t round_to_bucket(size_t bytes);
};

// ─── RAII guard ───────────────────────────────────────────────────────────────
/**
 * @brief RAII wrapper that returns memory to the pool on destruction.
 */
class PooledMemory {
public:
    explicit PooledMemory(size_t bytes, MemoryPool& pool = MemoryPool::global());
    ~PooledMemory();
    void* data() const { return ptr_; }

private:
    void*        ptr_;
    size_t       bytes_;
    MemoryPool&  pool_;
};

} // namespace minitensor
