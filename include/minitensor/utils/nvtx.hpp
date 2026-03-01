/**
 * @file nvtx.hpp
 * @brief NVTX profiling range markers for Nsight Compute / Systems.
 *
 * Usage: NVTX_RANGE("kernel_name") at the top of a host launcher function.
 * If NVTX is disabled at build time, the macro expands to nothing.
 */
#pragma once

#ifdef MINITENSOR_ENABLE_NVTX
#  include <nvToolsExt.h>

namespace minitensor {
namespace detail {

/// RAII NVTX range — pushes on construction, pops on destruction
struct NvtxRange {
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange()                          { nvtxRangePop(); }
};

} // namespace detail
} // namespace minitensor

#  define NVTX_RANGE(name) \
     ::minitensor::detail::NvtxRange _nvtx_range_##__LINE__(name)

#  define NVTX_MARK(name) nvtxMarkA(name)

#else // MINITENSOR_ENABLE_NVTX not set
#  define NVTX_RANGE(name) (void)0
#  define NVTX_MARK(name)  (void)0
#endif
