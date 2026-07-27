# ──────────────────────────────────────────────────────
# src/CMakeLists.txt
# ──────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.20)

# Collect all CUDA/C++ sources
file(GLOB_RECURSE MT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/layers/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/autograd/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/optimizer/*.cu"
)

# Build shared library
if(MINITENSOR_CPU_ONLY)
    set_source_files_properties(${MT_SOURCES} PROPERTIES LANGUAGE CXX)
endif()
add_library(minitensor SHARED ${MT_SOURCES})

target_include_directories(minitensor
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# CUDA compile options
if(NOT MINITENSOR_CPU_ONLY)
    target_compile_options(minitensor PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${MINITENSOR_CUDA_FLAGS}>
    )
endif()

# NVTX definition
if(MINITENSOR_ENABLE_NVTX AND NVTX_LIBRARY)
    target_compile_definitions(minitensor PUBLIC MINITENSOR_ENABLE_NVTX)
    target_link_libraries(minitensor PUBLIC ${NVTX_LIBRARY})
endif()

# CUDA runtime
if(NOT MINITENSOR_CPU_ONLY)
    target_link_libraries(minitensor PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()

# Enable separate compilation (needed for device lambdas)
if(NOT MINITENSOR_CPU_ONLY)
    set_target_properties(minitensor PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()
