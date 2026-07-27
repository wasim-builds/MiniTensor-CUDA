cmake_minimum_required(VERSION 3.20)

# ============================================================
# MiniTensor-CUDA — Production Deep Learning Engine
# ============================================================
project(MiniTensor
    VERSION 1.0.0
    DESCRIPTION "Production-level CUDA deep learning engine"
    LANGUAGES CXX
)
if(NOT MINITENSOR_CPU_ONLY)
    enable_language(CUDA)
endif()

# ─── Standards ───────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── Options ─────────────────────────────────────────────────
option(MINITENSOR_CPU_ONLY   "Build without CUDA (stub kernels)" OFF)
option(MINITENSOR_BUILD_TESTS      "Build test suite"            ON)
option(MINITENSOR_BUILD_BENCHMARKS "Build benchmark suite"       ON)
option(MINITENSOR_BUILD_DOCS       "Generate Doxygen docs"       OFF)
option(MINITENSOR_ENABLE_NVTX      "Enable NVTX profiling markers" ON)

# ─── Find packages ───────────────────────────────────────────
if(NOT MINITENSOR_CPU_ONLY)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Detect GPU architecture automatically (requires CUDA 11.6+)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    if(NOT CUDA_ARCH_LIST)
        # Fallback: common architectures (Volta, Turing, Ampere, Ada)
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
    endif()
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ─── NVTX ────────────────────────────────────────────────────
if(MINITENSOR_ENABLE_NVTX AND NOT MINITENSOR_CPU_ONLY)
    find_library(NVTX_LIBRARY nvToolsExt
        PATHS ${CUDAToolkit_LIBRARY_DIR}
        NO_DEFAULT_PATH
    )
    if(NVTX_LIBRARY)
        message(STATUS "NVTX found: ${NVTX_LIBRARY}")
    else()
        message(WARNING "NVTX library not found; profiling markers disabled")
        set(MINITENSOR_ENABLE_NVTX OFF)
    endif()
endif()

# ─── Compiler flags ──────────────────────────────────────────
set(MINITENSOR_CUDA_FLAGS
    --use_fast_math
    --generate-line-info      # for Nsight source correlation
    -Xcompiler=-Wall
    -Xcompiler=-Wextra
    -Xcompiler=-O3
)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND MINITENSOR_CUDA_FLAGS -G -g)
endif()

# ─── Core library ────────────────────────────────────────────
add_subdirectory(src)

# ─── Python Bindings ─────────────────────────────────────────
add_subdirectory(python)

# ─── Tests ───────────────────────────────────────────────────
if(MINITENSOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Benchmarks ──────────────────────────────────────────────
if(MINITENSOR_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ─── Docs ────────────────────────────────────────────────────
if(MINITENSOR_BUILD_DOCS)
    find_package(Doxygen REQUIRED)
    doxygen_add_docs(docs
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        COMMENT "Generating API documentation"
    )
endif()

# ─── Install rules ───────────────────────────────────────────
install(TARGETS minitensor
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
