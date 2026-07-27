if(MINITENSOR_CPU_ONLY)
    file(GLOB_RECURSE BENCH_SOURCES "*.cu")
    set_source_files_properties(${BENCH_SOURCES} PROPERTIES LANGUAGE CXX)
endif()
# ──────────────────────────────────────────────────────
# benchmarks/CMakeLists.txt
# ──────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.20)

macro(mt_bench name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE minitensor)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${MINITENSOR_CUDA_FLAGS}>
    )
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endmacro()

mt_bench(bench_matmul  bench_matmul.cu)
mt_bench(bench_conv2d  bench_conv2d.cu)
mt_bench(bench_e2e     bench_e2e.cu)
