# ──────────────────────────────────────────────────────
# tests/CMakeLists.txt
# ──────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.20)

# Helper macro to register a test executable
macro(mt_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE minitensor)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${MINITENSOR_CUDA_FLAGS}>
    )
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

mt_test(test_tensor   test_tensor.cu)
mt_test(test_matmul   test_matmul.cu)
mt_test(test_conv2d   test_conv2d.cu)
mt_test(test_autograd test_autograd.cu)
mt_test(test_layers   test_layers.cu)
