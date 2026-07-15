include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(minitensor_py MODULE bindings.cpp)
set_target_properties(minitensor_py PROPERTIES OUTPUT_NAME "minitensor")
target_link_libraries(minitensor_py PRIVATE minitensor)
