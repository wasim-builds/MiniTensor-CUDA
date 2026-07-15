#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <minitensor/tensor.hpp>

namespace py = pybind11;
using namespace minitensor;

PYBIND11_MODULE(minitensor_py, m) {
    m.doc() = "MiniTensor-CUDA Python Bindings"; // optional module docstring

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float16", DType::Float16)
        .value("Int32", DType::Int32)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def("to", &Tensor::to)
        .def("shape", &Tensor::shape)
        .def_static("zeros", &Tensor::zeros, 
                    py::arg("shape"), 
                    py::arg("dtype") = DType::Float32, 
                    py::arg("device") = Device::CUDA);
}
