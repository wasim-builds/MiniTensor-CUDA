# MiniTensor-CUDA

> A production-level CUDA deep learning engine built from scratch in C++17/CUDA.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.3-green.svg)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
[![CI](https://github.com/wasim-builds/MiniTensor-CUDA/actions/workflows/ci.yml/badge.svg)](https://github.com/wasim-builds/MiniTensor-CUDA/actions/workflows/ci.yml)

---

## Features

| Area | Details |
|------|---------|
| **GEMM** | Tiled shared-memory CUDA kernel (32×32 tiles), auto grid/block configuration |
| **Convolution** | im2col → GEMM, supports stride/padding/dilation, fwd + bwd |
| **Activations** | ReLU, Leaky-ReLU, Sigmoid, Tanh — element-wise CUDA kernels |
| **BatchNorm** | Two-pass mean/var, gamma/beta, train/eval mode |
| **Pooling** | MaxPool2D, AvgPool2D with backward masks |
| **Autograd** | Tape-based computation graph, topological backprop |
| **Optimizers** | SGD (momentum), Adam |
| **Benchmarking** | CPU vs GPU GFLOPS table, ResNet-50 conv dims |
| **Profiling** | NVTX range markers (Nsight Compute/Systems compatible) |
| **Memory** | Coalesced allocation, simple CUDA memory pool |
| **Build** | CMake 3.20+, auto GPU arch detection |
| **Docker** | Multi-stage image (builder → tester → slim runtime) |

---

## Architecture

```
MiniTensor-CUDA/
├── include/minitensor/          # Public headers
│   ├── tensor.hpp               # Core Tensor class
│   ├── memory.hpp               # Memory pool
│   ├── kernels.hpp              # Kernel launcher API
│   ├── autograd.hpp             # Variable / Node / Graph
│   ├── optimizer.hpp            # Optimizer base
│   └── layers/                  # Layer headers
│       ├── linear.hpp
│       ├── conv2d.hpp
│       ├── activation.hpp
│       ├── batchnorm.hpp
│       └── pooling.hpp
├── src/
│   ├── core/                    # Tensor, memory implementation
│   ├── kernels/                 # CUDA kernels (matmul, conv2d, …)
│   ├── layers/                  # Layer forward/backward
│   ├── autograd/                # Graph traversal, backward pass
│   └── optimizer/               # SGD, Adam
├── tests/                       # CTest suite
├── benchmarks/                  # CPU vs GPU benchmarks
├── scripts/                     # run_benchmarks.sh, profile.sh
├── docs/                        # Doxygen output
├── CMakeLists.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| CMake | ≥ 3.20 |
| CUDA Toolkit | ≥ 11.6 (12.x recommended) |
| GCC / Clang | C++17 capable |
| NVIDIA Driver | ≥ 520 |
| Docker (optional) | 20.10+ with `nvidia-container-toolkit` |

---

## Build

```bash
git clone https://github.com/youruser/MiniTensor-CUDA.git
cd MiniTensor-CUDA
mkdir build && cd build

# Release build (GPU)
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# CPU-only build (no GPU required)
cmake .. -DCMAKE_BUILD_TYPE=Release -DMINITENSOR_CPU_ONLY=ON
make -j$(nproc)
```

---

## Run Tests

```bash
cd build
ctest --output-on-failure -V
```

---

## Run Benchmarks

```bash
cd build
./benchmarks/bench_matmul
./benchmarks/bench_conv2d
./benchmarks/bench_e2e

# Or via script from project root:
./scripts/run_benchmarks.sh
```

Sample output:
```
=== Matrix Multiplication Benchmark (CUDA vs CPU) ===
   N    | CPU (ms) | GPU (ms) | Speedup |  GFLOPS (GPU)
--------+----------+----------+---------+--------------
   512  |   185.3  |    1.82  |  101.8x |    147.6
  1024  |  1480.1  |    4.41  |  335.6x |    487.1
  2048  |  11842.0 |   16.23  |  729.6x |    529.8
  4096  |  94836.0 |   96.15  |  986.4x |    714.3
```

---

## Profiling with Nsight

```bash
# Nsight Compute (kernel-level metrics)
./scripts/profile.sh bench_matmul

# Nsight Systems (timeline)
nsys profile --trace=cuda,nvtx ./build/benchmarks/bench_e2e
```

NVTX range markers are embedded at kernel boundaries for source-correlated profiling.

---

## Docker

```bash
# Build all stages
docker build -t minitensor-cuda .

# Run tests
docker compose run --rm test

# Run benchmarks
docker compose run --rm benchmark

# Interactive dev shell
docker compose run --rm dev
```

---

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `MINITENSOR_CPU_ONLY` | OFF | Stub CUDA kernels with host code |
| `MINITENSOR_BUILD_TESTS` | ON | Build CTest suite |
| `MINITENSOR_BUILD_BENCHMARKS` | ON | Build benchmark binaries |
| `MINITENSOR_BUILD_DOCS` | OFF | Generate Doxygen HTML docs |
| `MINITENSOR_ENABLE_NVTX` | ON | Embed NVTX profiling markers |

---

## License

MIT © 2026 MiniTensor Contributors
