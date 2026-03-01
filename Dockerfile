# syntax=docker/dockerfile:1.4
# ============================================================
# MiniTensor-CUDA — Multi-stage Docker Build
# Base: NVIDIA CUDA 12.3 + Ubuntu 22.04
# ============================================================

# ─── Stage 1: Builder ────────────────────────────────────────
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

LABEL maintainer="MiniTensor Team"
LABEL description="Production CUDA deep learning engine"

ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_TYPE=Release
ARG ENABLE_NVTX=ON

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake=3.22* \
    ninja-build \
    build-essential \
    git \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# NVTX headers (included in CUDA toolkit >= 11)
ENV NVTX_INCLUDE=/usr/local/cuda/include

WORKDIR /workspace

# Copy source
COPY . .

# Build
RUN cmake -S . -B build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DMINITENSOR_BUILD_TESTS=ON \
        -DMINITENSOR_BUILD_BENCHMARKS=ON \
        -DMINITENSOR_ENABLE_NVTX=${ENABLE_NVTX} \
    && cmake --build build --parallel $(nproc)

# ─── Stage 2: Test runner ────────────────────────────────────
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS tester

WORKDIR /workspace
COPY --from=builder /workspace/build /workspace/build
COPY --from=builder /workspace/scripts /workspace/scripts

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Default: run test suite
CMD ["ctest", "--test-dir", "/workspace/build", "--output-on-failure", "-V"]

# ─── Stage 3: Slim runtime ───────────────────────────────────
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS runtime

WORKDIR /workspace

COPY --from=builder /workspace/build/benchmarks /workspace/benchmarks
COPY --from=builder /workspace/scripts /workspace/scripts
COPY --from=builder /usr/local/lib/libminitensor* /usr/local/lib/ 2>/dev/null || true

RUN ldconfig

CMD ["/workspace/scripts/run_benchmarks.sh"]
