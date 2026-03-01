#!/usr/bin/env bash
# =============================================================
# scripts/run_benchmarks.sh
# Run all MiniTensor benchmarks and save results to results/
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================"
echo " MiniTensor-CUDA Benchmark Suite"
echo " $(date)"
echo "======================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total \
           --format=csv,noheader 2>/dev/null | \
    awk -F, '{printf " GPU:    %s\n Driver: %s\n Memory: %s\n", $1, $2, $3}'
echo "------------------------------------------------------"

run_bench() {
    local name="$1"
    local bin="${BUILD_DIR}/benchmarks/${name}"

    if [ ! -f "$bin" ]; then
        echo "[SKIP] $name — binary not found (build first)"
        return
    fi

    echo ""
    echo ">>> Running: $name"
    local out="${RESULTS_DIR}/${name}_${TIMESTAMP}.txt"
    "$bin" 2>&1 | tee "$out"
    echo "    Saved → $out"
}

run_bench bench_matmul
run_bench bench_conv2d
run_bench bench_e2e

echo ""
echo "======================================================"
echo " All benchmarks complete. Results in: $RESULTS_DIR"
echo "======================================================"
