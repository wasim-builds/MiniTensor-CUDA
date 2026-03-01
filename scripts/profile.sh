#!/usr/bin/env bash
# =============================================================
# scripts/profile.sh
# Profile a MiniTensor benchmark binary with Nsight Compute (ncu).
# Usage: ./scripts/profile.sh [binary_name]  (default: bench_matmul)
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${ROOT_DIR}/build"
RESULTS_DIR="${ROOT_DIR}/results"
mkdir -p "$RESULTS_DIR"

BINARY="${1:-bench_matmul}"
BIN_PATH="${BUILD_DIR}/benchmarks/${BINARY}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${RESULTS_DIR}/${BINARY}_${TIMESTAMP}.ncu-rep"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Binary not found: $BIN_PATH"
    echo "Run: cmake --build build --target ${BINARY}"
    exit 1
fi

NCU=$(command -v ncu || command -v nv-nsight-cu-cli || echo "")
if [ -z "$NCU" ]; then
    echo "Error: Nsight Compute (ncu) not found in PATH."
    echo "Install CUDA toolkit >= 11.0 or add /usr/local/cuda/bin to PATH."
    exit 1
fi

echo "Profiling: $BIN_PATH"
echo "Report:    $REPORT"
echo ""

# Collect key metrics:
#  - sm__throughput.avg.pct_of_peak_sustained_elapsed  (SM utilization)
#  - dram__throughput.avg.pct_of_peak_sustained_elapsed (memory BW)
#  - l1tex__t_sector_hit_rate.pct                       (L1 hit rate)
#  - smsp__sass_thread_inst_executed_op_fp32_pred_on.sum (FP32 ops)
"$NCU" \
    --target-processes all \
    --set full \
    --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
smsp__sass_thread_inst_executed_op_fp32_pred_on.sum" \
    --export "$REPORT" \
    "$BIN_PATH"

echo ""
echo "Profile saved: $REPORT"
echo "Open in Nsight Compute UI or view with:"
echo "  ncu --import $REPORT"
