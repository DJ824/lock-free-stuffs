#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_EXE="${1:-$ROOT_DIR/build/atomic_queue_benchmarks}"
CPU_LIST="${ATOMIC_QUEUE_CPU_LIST:-14,15}"
RUNS="${ATOMIC_QUEUE_BENCHMARK_RUNS:-${N:-33}}"
RESULTS_DIR="${ATOMIC_QUEUE_RESULTS_DIR:-$ROOT_DIR/bench/results}"
STAMP="$(date --utc +%Y%m%dT%H%M%S)"
SAFE_CPU_LIST="${CPU_LIST//,/__}"
RAW_RESULTS="$RESULTS_DIR/atomic_queue_benchmarks.${SAFE_CPU_LIST}.${STAMP}.raw.txt"
SUMMARY_RESULTS="$RESULTS_DIR/atomic_queue_benchmarks.${SAFE_CPU_LIST}.${STAMP}.summary.txt"

if [[ ! -x "$BENCHMARK_EXE" ]]; then
    echo "Benchmark executable not found or not executable: $BENCHMARK_EXE" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

source "$ROOT_DIR/third_party/atomic_queue/scripts/benchmark.sh"

trap epilogue EXIT
prologue

echo "Running atomic_queue benchmarks"
echo "Executable: $BENCHMARK_EXE"
echo "Hardware threads: $CPU_LIST"
echo "Runs: $RUNS"
echo "Raw results: $RAW_RESULTS"
echo "Summary: $SUMMARY_RESULTS"

: >"$RAW_RESULTS"
for ((i = 1; i <= RUNS; ++i)); do
    printf '[%d/%d] ' "$i" "$RUNS" | tee -a "$RAW_RESULTS"
    sudo env ATOMIC_QUEUE_CPU_LIST="$CPU_LIST" chrt -f 50 "$BENCHMARK_EXE" | tee -a "$RAW_RESULTS"
done

python3 "$ROOT_DIR/third_party/atomic_queue/scripts/stats.py" <"$RAW_RESULTS" | tee "$SUMMARY_RESULTS"
