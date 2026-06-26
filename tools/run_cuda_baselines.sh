#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-/tmp/vgpu_baselines}"
mkdir -p "${out_dir}"

nvcc -cudart shared tools/vector_add_smoke.cu -o "${out_dir}/vector_add_smoke"
nvcc -cudart shared tools/memcpy_baseline.cu -o "${out_dir}/memcpy_baseline"
nvcc -cudart shared tools/matmul_baseline.cu -o "${out_dir}/matmul_baseline"

"${out_dir}/vector_add_smoke"
"${out_dir}/memcpy_baseline"
"${out_dir}/matmul_baseline"

"${out_dir}/vector_add_smoke" &
pid1=$!
"${out_dir}/vector_add_smoke" &
pid2=$!
wait "${pid1}"
wait "${pid2}"

echo "dual-process vector_add baseline passed"

