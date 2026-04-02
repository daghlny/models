#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NPROC="${NPROC:-$(python3 -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "$NPROC" -lt 1 ]]; then
  echo "No CUDA device found."
  exit 1
fi

export HF_ENDPOINT="https://hf-mirror.com"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "$NPROC" -eq 1 ]]; then
  python3 gpt2.py
else
  torchrun --standalone --nproc_per_node="$NPROC" gpt2.py
fi
