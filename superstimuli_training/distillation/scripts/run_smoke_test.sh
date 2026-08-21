#!/usr/bin/env bash
# Run one optimizer step on the local Qwen 3.5 35B-A3B checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/kaijian_zou/miniconda3}"
CONDA_ENV="${CONDA_ENV:-wellbeing_distillation}"
NUM_GPUS="${NUM_GPUS:-8}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$WORK_DIR"

export FORCE_TORCHRUN=1
export NPROC_PER_NODE="$NUM_GPUS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_DIR/.cache}"
export HF_HOME="${HF_HOME:-$WORK_DIR/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$WORK_DIR/.cache/matplotlib}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$WORK_DIR/.cache/triton}"
export WANDB_PROJECT="${WANDB_PROJECT:-wellbeing-distillation}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

llamafactory-cli train configs/qwen35_35b_a3b_lora_sft_smoke.yaml
