#!/usr/bin/env bash
# Run four-GPU regular-LoRA DPO over soft-prompt versus baseline responses.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/kaijian_zou/miniconda3}"
CONDA_ENV="${CONDA_ENV:-wellbeing_distillation}"
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/qwen35_35b_a3b_lora_dpo_4gpu.yaml}"

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
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY must be exported when WANDB_MODE=online." >&2
    exit 1
fi

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

llamafactory-cli train "$TRAIN_CONFIG"
