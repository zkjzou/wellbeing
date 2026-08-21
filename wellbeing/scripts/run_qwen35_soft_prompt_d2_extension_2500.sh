#!/usr/bin/env bash
# Generate the 2,500 D2-extension conversations with Qwen3.5-35B soft prompt 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/data/kaijian_zou/miniconda3}"
CONDA_ENV="${CONDA_ENV:-aiwi-qwen35}"
MODEL_KEY="qwen35-35b-a3b-euphorics-top1"
DATASET="d2_extension_2500"
DATASET_DIR="$REPO_ROOT/wellbeing/datasets/experiences/$DATASET"
RESPONSES_DIR="$DATASET_DIR/responses"
SOFT_PROMPT_PATH="$REPO_ROOT/superstimuli_training/soft_prompt/optimized_soft_prompts/euphorics/qwen35-35b-a3b_euphorics_soft_prompt_top_1.pt"
JOB_TAG="${SLURM_JOB_ID:-local_$$}"
LOCAL_CACHE="${AIWI_SOFT_PROMPT_CACHE:-/tmp/aiwi_soft_prompt_${JOB_TAG}}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/wellbeing${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# Four independent TP=2 replicas reduce tensor-parallel communication while
# retaining enough KV cache for long concurrent conversations.
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export VLLM_REQUEST_TIMEOUT_SECONDS="${VLLM_REQUEST_TIMEOUT_SECONDS:-1800}"
export VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-1800}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
export VLLM_SAFETENSORS_LOAD_STRATEGY="${VLLM_SAFETENSORS_LOAD_STRATEGY:-prefetch}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_CACHE/triton}"
export WELLBEING_EVALS_CACHE_DIR="${WELLBEING_EVALS_CACHE_DIR:-$LOCAL_CACHE/wellbeing_evals}"
export TMPDIR="${TMPDIR:-$LOCAL_CACHE/tmp}"

mkdir -p "$RESPONSES_DIR" "$WELLBEING_EVALS_CACHE_DIR"
for SHARD_INDEX in 0 1 2 3; do
    mkdir -p "$TRITON_CACHE_DIR/shard$SHARD_INDEX"
    mkdir -p "$TMPDIR/shard$SHARD_INDEX"
done

if [[ ! -f "$SOFT_PROMPT_PATH" ]]; then
    echo "Soft prompt 1 not found: $SOFT_PROMPT_PATH" >&2
    exit 1
fi

"$PYTHON_BIN" - "$DATASET_DIR/experiences_text.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as file:
    prompts = json.load(file)["prompts"]
if len(prompts) != 2500:
    raise SystemExit(f"Expected 2500 prompts, found {len(prompts)}")
ids = [item["final_id"] for item in prompts]
if len(set(ids)) != len(ids):
    raise SystemExit("Prompt final_id values are not unique")
print(f"Validated {len(prompts)} prompts with unique final_id values")
PY

cd "$REPO_ROOT"
echo "Generating $DATASET with $MODEL_KEY"
echo "Sampling: temperature=0.7 top_p=0.9 max_tokens=4096 n=1"
echo "Serving: 4 x TP=2, concurrency=32 per server (128 total)"
echo "vLLM memory utilization: $VLLM_GPU_MEMORY_UTILIZATION"

GPU_GROUPS=("0,1" "2,3" "4,5" "6,7")
SHARD_PIDS=()
for SHARD_INDEX in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$SHARD_INDEX]}" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR/shard$SHARD_INDEX" \
    TMPDIR="$TMPDIR/shard$SHARD_INDEX" \
    "$PYTHON_BIN" wellbeing/scripts/generate_d2_extension_2500_resumable.py \
        --shard-index "$SHARD_INDEX" --num-shards 4 &
    SHARD_PIDS+=("$!")
done

SHARD_STATUS=0
for SHARD_PID in "${SHARD_PIDS[@]}"; do
    wait "$SHARD_PID" || SHARD_STATUS=$?
done
if [[ "$SHARD_STATUS" -ne 0 ]]; then
    echo "At least one generation shard failed (status=$SHARD_STATUS)." >&2
    echo "Rerun this launcher to resume from the last 25-item checkpoints." >&2
    exit "$SHARD_STATUS"
fi

"$PYTHON_BIN" wellbeing/scripts/merge_d2_extension_2500_shards.py

echo "Completed: $RESPONSES_DIR/$MODEL_KEY.json"
