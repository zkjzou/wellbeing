#!/usr/bin/env bash
# Sample the 2,500 D2-extension conversations from the persistent base model.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/data/kaijian_zou/miniconda3/envs/aiwi-qwen35}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ENV_DIR/bin/python}"

export PATH="$CONDA_ENV_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/wellbeing${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONUNBUFFERED=1
export VLLM_HTTP_CONCURRENCY="${VLLM_HTTP_CONCURRENCY:-128}"
export VLLM_MAX_RETRIES="${VLLM_MAX_RETRIES:-5}"
export VLLM_URLS="${VLLM_URLS:-http://127.0.0.1:8000,http://127.0.0.1:8001}"

IFS=',' read -r -a SERVER_URLS <<< "$VLLM_URLS"
for server_url in "${SERVER_URLS[@]}"; do
    server_url="${server_url%/}"
    if ! curl --silent --fail --max-time 5 "$server_url/health" >/dev/null; then
        echo "Persistent vLLM server at $server_url is not ready." >&2
        exit 1
    fi
done

cd "$REPO_ROOT"
exec "$PYTHON_BIN" wellbeing/scripts/generate_d2_extension_2500_resumable.py \
    --inference-model-key qwen35-35b-a3b-server-judge \
    --output-model-key qwen35-35b-a3b \
    --system-prompt "You are an assistant." \
    --concurrency "$VLLM_HTTP_CONCURRENCY" \
    --batch-size 128
