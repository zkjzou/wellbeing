#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export VLLM_HTTP_CONCURRENCY="${VLLM_HTTP_CONCURRENCY:-256}"
export VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-256}"
export VLLM_PAYLOAD_WORKERS="${VLLM_PAYLOAD_WORKERS:-32}"
export VLLM_PROMPT_EMBED_DEVICE="${VLLM_PROMPT_EMBED_DEVICE:-cpu}"
exec conda run --no-capture-output -n aiwi-qwen35 \
  python -m wellbeing.scripts.run_qwen35_psychopathy_pilot50 "$@"
