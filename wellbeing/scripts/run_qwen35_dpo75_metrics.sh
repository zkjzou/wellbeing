#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
DPO_EVAL_ENV_NAME="${DPO_EVAL_ENV_NAME:-aiwi-qwen35}"
if [[ -n "${DPO_EVAL_ENV_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$DPO_EVAL_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export VLLM_HTTP_CONCURRENCY="${VLLM_HTTP_CONCURRENCY:-256}"
export VLLM_CONCURRENCY="${VLLM_CONCURRENCY:-256}"
exec conda run --no-capture-output -n "$DPO_EVAL_ENV_NAME" \
  python -m wellbeing.scripts.run_qwen35_dpo75_metrics "$@"
