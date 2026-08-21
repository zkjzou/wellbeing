#!/usr/bin/env bash
set -euo pipefail

cd /data/kaijian_zou/wellbeing
export VLLM_HTTP_CONCURRENCY="${VLLM_HTTP_CONCURRENCY:-256}"
exec conda run --no-capture-output -n aiwi-qwen35 \
  python -m wellbeing.scripts.run_qwen35_threeway_wellbeing "$@"
