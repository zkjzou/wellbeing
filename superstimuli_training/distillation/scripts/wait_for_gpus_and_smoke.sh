#!/usr/bin/env bash
# Wait for idle GPUs, then launch the one-step LoRA smoke test.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_USED_MIB="${MAX_USED_MIB:-4096}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-7200}"
elapsed=0

while true; do
    gpu_rows=()
    if gpu_output="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)"; then
        mapfile -t gpu_rows < <(printf '%s\n' "$gpu_output")
    fi

    idle_indices=()
    usage_summary=()
    for row in "${gpu_rows[@]}"; do
        IFS=',' read -r index used <<< "$row"
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        usage_summary+=("${index}:${used}")
        if (( used <= MAX_USED_MIB )); then
            idle_indices+=("$index")
        fi
    done

    if (( ${#idle_indices[@]} >= NUM_GPUS )); then
        selected=("${idle_indices[@]:0:NUM_GPUS}")
        selected_csv="$(IFS=,; echo "${selected[*]}")"
        echo "GPUs $selected_csv are available; starting the one-step smoke test."
        exec env CUDA_VISIBLE_DEVICES="$selected_csv" NUM_GPUS="$NUM_GPUS" \
            bash "$SCRIPT_DIR/run_smoke_test.sh"
    fi

    if (( elapsed >= MAX_WAIT_SECONDS )); then
        echo "Timed out after ${MAX_WAIT_SECONDS}s waiting for $NUM_GPUS GPUs." >&2
        exit 2
    fi

    if [[ "${#gpu_rows[@]}" -gt 0 ]]; then
        printf 'Need %s idle GPUs (index:MiB used: %s); checking again in %ss.\n' \
            "$NUM_GPUS" "${usage_summary[*]}" "$POLL_SECONDS"
    else
        printf 'Waiting for %s GPUs to become visible; checking again in %ss.\n' "$NUM_GPUS" "$POLL_SECONDS"
    fi
    sleep "$POLL_SECONDS"
    elapsed=$((elapsed + POLL_SECONDS))
done
