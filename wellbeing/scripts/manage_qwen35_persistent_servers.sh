#!/usr/bin/env bash
# Manage two independent persistent Qwen3.5-35B vLLM servers.

set -euo pipefail

ACTION="${1:-status}"
TARGET="${2:-all}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/data/kaijian_zou/miniconda3/envs/aiwi-qwen35}"
VLLM_BIN="${VLLM_BIN:-$CONDA_ENV_DIR/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/data/huggingface/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-35b-a3b}"
RUNTIME_DIR="${QWEN35_SERVER_RUNTIME_DIR:-/tmp/qwen35_persistent_servers_${USER}}"

HOST="${VLLM_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_LORAS="${VLLM_MAX_LORAS:-4}"
MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-64}"

mkdir -p "$RUNTIME_DIR"

usage() {
    cat <<'EOF'
Usage: manage_qwen35_persistent_servers.sh ACTION [TARGET]

ACTION: start | stop | restart | status | logs
TARGET: a | b | all (default: all)

Server a uses GPUs 0-3 at http://127.0.0.1:8000
Server b uses GPUs 4-7 at http://127.0.0.1:8001
EOF
}

server_gpus() {
    case "$1" in
        a) echo "0,1,2,3" ;;
        b) echo "4,5,6,7" ;;
        *) return 1 ;;
    esac
}

server_port() {
    case "$1" in
        a) echo "8000" ;;
        b) echo "8001" ;;
        *) return 1 ;;
    esac
}

pid_file() {
    echo "$RUNTIME_DIR/server_$1.pid"
}

log_file() {
    echo "$RUNTIME_DIR/server_$1.log"
}

read_live_pid() {
    local server="$1"
    local file pid
    file="$(pid_file "$server")"
    [[ -f "$file" ]] || return 1
    read -r pid < "$file"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    printf '%s\n' "$pid"
}

start_server() {
    local server="$1"
    local gpus port pid log triton_dir tmp_dir
    if pid="$(read_live_pid "$server")"; then
        echo "Server $server is already running (PID $pid)."
        return
    fi

    gpus="$(server_gpus "$server")"
    port="$(server_port "$server")"
    log="$(log_file "$server")"
    triton_dir="$RUNTIME_DIR/triton_$server"
    tmp_dir="$RUNTIME_DIR/tmp_$server"
    mkdir -p "$triton_dir" "$tmp_dir"
    : > "$log"

    echo "Starting server $server on GPUs $gpus, port $port..."
    nohup setsid env \
        CUDA_VISIBLE_DEVICES="$gpus" \
        HF_HOME="${HF_HOME:-/data/huggingface}" \
        TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/huggingface}" \
        PATH="$CONDA_ENV_DIR/bin:$PATH" \
        LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        PYTHONPATH="$REPO_ROOT:$REPO_ROOT/wellbeing${PYTHONPATH:+:$PYTHONPATH}" \
        TRITON_CACHE_DIR="$triton_dir" \
        TMPDIR="$tmp_dir" \
        NCCL_IGNORE_CPU_AFFINITY=1 \
        NCCL_CUMEM_ENABLE=0 \
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
        "$VLLM_BIN" serve "$MODEL_PATH" \
            --host "$HOST" \
            --port "$port" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --tensor-parallel-size 4 \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
            --safetensors-load-strategy prefetch \
            --enable-prefix-caching \
            --enable-prompt-embeds \
            --enable-lora \
            --max-loras "$MAX_LORAS" \
            --max-lora-rank "$MAX_LORA_RANK" \
            > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$(pid_file "$server")"
    echo "Server $server launched as PID $pid; log: $log"
}

stop_server() {
    local server="$1"
    local pid attempt
    if ! pid="$(read_live_pid "$server")"; then
        echo "Server $server is not running."
        rm -f "$(pid_file "$server")"
        return
    fi

    echo "Stopping server $server process group (leader PID $pid)..."
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for attempt in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$(pid_file "$server")"
            echo "Server $server stopped."
            return
        fi
        sleep 1
    done
    echo "Server $server did not stop within 60 seconds; forcing its process group down." >&2
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    for attempt in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$(pid_file "$server")"
            echo "Server $server stopped after forced cleanup."
            return
        fi
        sleep 1
    done
    echo "Server $server process $pid still exists after forced cleanup." >&2
    return 1
}

status_server() {
    local server="$1"
    local pid port
    port="$(server_port "$server")"
    if pid="$(read_live_pid "$server")"; then
        if curl --silent --fail --max-time 2 "http://$HOST:$port/health" >/dev/null; then
            echo "Server $server: READY (PID $pid, GPUs $(server_gpus "$server"), port $port)"
        else
            echo "Server $server: LOADING (PID $pid, GPUs $(server_gpus "$server"), port $port)"
        fi
    else
        echo "Server $server: STOPPED (GPUs $(server_gpus "$server"), port $port)"
    fi
}

logs_server() {
    local server="$1"
    local log
    log="$(log_file "$server")"
    if [[ -f "$log" ]]; then
        echo "===== server $server: $log ====="
        tail -n 80 "$log"
    else
        echo "No log for server $server."
    fi
}

case "$TARGET" in
    a|b) SERVERS=("$TARGET") ;;
    all) SERVERS=(a b) ;;
    *) usage >&2; exit 2 ;;
esac

case "$ACTION" in
    start)
        for server in "${SERVERS[@]}"; do start_server "$server"; done
        ;;
    stop)
        for server in "${SERVERS[@]}"; do stop_server "$server"; done
        ;;
    restart)
        for server in "${SERVERS[@]}"; do stop_server "$server"; done
        for server in "${SERVERS[@]}"; do start_server "$server"; done
        ;;
    status)
        for server in "${SERVERS[@]}"; do status_server "$server"; done
        ;;
    logs)
        for server in "${SERVERS[@]}"; do logs_server "$server"; done
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
