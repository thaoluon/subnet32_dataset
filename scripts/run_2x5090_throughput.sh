#!/usr/bin/env bash
# 2× GPU throughput: two Ollama daemons (one GPU each) + optional dual dataset_builder workers.
# Requires: ollama, curl; NVIDIA driver for CUDA_VISIBLE_DEVICES.
#
# Usage:
#   ./scripts/run_2x5090_throughput.sh start-ollama    # stop systemd ollama first if it uses 11434
#   ./scripts/run_2x5090_throughput.sh status
#   ./scripts/run_2x5090_throughput.sh stop-ollama
#   ./scripts/run_2x5090_throughput.sh env-print     # export lines for your shell
#   NUM_PAIRS=500 OUT_BASE=./out_dual ./scripts/run_2x5090_throughput.sh run-dual-builders
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/.ollama_cluster_logs"
PIDFILE="${ROOT}/.ollama_cluster.pids"
PORT0="${OLLAMA_PORT_GPU0:-11434}"
PORT1="${OLLAMA_PORT_GPU1:-11435}"
OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"

# Root-started ``ollama serve`` defaults to ~/.ollama (often empty). systemd installs use
# /usr/share/ollama/.ollama/models. Point both cluster daemons at the store that has blobs.
resolve_ollama_models_dir() {
  if [[ -n "${OLLAMA_MODELS:-}" ]]; then
    printf '%s' "${OLLAMA_MODELS}"
    return
  fi
  local sys=/usr/share/ollama/.ollama/models
  local home="${HOME:-/root}/.ollama/models"
  if [[ -d "${sys}/manifests" ]] && [[ -n "$(ls -A "${sys}/manifests" 2>/dev/null)" ]]; then
    printf '%s' "${sys}"
    return
  fi
  if [[ -d "${home}/manifests" ]] && [[ -n "$(ls -A "${home}/manifests" 2>/dev/null)" ]]; then
    printf '%s' "${home}"
    return
  fi
  if [[ -d "${sys}" ]]; then
    printf '%s' "${sys}"
    return
  fi
  printf '%s' "${home}"
}

ollama_tags_up() {
  curl -sf "http://127.0.0.1:${PORT0}/api/tags" >/dev/null && curl -sf "http://127.0.0.1:${PORT1}/api/tags" >/dev/null
}

wait_ollama() {
  local port=$1 name=$2
  local i
  for i in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null; then
      echo "${name} listening on ${port}"
      return 0
    fi
    sleep 1
  done
  echo "Timeout waiting for Ollama on ${port} (${name})" >&2
  return 1
}

cmd_start_ollama() {
  mkdir -p "$LOG_DIR"
  if systemctl is-active --quiet ollama 2>/dev/null; then
    echo "systemd unit 'ollama' is active and usually binds port ${PORT0}." >&2
    echo "Stop it first:  sudo systemctl stop ollama" >&2
    echo "Then re-run:    $0 start-ollama" >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT0}/api/tags" >/dev/null 2>&1; then
    echo "Something already responds on port ${PORT0}. Free it or set OLLAMA_PORT_GPU0." >&2
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT1}/api/tags" >/dev/null 2>&1; then
    echo "Something already responds on port ${PORT1}. Free it or set OLLAMA_PORT_GPU1." >&2
    exit 1
  fi

  OLLAMA_MODELS="$(resolve_ollama_models_dir)"
  export OLLAMA_MODELS
  echo "Using OLLAMA_MODELS=${OLLAMA_MODELS} (override with env OLLAMA_MODELS if needed)"

  : >"$PIDFILE"
  echo "Starting Ollama GPU0 -> 127.0.0.1:${PORT0} (OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL})"
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES=0
    export OLLAMA_HOST="127.0.0.1:${PORT0}"
    export OLLAMA_NUM_PARALLEL
    export OLLAMA_MODELS
    nohup ollama serve >>"${LOG_DIR}/ollama-gpu0.log" 2>&1 &
    echo $! >>"$PIDFILE"
  )
  wait_ollama "$PORT0" "GPU0"

  echo "Starting Ollama GPU1 -> 127.0.0.1:${PORT1}"
  (
    cd "$ROOT"
    export CUDA_VISIBLE_DEVICES=1
    export OLLAMA_HOST="127.0.0.1:${PORT1}"
    export OLLAMA_NUM_PARALLEL
    export OLLAMA_MODELS
    nohup ollama serve >>"${LOG_DIR}/ollama-gpu1.log" 2>&1 &
    echo $! >>"$PIDFILE"
  )
  wait_ollama "$PORT1" "GPU1"

  echo "Both Ollama servers are up. Logs: $LOG_DIR"
  echo "Models are shared via OLLAMA_MODELS; pull once if needed:  OLLAMA_MODELS=\"\${OLLAMA_MODELS}\" ollama pull qwen3:14b"
}

cmd_stop_ollama() {
  if [[ ! -f "$PIDFILE" ]]; then
    echo "No pidfile at $PIDFILE (nothing to stop from this script)." >&2
    exit 0
  fi
  while read -r pid; do
    [[ -z "${pid}" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping pid $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done <"$PIDFILE"
  rm -f "$PIDFILE"
  echo "Stopped cluster PIDs from $PIDFILE"
}

cmd_status() {
  echo "=== Port ${PORT0} ==="
  curl -sf "http://127.0.0.1:${PORT0}/api/tags" | head -c 400 || echo "(not reachable)"
  echo ""
  echo "=== Port ${PORT1} ==="
  curl -sf "http://127.0.0.1:${PORT1}/api/tags" | head -c 400 || echo "(not reachable)"
  echo ""
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
}

cmd_env_print() {
  local md
  md="$(resolve_ollama_models_dir)"
  cat <<EOF
# Paste into your shell before running dataset_builder (or use run-dual-builders):
export OLLAMA_MODELS="${md}"
export OLLAMA_BASE_URLS="http://127.0.0.1:${PORT0},http://127.0.0.1:${PORT1}"
export HF_HUB_ENABLE_HF_TRANSFER="\${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="\${TOKENIZERS_PARALLELISM:-true}"
EOF
}

cmd_run_dual_builders() {
  cd "$ROOT"
  if grep -qE '^[[:space:]]*use_sample_quotas:[[:space:]]*true' "${ROOT}/configs/datasets.yaml"; then
    echo "configs/datasets.yaml has use_sample_quotas: true — quota fills one output tree; do not run two" >&2
    echo "parallel builders (each would target the full 3M). Use:  scripts/run_mdok_3m_quota.sh" >&2
    echo "Ollama round-robin across GPUs still applies to that single process." >&2
    exit 1
  fi
  if [[ -f "${ROOT}/venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${ROOT}/venv/bin/activate"
  fi
  if ! ollama_tags_up; then
    echo "Ollama not reachable on both ports. Run:  $0 start-ollama" >&2
    exit 1
  fi
  export OLLAMA_BASE_URLS="http://127.0.0.1:${PORT0},http://127.0.0.1:${PORT1}"
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

  local pairs="${NUM_PAIRS:-1000}"
  local base="${OUT_BASE:-${ROOT}/outputs_dual}"
  local layout="${OUTPUT_LAYOUT:-by_label}"
  mkdir -p "${base}/shard0" "${base}/shard1"

  echo "Launching two workers: num_pairs=${pairs} layout=${layout} out=${base}"
  set +e
  python -m src.dataset_builder \
    --num-pairs "$pairs" \
    --output-dir "${base}/shard0" \
    --output-layout "$layout" \
    --stream-num-shards 2 \
    --stream-shard-index 0 \
    &
  p0=$!
  python -m src.dataset_builder \
    --num-pairs "$pairs" \
    --output-dir "${base}/shard1" \
    --output-layout "$layout" \
    --stream-num-shards 2 \
    --stream-shard-index 1 \
    &
  p1=$!
  ec=0
  wait "$p0" || ec=1
  wait "$p1" || ec=1
  set -e
  exit "$ec"
}

usage() {
  echo "Usage: $0 {start-ollama|stop-ollama|status|env-print|run-dual-builders}" >&2
  exit 1
}

[[ $# -ge 1 ]] || usage
case "$1" in
  start-ollama) cmd_start_ollama ;;
  stop-ollama) cmd_stop_ollama ;;
  status) cmd_status ;;
  env-print) cmd_env_print ;;
  run-dual-builders) cmd_run_dual_builders ;;
  *) usage ;;
esac
