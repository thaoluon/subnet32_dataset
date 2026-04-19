#!/usr/bin/env bash
# Full mdok-style quota build: sample_targets in configs/datasets.yaml (sums to 3M when default).
# Requires use_sample_quotas: true. Stops when all buckets are full (--num-pairs is only a CLI placeholder).
#
# Prereqs: dual Ollama (./scripts/run_2x5090_throughput.sh start-ollama) on the same ports as below;
# Azure + .env as in configs/models.yaml.
#
# Do not run two builders on the same OUT_DIR (e.g. openai_only + ollama_locked together): they corrupt
# sample_counts and IDs. The builder refuses a second process on the same directory (POSIX lock).
#
# Ollama URLs match scripts/run_2x5090_throughput.sh (OLLAMA_PORT_GPU0 / OLLAMA_PORT_GPU1).
# By default this script **forces** two hosts so a stale single-URL OLLAMA_BASE_URLS in your shell
# cannot disable the second GPU. Set OLLAMA_USE_DUAL_CLUSTER=0 to honor existing OLLAMA_BASE_URLS.
#
# Usage:
#   ./scripts/run_mdok_3m_quota.sh
#   OUT_DIR=./out_mdok_3m_subnet32 ./scripts/run_mdok_3m_quota.sh --append
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "${ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/venv/bin/activate"
fi

PORT0="${OLLAMA_PORT_GPU0:-11434}"
PORT1="${OLLAMA_PORT_GPU1:-11435}"
DEFAULT_OLLAMA_URLS="http://127.0.0.1:${PORT0},http://127.0.0.1:${PORT1}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ "${OLLAMA_USE_DUAL_CLUSTER:-1}" != "0" ]]; then
  export OLLAMA_BASE_URLS="${DEFAULT_OLLAMA_URLS}"
  echo "run_mdok_3m_quota: OLLAMA_BASE_URLS=${OLLAMA_BASE_URLS} (set OLLAMA_USE_DUAL_CLUSTER=0 to use env/yaml only)" >&2
else
  export OLLAMA_BASE_URLS="${OLLAMA_BASE_URLS:-${DEFAULT_OLLAMA_URLS}}"
  echo "run_mdok_3m_quota: OLLAMA_BASE_URLS=${OLLAMA_BASE_URLS}" >&2
fi

OUT_DIR="${OUT_DIR:-./out_mdok_3m_subnet32}"

exec python -m src.dataset_builder \
  --num-pairs 1 \
  --output-dir "${OUT_DIR}" \
  "$@"
