# Throughput-oriented env for 2x RTX 5090 + fast CPU + NVMe (Vast / runpod style hosts).
# 1) Start TWO Ollama daemons (one GPU each), then 2) run one or two dataset_builder workers.
#
# === Terminal A — GPU 0, port 11434 ===
#   $env:CUDA_VISIBLE_DEVICES = "0"
#   $env:OLLAMA_HOST = "127.0.0.1:11434"
#   ollama serve
#
# === Terminal B — GPU 1, port 11435 ===
#   $env:CUDA_VISIBLE_DEVICES = "1"
#   $env:OLLAMA_HOST = "127.0.0.1:11435"
#   ollama serve
#
# Pull the same model tags on BOTH (e.g. ollama pull qwen3:14b).
#
# === models.yaml ===
# Uncomment and set:
#   ollama_base_urls:
#     - http://127.0.0.1:11434
#     - http://127.0.0.1:11435
#
# Optional Ollama server-side parallelism (set BEFORE each ollama serve; values depend on VRAM/model size):
#   $env:OLLAMA_NUM_PARALLEL = "2"   # try 1–2 for 14B-class on 32GB; reduce if OOM

$ErrorActionPreference = "Stop"
Write-Host "Setting process environment for dataset build (HF + Python threads)..."

# Faster Hub downloads when hf_transfer is installed: pip install huggingface_hub[hf_transfer]
if (-not $env:HF_HUB_ENABLE_HF_TRANSFER) {
    $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
}

# Use several CPU threads for tokenization / data prep (tune 8–16 on 24c host)
if (-not $env:TOKENIZERS_PARALLELISM) {
    $env:TOKENIZERS_PARALLELISM = "true"
}

# Example: single process using round-robin across both Ollamas (configs must list ollama_base_urls)
# cd subnet32_dataset
# python -m src.dataset_builder --num-pairs 1000 --output-dir .\outputs --verbose
#
# Example: TWO parallel builders (merge JSONL with append + distinct output dirs, then combine):
# Worker 0: set datasets.yaml stream_num_shards: 2, stream_shard_index: 0
# Worker 1: set stream_shard_index: 1 (same seed), run second process with --output-dir .\out_shard1
#
Write-Host "Done. Start Ollama on two ports, update models.yaml ollama_base_urls, then run dataset_builder."
