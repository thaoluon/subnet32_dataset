# Subnet32 dataset generator

Python tool that builds **human vs machine** training/eval JSONL from the **Pile** (streaming), with optional **local Ollama** and/or **OpenAI** generation, augmentation, and production-oriented **quota** and **dedup** controls.

---

## Principles

1. **Same-source (validator-style)**  
   Human text and AI text are tied to the **same Pile document**: a prefix is taken from the start of the document’s sentences; an LLM writes a **continuation**. That keeps topic/domain aligned so the main difference is **human vs model**, not random topic mismatch.

2. **Binary labels for detectors**  
   - `label_int: 0` → human  
   - `label_int: 1` → machine (includes **direct_ai**, **hard_ai**, and **mixed** rows)  
   **`sample_kind`** (`human` / `direct_ai` / `hard_ai` / `mixed`) is metadata for stratified eval and calibration—not a third class.

3. **Quota mode (optional)**  
   Set `use_sample_quotas: true` in `configs/datasets.yaml` and define **`sample_targets`** for `human`, `direct_ai`, `hard_ai`, `mixed`. The run stops when each bucket hits its target (ignores `--num-pairs` as the stop condition). See comments in that file for example 3M recipes.

4. **Split isolation**  
   **Train / val / test / stress** are chosen **once per `source_doc_id`** and reused for every row derived from that document, reducing leakage across splits.

5. **Augmentation**  
   Light **char-level** noise and optional **adjective removal** on **both** human and AI sides (see `configs/augmentation.yaml`) to reduce trivial hash matching on raw Pile.

6. **Hard AI**  
   After a direct completion, an optional **LLM rewrite** step uses rotating prompts (`hard_transform_systems` in `configs/generation.yaml`) for harder negatives.

7. **Mixed rows**  
   One sample = **human span ∥ joiner ∥ AI continuation** from the same doc; stored as `label` `"ai"`, `label_int` **1**, `sample_kind` **`mixed`**.

8. **Throughput**  
   Multiple Ollama hosts (**`ollama_base_urls`** in `configs/models.yaml`) are **round-robin** per request. Optional **Pile stream sharding** (`stream_num_shards` / `stream_shard_index`) for parallel processes. See `scripts/run_2x5090_throughput.ps1`.

---

## Install

```bash
cd subnet32_dataset
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Use Python **3.10+**. On first run, NLTK may download tokenizer data.

---

## Production run (~3M rows for mdok / Subnet32)

Your repo is already configured with **`sample_targets`** that sum to **3M** (`1.5M` human + `0.7M` direct_ai + `0.5M` hard_ai + `0.3M` mixed). To **actually** build that dataset, turn on quota mode once, then run the command below.

### Step 1 — Enable quota mode

In **`configs/datasets.yaml`**, **`use_sample_quotas`** must be **`true`** (it is **on** in the default repo config for the mdok 3M recipe).

(Leave `sample_targets` as-is unless you change the recipe.) When this is `true`, **`--num-pairs` does not stop the run**—the job finishes when every bucket hits its target (or `max_documents_scan` is hit).

Optional for big RAM hosts: raise **`shuffle_buffer_size`** (e.g. `250000`) for a stronger Pile shuffle.

### Step 2 — Models and secrets

1. **`configs/models.yaml`**: confirm **Ollama** and/or **OpenAI** `train_generators` / `stress_generators`. For **2× GPU**, uncomment **`ollama_base_urls`** with two ports (see `scripts/run_2x5090_throughput.ps1`).  
2. **OpenAI or Azure**: platform API — `export OPENAI_API_KEY=...` (or PowerShell `$env:OPENAI_API_KEY`). For **Azure OpenAI**, set **`openai.use_azure: true`**, **`api_key_env: AZURE_OPENAI_API_KEY`**, and the **`AZURE_OPENAI_*`** env vars (see [Azure OpenAI](#azure-openai) below).

### Step 3 — Run (single process, recommended first)

From **`subnet32_dataset/`** (so `configs/` loads correctly). With **`use_sample_quotas: true`**, **`--num-pairs` is ignored** for stopping (use any positive int to satisfy the CLI—e.g. `1`). **`output_layout`** defaults to **`by_label`** in `datasets.yaml` for this recipe.

**Linux / macOS**

```bash
cd subnet32_dataset
chmod +x scripts/run_mdok_3m_quota.sh
# .env loads Azure keys; dual Ollama: ./scripts/run_2x5090_throughput.sh start-ollama
./scripts/run_mdok_3m_quota.sh
# Resume after interrupt:
#   OUT_DIR=./out_mdok_3m_subnet32 ./scripts/run_mdok_3m_quota.sh --append
```

Or explicitly:

```bash
cd subnet32_dataset
export OPENAI_API_KEY="your-key-here"   # or Azure: see openai.use_azure + AZURE_OPENAI_* env vars
python -m src.dataset_builder \
  --num-pairs 1 \
  --output-dir ./out_mdok_3m_subnet32 \
  --output-layout by_label
```

**Windows (PowerShell)**

```powershell
cd subnet32_dataset
$env:OPENAI_API_KEY = "your-key-here"   # or Azure vars; omit if not using API models
python -m src.dataset_builder `
  --num-pairs 1 `
  --output-dir .\out_mdok_3m_subnet32 `
  --output-layout by_label
```

Outputs: **`out_mdok_3m_subnet32/human/{train,val,test,stress}.jsonl`**, **`ai/…`**, **`mixed/…`**, plus **`build_stats.json`** and **`sample_counts.json`**.

### Step 4 — Resume after interruption

```bash
python -m src.dataset_builder --num-pairs 1 --output-dir ./out_mdok_3m_subnet32 --output-layout by_label --append
```

### Optional — Two processes (2× data + 2× Ollama hosts)

Prefer **CLI overrides** (no YAML edits) plus **`scripts/run_2x5090_throughput.sh`** on Linux:

```bash
chmod +x scripts/run_2x5090_throughput.sh
sudo systemctl stop ollama   # if the default single-node service holds port 11434
./scripts/run_2x5090_throughput.sh start-ollama
# Models live under ~/.ollama once; each server loads into its own GPU when you generate.
./scripts/run_2x5090_throughput.sh run-dual-builders   # NUM_PAIRS, OUT_BASE, OUTPUT_LAYOUT env optional
```

Or run two builders manually with the **same** `random_seed` in **`datasets.yaml`**:

- Process A: `--stream-num-shards 2 --stream-shard-index 0 --output-dir ./out_shard0`  
- Process B: `--stream-num-shards 2 --stream-shard-index 1 --output-dir ./out_shard1`  

Merge JSONL / stats when both complete (same schema).

---

## How to run (smoke / dev)

From the **`subnet32_dataset`** directory (so `configs/` resolves correctly):

```bash
# Default: small smoke build (paired human + AI), writes under ./outputs
python -m src.dataset_builder --num-pairs 50

# Human-only (no Ollama / OpenAI)
python -m src.dataset_builder --num-pairs 1000 --skip-ai

# Custom output + verbose logs
python -m src.dataset_builder --num-pairs 200 --output-dir ./my_run --verbose

# Split files: human/*.jsonl, ai/*.jsonl, mixed/*.jsonl
python -m src.dataset_builder --num-pairs 100 --output-layout by_label

# Resume: append rows, skip duplicate original_text_hash, reload quota counts if applicable
python -m src.dataset_builder --num-pairs 500 --append

# Single Ollama host override (wins over models.yaml / OLLAMA_BASE_URLS)
python -m src.dataset_builder --num-pairs 20 --ollama-url http://127.0.0.1:11434

# Multiple Ollama hosts (comma list; round-robin). Same effect as env OLLAMA_BASE_URLS.
python -m src.dataset_builder --num-pairs 20 --ollama-urls http://127.0.0.1:11434,http://127.0.0.1:11435
```

If you run from the **parent** repo, set the config path explicitly:

```bash
python -m src.dataset_builder --config-dir path/to/subnet32_dataset/configs --output-dir path/to/out
```

### Quota mode

See the **Production run (~3M for mdok / Subnet32)** section above. In short: set **`use_sample_quotas: true`**, keep **`sample_targets`**, run with **`--output-layout by_label`**; completion is when all buckets are full (or **`max_documents_scan`** stops the run).

### OpenAI (platform API)

- Set **`OPENAI_API_KEY`** (or the env name in `openai.api_key_env`).  
- Optional: `openai.omit_if_no_api_key: true` drops OpenAI rows when the key is missing (Ollama-only dev runs).

### `.env` file

1. Copy **`env.example`** to **`.env`** in the project root (`subnet32_dataset/`).  
2. Fill in secrets (never commit **`.env`**).  
3. Install dependencies so **`python-dotenv`** is available; **`python -m src.dataset_builder`** loads **`.env`** on startup (existing OS env vars win).

### Azure OpenAI

Use the same **`provider: openai`** entries in **`configs/models.yaml`**, but enable Azure on the **`openai:`** block:

```yaml
openai:
  use_azure: true
  api_key_env: AZURE_OPENAI_API_KEY
```

Set environment variables (PowerShell examples):

```powershell
$env:AZURE_OPENAI_API_KEY = "..."           # never commit real keys
$env:AZURE_OPENAI_ENDPOINT = "https://YOUR_RESOURCE.cognitiveservices.azure.com/"
$env:AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-mini"
$env:AZURE_OPENAI_API_VERSION = "2024-12-01-preview"   # optional; same default in code if unset
```

Each **`model:`** under **`train_generators` / `stress_generators`** should match the **deployment name** you call (Azure selects the model from the deployment URL; the YAML `model` value is still used for JSONL metadata). If you only have one deployment, point every OpenAI row at that name.

You can instead set **`openai.azure_endpoint`**, **`openai.azure_deployment`**, and **`openai.azure_api_version`** in YAML (prefer env vars for secrets).

**Security:** if an API key was pasted into chat or committed to git, **rotate it** in Azure and use the new value only in your local environment.

### Ollama multi-GPU

1. Run one **`ollama serve`** per GPU on **different ports** (e.g. `OLLAMA_HOST=127.0.0.1:11434` / `11435`, with **`CUDA_VISIBLE_DEVICES=0`** / **`1`**).  
2. List both URLs in **`configs/models.yaml`** (`ollama_base_urls`) **or** export **`OLLAMA_BASE_URLS=http://127.0.0.1:11434,http://127.0.0.1:11435`** **or** pass **`--ollama-urls ...`**.  
3. Run one **`dataset_builder`** to round-robin across hosts, **or** two workers with **`--stream-num-shards` / `--stream-shard-index`**—see **`scripts/run_2x5090_throughput.sh`** (Linux) and **`scripts/run_2x5090_throughput.ps1`** (Windows notes).

**404 “model not found” on dual servers:** if you start **`ollama serve` as root** while models were pulled under the **`ollama`** user (typical Linux install), point every server at the same blob dir: **`export OLLAMA_MODELS=/usr/share/ollama/.ollama/models`** before **`ollama serve`**. The **`start-ollama`** subcommand in **`scripts/run_2x5090_throughput.sh`** sets this automatically when that directory has manifests.

### Faster generation (Ollama model swapping)

Each **`pick_generator_entry`** call may choose a **different** Ollama tag. Ollama then **loads/unloads weights** often, which is slow on large models—this is usually the main reason throughput looks low.

Mitigations (pick one or combine):

1. **Fewer Ollama tags in `models.yaml`** — e.g. one **`provider: ollama`** row for the whole run (largest practical win while staying in one process + one quota).
2. **`--ai-phase openai_only`** — every AI completion uses **only OpenAI** rows from the YAML pools; **Ollama stays idle** for that run (good for a “GPT-only” slice of the quota in one pass).
3. **`--ai-phase ollama_locked --ollama-phase-model qwen3:14b`** — every Ollama call uses **only that tag**; **OpenAI rows are ignored** for that run (good for a “single Ollama model” pass with minimal VRAM churn).

Example:

```bash
python -m src.dataset_builder --num-pairs 1 --output-dir ./out --output-layout by_label \
  --ai-phase ollama_locked --ollama-phase-model mistral-small3.2:24b
```

**One process per `--output-dir`:** never run two **`dataset_builder`** jobs (e.g. `--ai-phase openai_only` and `--ai-phase ollama_locked` at the same time) on the **same** output folder. They overwrite **`sample_counts.json`**, can **duplicate IDs**, and **interleave** JSONL—counts look stuck and totals look wrong. The CLI now takes a **POSIX lock** on each output directory.

**Quota caveat:** one **`dataset_builder`** process with **`use_sample_quotas: true`** fills **all** `sample_targets` buckets in one pass. A phase flag changes **which backends are allowed for that entire pass**, not “fill 30% with GPT then stop”. To approximate **sequential** multi-model mixes under one global recipe, you either shrink targets per pass and merge outputs (manual), or keep **one** Ollama model in YAML for a single long run.

---

## Configuration (`configs/`)

| File | Role |
|------|------|
| `datasets.yaml` | Pile source, domain quotas, splits, **quota** / sharding / dedup / `output_layout` |
| `generation.yaml` | Length limits, decoding ranges, continuation + hard-rewrite + mixed settings |
| `augmentation.yaml` | Human/AI corruption probabilities |
| `models.yaml` | **train_generators** / **stress_generators** (Ollama + OpenAI), optional **hard_transform_generators**, **`ollama_base_urls`** |

---

## Outputs

- **`{train,val,test,stress}.jsonl`** when `output_layout: mixed`, **or**  
  **`human/`**, **`ai/`**, **`mixed/`** subtrees when `output_layout: by_label`.  
- Each line is a JSON object: `id`, `label`, **`label_int`**, **`sample_kind`**, `final_text`, `base_text`, `original_text_hash`, `split`, domain fields, generator fields for AI, etc.  
- **`build_stats.json`** — counts, layout, optional bucket totals, summaries.  
- Quota runs also write **`sample_counts.json`** (periodic + final).

---

## CLI reference

| Option | Description |
|--------|-------------|
| `--config-dir` | YAML directory (default: `subnet32_dataset/configs`) |
| `--output-dir` | Output directory (default: `subnet32_dataset/outputs`) |
| `--num-pairs` | Legacy / non-quota: target **pairs** (human+AI) or **human rows** with `--skip-ai` |
| `--stress-fraction` | Fraction routed to stress split |
| `--skip-ai` | Only human rows (no LLM) |
| `--output-layout` | `mixed` or `by_label` (overrides `datasets.yaml` if set) |
| `--ollama-url` | Single Ollama base URL override |
| `--ollama-urls` | Comma-separated Ollama URLs (round-robin); overrides YAML / `OLLAMA_BASE_URLS` |
| `--stream-num-shards` | With `--stream-shard-index`, override Pile sharding for parallel workers |
| `--stream-shard-index` | Worker index in `[0, stream_num_shards)` |
| `--ai-phase` | `all` (default), `openai_only`, or `ollama_locked` (see “Faster generation”) |
| `--ollama-phase-model` | Required with `ollama_locked` — exact Ollama model name |
| `--append` | Resume without truncating JSONL; dedupe by hash |
| `--verbose` | Debug logging |

---

## License / data

The **Pile** is used via Hugging Face (`monology/pile-uncopyrighted` by default). Respect the **dataset license** and your **OpenAI / Ollama** terms when generating at scale.
