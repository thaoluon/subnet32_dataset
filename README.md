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

In **`configs/datasets.yaml`** set:

```yaml
use_sample_quotas: true
```

(Leave `sample_targets` as-is unless you change the recipe.) When this is `true`, **`--num-pairs` does not stop the run**—the job finishes when every bucket hits its target (or `max_documents_scan` is hit).

Optional for big RAM hosts: raise **`shuffle_buffer_size`** (e.g. `250000`) for a stronger Pile shuffle.

### Step 2 — Models and secrets

1. **`configs/models.yaml`**: confirm **Ollama** and/or **OpenAI** `train_generators` / `stress_generators`. For **2× GPU**, uncomment **`ollama_base_urls`** with two ports (see `scripts/run_2x5090_throughput.ps1`).  
2. **OpenAI or Azure**: platform API — `export OPENAI_API_KEY=...` (or PowerShell `$env:OPENAI_API_KEY`). For **Azure OpenAI**, set **`openai.use_azure: true`**, **`api_key_env: AZURE_OPENAI_API_KEY`**, and the **`AZURE_OPENAI_*`** env vars (see [Azure OpenAI](#azure-openai) below).

### Step 3 — Run (single process, recommended first)

From **`subnet32_dataset/`** (so `configs/` loads correctly). With **`use_sample_quotas: true`**, **`--num-pairs` is ignored** for stopping (use any positive int to satisfy the CLI—e.g. `1`).

**Linux / macOS**

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

Use the **same** `random_seed` and **`datasets.yaml`** with **`stream_num_shards: 2`**, then:

- Process A: `stream_shard_index: 0`, `--output-dir ./out_shard0`  
- Process B: `stream_shard_index: 1`, `--output-dir ./out_shard1`  

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

# Single Ollama host override (disables models.yaml ollama_base_urls list)
python -m src.dataset_builder --num-pairs 20 --ollama-url http://127.0.0.1:11434
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

1. Run one **`ollama serve`** per GPU on **different ports** (e.g. `OLLAMA_HOST=127.0.0.1:11434` / `11435`).  
2. Set **`ollama_base_urls`** in **`configs/models.yaml`**.  
3. Run a **single** `dataset_builder` process to round-robin across hosts, **or** use **`stream_num_shards` / `stream_shard_index`** and two processes—see **`scripts/run_2x5090_throughput.ps1`**.

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
| `--append` | Resume without truncating JSONL; dedupe by hash |
| `--verbose` | Debug logging |

---

## License / data

The **Pile** is used via Hugging Face (`monology/pile-uncopyrighted` by default). Respect the **dataset license** and your **OpenAI / Ollama** terms when generating at scale.
