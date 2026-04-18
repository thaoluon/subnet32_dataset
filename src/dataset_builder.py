from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from .ai_generator import (
    OllamaCompletionClient,
    OllamaUnreachableError,
    combine_prefix_completion,
    is_text_model_name,
    sample_generation_params,
    strip_leading_boilerplate,
    trim_to_sentence_boundary,
)
from .generator_pools import (
    GeneratorEntry,
    assert_generators_configured,
    pick_generator_entry,
    providers_used,
    resolve_stress_pool,
    resolve_train_pool,
    strip_openai_if_key_missing,
)
from .llm_rewrite import llm_paraphrase
from .openai_generator import OpenAIChatCompletionClient, OpenAINotConfiguredError
from .sample_quotas import (
    BUCKET_KEYS,
    load_counts_json,
    parse_sample_targets,
    pick_bucket,
    quotas_complete,
    resolve_hard_transform_pool,
    save_counts_json,
)
from .augmentor import Subnet32Augmentor
from .doc_split_registry import DocSplitRegistry
from .metadata_writer import JsonlWriter
from .pile_loader import PileLoader
from .prompt_builder import build_prefix_span
from .quality_filter import passes_quality
from .sentence_splitter import is_garbage_document, split_sentences
from .span_sampler import join_span, sample_consecutive_span
from .text_cleaner import clean_text
from .utils import load_yaml, normalized_text_hash, set_global_seed, text_sha256, word_count, write_json
from .validator_matcher import summarize_jsonl

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_ID_RE = re.compile(r"^s32_([a-z]+)_(\d+)$")


def _jsonl_paths_for_layout(output_dir: Path, layout: str) -> list[Path]:
    """All JSONL paths used for a given layout (for append state + truncation)."""
    if layout == "by_label":
        paths: list[Path] = []
        for lab in ("human", "ai", "mixed"):
            for split in ("train", "val", "test", "stress"):
                paths.append(output_dir / lab / f"{split}.jsonl")
        return paths
    return [output_dir / f"{split}.jsonl" for split in ("train", "val", "test", "stress")]


def _load_jsonl_output_state(output_dir: Path, layout: str) -> tuple[set[str], dict[str, int]]:
    """Hashes already present and highest numeric id per split (for append mode)."""
    seen_hashes: set[str] = set()
    max_id_by_split: dict[str, int] = {}
    for path in _jsonl_paths_for_layout(output_dir, layout):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            h = rec.get("original_text_hash")
            if isinstance(h, str):
                seen_hashes.add(h)
            rid = rec.get("id")
            if isinstance(rid, str):
                m = _ID_RE.match(rid)
                if m:
                    split = m.group(1)
                    n = int(m.group(2))
                    max_id_by_split[split] = max(max_id_by_split.get(split, 0), n)
    return seen_hashes, max_id_by_split


def _truncate_split_jsonls(output_dir: Path, layout: str) -> None:
    if layout == "by_label":
        for lab in ("human", "ai", "mixed"):
            (output_dir / lab).mkdir(parents=True, exist_ok=True)
            for split in ("train", "val", "test", "stress"):
                p = output_dir / lab / f"{split}.jsonl"
                if p.is_file():
                    p.write_text("", encoding="utf-8")
        return
    for split in ("train", "val", "test", "stress"):
        p = output_dir / f"{split}.jsonl"
        if p.is_file():
            p.write_text("", encoding="utf-8")


def _load_normalized_hashes_from_disk(output_dir: Path, layout: str) -> set[str]:
    """Rebuild normalized-text fingerprints from existing JSONL (append / dedup resume)."""
    out: set[str] = set()
    for path in _jsonl_paths_for_layout(output_dir, layout):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            b = rec.get("base_text")
            if isinstance(b, str) and b.strip():
                out.add(normalized_text_hash(b))
    return out


class DomainBalancer:
    """Prefer underrepresented coarse domains according to quotas."""

    def __init__(self, quotas: dict[str, float]):
        raw = {str(k): float(v) for k, v in (quotas or {}).items()}
        s = sum(raw.values()) or 1.0
        self.quotas = {k: v / s for k, v in raw.items()}
        self.counts: Counter[str] = Counter()

    def accept(self, domain: str) -> bool:
        if not self.quotas:
            return True
        d = domain if domain in self.quotas else min(self.quotas.keys(), key=lambda k: self.counts[k])
        total = sum(self.counts.values())
        if total < 80:
            return True
        cur = self.counts[d]
        share = cur / total
        target = self.quotas.get(d, 1.0 / len(self.quotas))
        if share < target * 1.12:
            return True
        return random.random() < 0.22

    def record(self, domain: str) -> None:
        if not self.quotas:
            self.counts[domain] += 1
            return
        d = domain if domain in self.quotas else min(self.quotas.keys(), key=lambda k: self.counts[k])
        self.counts[d] += 1


def _quality_dict(gen_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "word_count_min": int(gen_cfg.get("word_count_min", 80)),
        "word_count_max": int(gen_cfg.get("word_count_max", 300)),
        "min_sentences": 3,
        "max_non_alnum_ratio": 0.18,
        "max_line_repetition": 0.35,
    }


def run_build(
    config_dir: Path,
    output_dir: Path,
    num_pairs: int,
    stress_fraction: float,
    skip_ai: bool,
    ollama_url: str | None,
    append: bool,
    output_layout: str | None = None,
) -> None:
    ds_cfg = load_yaml(config_dir / "datasets.yaml")
    targets = parse_sample_targets(ds_cfg)
    quota_mode = targets is not None
    if quota_mode and skip_ai:
        raise ValueError(
            "datasets.yaml has use_sample_quotas: true — AI buckets need LLM generation. "
            "Omit --skip-ai or set use_sample_quotas: false."
        )

    gen_cfg = load_yaml(config_dir / "generation.yaml")
    aug_cfg = load_yaml(config_dir / "augmentation.yaml")
    model_cfg = load_yaml(config_dir / "models.yaml")

    seed = int(ds_cfg.get("random_seed", 42))
    set_global_seed(seed)

    base_url = (ollama_url or model_cfg.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
    train_pool = resolve_train_pool(model_cfg)
    stress_pool = resolve_stress_pool(model_cfg)
    train_pool, stress_pool = strip_openai_if_key_missing(model_cfg, train_pool, stress_pool)
    if not skip_ai and not train_pool:
        raise ValueError(
            "configs/models.yaml: no train generators (train_generators or train_models / default_model). "
            "If you used openai.omit_if_no_api_key, set OPENAI_API_KEY or add non-openai generators."
        )

    quotas = dict(ds_cfg.get("domain_quotas") or {})
    balancer = DomainBalancer(quotas)
    loader = PileLoader(ds_cfg)

    used_providers = providers_used([train_pool, stress_pool]) if not skip_ai else set()
    ollama_client = (
        OllamaCompletionClient(base_url, gen_cfg) if (not skip_ai and "ollama" in used_providers) else None
    )
    openai_cfg = dict(model_cfg.get("openai") or {})
    openai_client = (
        OpenAIChatCompletionClient(gen_cfg, openai_cfg)
        if (not skip_ai and "openai" in used_providers)
        else None
    )
    if not skip_ai:
        assert_generators_configured(
            skip_ai=False,
            ollama_base_url=base_url,
            model_cfg=model_cfg,
            gen_cfg=gen_cfg,
            train_pool=train_pool,
            stress_pool=stress_pool,
        )

    hard_transform_pool: list[GeneratorEntry] = []
    if quota_mode and not skip_ai:
        hard_transform_pool = resolve_hard_transform_pool(model_cfg)
        if not hard_transform_pool:
            raise ValueError(
                "Quota mode requires at least one LLM for hard_ai paraphrase. "
                "Configure train_generators or hard_transform_generators in models.yaml."
            )

    span_min = int(gen_cfg.get("sentence_span_min", 4))
    span_max = int(gen_cfg.get("sentence_span_max", 8))
    pmin = int(gen_cfg.get("prefix_sentence_min", 2))
    pmax = int(gen_cfg.get("prefix_sentence_max", 4))
    sys_msg = str(gen_cfg.get("continuation_system", ""))
    use_raw = bool(gen_cfg.get("use_raw_completion_for_text_models", True))

    qdict = _quality_dict(gen_cfg)
    split_ratios = dict(ds_cfg.get("split_ratios") or {"train": 0.8, "val": 0.1, "test": 0.1})
    doc_split_registry = DocSplitRegistry(split_ratios, stress_fraction)
    dedup_norm = bool(ds_cfg.get("dedup_normalize_text", True))

    layout = str(output_layout or ds_cfg.get("output_layout") or "mixed").lower()
    if layout not in ("mixed", "by_label"):
        raise ValueError(f"output_layout must be 'mixed' or 'by_label', got {layout!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if append:
        seen_hashes, max_id_by_split = _load_jsonl_output_state(output_dir, layout)
    else:
        seen_hashes = set()
        max_id_by_split = {}
        _truncate_split_jsonls(output_dir, layout)
        if quota_mode:
            scp = output_dir / "sample_counts.json"
            if scp.is_file():
                scp.unlink()

    seen_normalized: set[str] = set()
    if dedup_norm and append:
        seen_normalized = _load_normalized_hashes_from_disk(output_dir, layout)

    def is_norm_dup(base: str) -> bool:
        return dedup_norm and normalized_text_hash(base) in seen_normalized

    def record_norm(base: str) -> None:
        if dedup_norm:
            seen_normalized.add(normalized_text_hash(base))

    writers: dict[str, JsonlWriter] = {}
    counters: dict[str, int] = defaultdict(int)
    for split, n in max_id_by_split.items():
        counters[split] = n

    def writer_for(split: str, label_side: str | None = None) -> JsonlWriter:
        if layout == "by_label":
            assert label_side in ("human", "ai", "mixed")
            key = f"{label_side}/{split}"
            path = output_dir / label_side / f"{split}.jsonl"
        else:
            key = split
            path = output_dir / f"{split}.jsonl"
        if key not in writers:
            writers[key] = JsonlWriter(path)
            writers[key].__enter__()
        return writers[key]

    def next_id(split: str) -> str:
        counters[split] += 1
        return f"s32_{split}_{counters[split]:06d}"

    bucket_counts: dict[str, int] = {k: 0 for k in BUCKET_KEYS}
    if quota_mode and append:
        loaded = load_counts_json(output_dir / "sample_counts.json")
        if loaded:
            bucket_counts.update({k: int(loaded[k]) for k in BUCKET_KEYS})

    def call_generate(gen_entry: GeneratorEntry, prompt: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if gen_entry.provider == "ollama":
            if ollama_client is None:
                raise RuntimeError("ollama client missing")
            raw_mode = use_raw and is_text_model_name(gen_entry.model)
            return ollama_client.generate(
                gen_entry.model,
                prompt,
                params=params,
                raw_completion=raw_mode,
                system=sys_msg if not raw_mode else None,
            )
        if gen_entry.provider == "openai":
            if openai_client is None:
                raise RuntimeError("openai client missing")
            return openai_client.generate(
                gen_entry.model,
                prompt,
                system=sys_msg or None,
                params=params,
            )
        raise RuntimeError(f"unknown provider {gen_entry.provider!r}")

    pairs_done = 0
    human_rows = 0
    ai_rows = 0
    mixed_rows = 0
    docs_seen = 0
    writes_since_save = 0

    if not quota_mode:
        for doc in loader.iter_documents():
            if pairs_done >= num_pairs:
                break
            docs_seen += 1
            if not balancer.accept(doc.coarse_domain):
                continue

            cleaned = clean_text(doc.text)
            sents = split_sentences(cleaned)
            if is_garbage_document(sents, min_sentences=4):
                continue

            split, stress_mode = doc_split_registry.split_for(doc.doc_id)
            augmentor = Subnet32Augmentor(aug_cfg, stress=stress_mode)

            span, i0, i1 = sample_consecutive_span(sents, span_min, span_max)
            human_base = join_span(span)
            human_final, human_aug_meta = augmentor.augment(human_base)
            ok_h, reason_h = passes_quality(human_final, qdict, is_ai=False)
            if not ok_h:
                logger.debug("skip human: %s", reason_h)
                continue

            human_hash = text_sha256(human_base)
            if human_hash in seen_hashes:
                logger.debug("skip duplicate original_text_hash (human span)")
                continue
            if is_norm_dup(human_base):
                logger.debug("skip near-duplicate normalized human base_text")
                continue

            hid = next_id(split)
            human_record = {
                "id": hid,
                "label": "human",
                "label_int": 0,
                "sample_kind": "human",
                "domain": doc.coarse_domain,
                "source_doc_id": doc.doc_id,
                "source_type": "pile",
                "pile_subset": doc.pile_subset,
                "prompt_text": None,
                "base_text": human_base,
                "final_text": human_final,
                "generator_provider": None,
                "generator_model": None,
                "generation_params": None,
                "augmentation": human_aug_meta,
                "num_sentences": len(split_sentences(human_final)),
                "num_words": word_count(human_final),
                "split": split,
                "span_sentence_indices": [i0, i1],
                "original_text_hash": human_hash,
                "stress_mode": stress_mode,
            }

            if skip_ai:
                hw = writer_for(split, "human") if layout == "by_label" else writer_for(split)
                hw.write(human_record)
                seen_hashes.add(human_hash)
                record_norm(human_base)
                balancer.record(doc.coarse_domain)
                human_rows += 1
                pairs_done += 1
                continue

            assert ollama_client is not None or openai_client is not None
            pool = stress_pool if (stress_mode and stress_pool) else train_pool
            gen_entry: GeneratorEntry = pick_generator_entry(pool)
            prefix, _pk = build_prefix_span(sents, pmin, pmax)
            if not prefix.strip():
                logger.debug("skip ai: empty prefix")
                continue

            try:
                params = sample_generation_params(gen_cfg)
                raw_completion, gen_params_meta = call_generate(gen_entry, prefix, params)
            except Exception as e:
                logger.warning(
                    "AI generation failed (%s model=%s) doc %s: %s",
                    gen_entry.provider,
                    gen_entry.model,
                    doc.doc_id,
                    e,
                )
                continue

            completion = strip_leading_boilerplate(trim_to_sentence_boundary(raw_completion))
            ai_base = combine_prefix_completion(prefix, completion)
            if word_count(ai_base) < qdict["word_count_min"]:
                logger.debug("skip ai: combined too short")
                continue

            ai_final, ai_aug_meta = augmentor.augment(ai_base)
            ok_a, reason_a = passes_quality(
                ai_final,
                qdict,
                is_ai=True,
                prefix=prefix,
            )
            if not ok_a:
                logger.debug("skip ai: %s", reason_a)
                continue

            ai_hash = text_sha256(ai_base)
            if ai_hash in seen_hashes:
                logger.debug("skip duplicate original_text_hash (ai sample)")
                continue
            if is_norm_dup(ai_base):
                logger.debug("skip near-duplicate normalized ai base_text")
                continue

            if layout == "by_label":
                writer_for(split, "human").write(human_record)
            else:
                writer_for(split).write(human_record)
            seen_hashes.add(human_hash)
            record_norm(human_base)
            balancer.record(doc.coarse_domain)
            human_rows += 1

            aid = next_id(split)
            ai_record = {
                "id": aid,
                "label": "ai",
                "label_int": 1,
                "sample_kind": "direct_ai",
                "domain": doc.coarse_domain,
                "source_doc_id": doc.doc_id,
                "source_type": "pile",
                "pile_subset": doc.pile_subset,
                "prompt_text": prefix,
                "base_text": ai_base,
                "final_text": ai_final,
                "generator_provider": gen_entry.provider,
                "generator_model": gen_entry.record_model_id(),
                "generation_params": gen_params_meta,
                "augmentation": ai_aug_meta,
                "num_sentences": len(split_sentences(ai_final)),
                "num_words": word_count(ai_final),
                "split": split,
                "span_sentence_indices": None,
                "original_text_hash": ai_hash,
                "stress_mode": stress_mode,
            }
            if layout == "by_label":
                writer_for(split, "ai").write(ai_record)
            else:
                writer_for(split).write(ai_record)
            seen_hashes.add(ai_hash)
            record_norm(ai_base)
            balancer.record(doc.coarse_domain)
            ai_rows += 1
            pairs_done += 1

            if pairs_done % 20 == 0:
                logger.info(
                    "pairs=%s human_rows=%s ai_rows=%s docs_seen=%s",
                    pairs_done,
                    human_rows,
                    ai_rows,
                    docs_seen,
                )

    else:
        assert targets is not None
        mixed_join = str(gen_cfg.get("mixed_paragraph_joiner", "\n\n"))
        mixed_qdict = {
            **qdict,
            "word_count_max": int(gen_cfg.get("mixed_word_count_max", 520)),
        }
        sync_every = max(1, int(ds_cfg.get("sample_counts_sync_every", 100)))
        max_docs_scan = ds_cfg.get("max_documents_scan")

        for doc in loader.iter_documents():
            if quotas_complete(bucket_counts, targets):
                break
            docs_seen += 1
            if max_docs_scan is not None and docs_seen > int(max_docs_scan):
                logger.warning("Stopped: max_documents_scan=%s reached", max_docs_scan)
                break
            if not balancer.accept(doc.coarse_domain):
                continue

            cleaned = clean_text(doc.text)
            sents = split_sentences(cleaned)
            if is_garbage_document(sents, min_sentences=4):
                continue

            split, stress_mode = doc_split_registry.split_for(doc.doc_id)
            augmentor = Subnet32Augmentor(aug_cfg, stress=stress_mode)
            bucket = pick_bucket(bucket_counts, targets)
            if bucket is None:
                break

            if bucket == "human":
                span, i0, i1 = sample_consecutive_span(sents, span_min, span_max)
                human_base = join_span(span)
                human_final, human_aug_meta = augmentor.augment(human_base)
                ok_h, reason_h = passes_quality(human_final, qdict, is_ai=False)
                if not ok_h:
                    logger.debug("quota skip human: %s", reason_h)
                    continue
                human_hash = text_sha256(human_base)
                if human_hash in seen_hashes:
                    continue
                if is_norm_dup(human_base):
                    continue
                hid = next_id(split)
                human_record = {
                    "id": hid,
                    "label": "human",
                    "label_int": 0,
                    "sample_kind": "human",
                    "domain": doc.coarse_domain,
                    "source_doc_id": doc.doc_id,
                    "source_type": "pile",
                    "pile_subset": doc.pile_subset,
                    "prompt_text": None,
                    "base_text": human_base,
                    "final_text": human_final,
                    "generator_provider": None,
                    "generator_model": None,
                    "generation_params": None,
                    "augmentation": human_aug_meta,
                    "num_sentences": len(split_sentences(human_final)),
                    "num_words": word_count(human_final),
                    "split": split,
                    "span_sentence_indices": [i0, i1],
                    "original_text_hash": human_hash,
                    "stress_mode": stress_mode,
                }
                w = writer_for(split, "human") if layout == "by_label" else writer_for(split)
                w.write(human_record)
                seen_hashes.add(human_hash)
                record_norm(human_base)
                balancer.record(doc.coarse_domain)
                bucket_counts["human"] += 1
                human_rows += 1
                writes_since_save += 1
                if writes_since_save >= sync_every:
                    save_counts_json(output_dir / "sample_counts.json", bucket_counts, targets)
                    writes_since_save = 0
                if human_rows % 50 == 0:
                    logger.info("quota %s docs_seen=%s", bucket_counts, docs_seen)
                continue

            assert ollama_client is not None or openai_client is not None
            pool = stress_pool if (stress_mode and stress_pool) else train_pool
            prefix, _pk = build_prefix_span(sents, pmin, pmax)
            if not prefix.strip():
                continue
            gen_entry = pick_generator_entry(pool)
            try:
                params = sample_generation_params(gen_cfg)
                raw_completion, gen_params_meta = call_generate(gen_entry, prefix, params)
            except Exception as e:
                logger.warning(
                    "quota AI gen failed (%s) doc %s: %s",
                    gen_entry.model,
                    doc.doc_id,
                    e,
                )
                continue

            completion = strip_leading_boilerplate(trim_to_sentence_boundary(raw_completion))
            ai_base = combine_prefix_completion(prefix, completion)
            if word_count(ai_base) < qdict["word_count_min"]:
                continue

            if bucket == "direct_ai":
                ai_final, ai_aug_meta = augmentor.augment(ai_base)
                ok_a, reason_a = passes_quality(ai_final, qdict, is_ai=True, prefix=prefix)
                if not ok_a:
                    logger.debug("quota skip direct_ai: %s", reason_a)
                    continue
                ai_hash = text_sha256(ai_base)
                if ai_hash in seen_hashes:
                    continue
                if is_norm_dup(ai_base):
                    continue
                aid = next_id(split)
                ai_record = {
                    "id": aid,
                    "label": "ai",
                    "label_int": 1,
                    "sample_kind": "direct_ai",
                    "domain": doc.coarse_domain,
                    "source_doc_id": doc.doc_id,
                    "source_type": "pile",
                    "pile_subset": doc.pile_subset,
                    "prompt_text": prefix,
                    "base_text": ai_base,
                    "final_text": ai_final,
                    "generator_provider": gen_entry.provider,
                    "generator_model": gen_entry.record_model_id(),
                    "generation_params": gen_params_meta,
                    "augmentation": ai_aug_meta,
                    "num_sentences": len(split_sentences(ai_final)),
                    "num_words": word_count(ai_final),
                    "split": split,
                    "span_sentence_indices": None,
                    "original_text_hash": ai_hash,
                    "stress_mode": stress_mode,
                }
                w = writer_for(split, "ai") if layout == "by_label" else writer_for(split)
                w.write(ai_record)
                seen_hashes.add(ai_hash)
                record_norm(ai_base)
                balancer.record(doc.coarse_domain)
                bucket_counts["direct_ai"] += 1
                ai_rows += 1
                writes_since_save += 1
                if writes_since_save >= sync_every:
                    save_counts_json(output_dir / "sample_counts.json", bucket_counts, targets)
                    writes_since_save = 0
                continue

            if bucket == "hard_ai":
                gen_hard = pick_generator_entry(hard_transform_pool)
                try:
                    para, rw_meta = llm_paraphrase(
                        ai_base,
                        gen_entry=gen_hard,
                        system=None,
                        ollama_client=ollama_client,
                        openai_client=openai_client,
                        gen_cfg=gen_cfg,
                        use_raw=use_raw,
                    )
                except Exception as e:
                    logger.warning("hard_ai paraphrase failed doc %s: %s", doc.doc_id, e)
                    continue
                hard_base = para.strip()
                if word_count(hard_base) < qdict["word_count_min"]:
                    continue
                ai_final, ai_aug_meta = augmentor.augment(hard_base)
                ok_a, reason_a = passes_quality(
                    ai_final,
                    qdict,
                    is_ai=True,
                    prefix=prefix,
                    skip_prefix_overlap_check=True,
                )
                if not ok_a:
                    logger.debug("quota skip hard_ai: %s", reason_a)
                    continue
                ai_hash = text_sha256(hard_base)
                if ai_hash in seen_hashes:
                    continue
                if is_norm_dup(hard_base):
                    continue
                aid = next_id(split)
                ai_record = {
                    "id": aid,
                    "label": "ai",
                    "label_int": 1,
                    "sample_kind": "hard_ai",
                    "domain": doc.coarse_domain,
                    "source_doc_id": doc.doc_id,
                    "source_type": "pile",
                    "pile_subset": doc.pile_subset,
                    "prompt_text": prefix,
                    "base_text": hard_base,
                    "final_text": ai_final,
                    "generator_provider": gen_hard.provider,
                    "generator_model": gen_hard.record_model_id(),
                    "continuation_generator": gen_entry.record_model_id(),
                    "continuation_provider": gen_entry.provider,
                    "generation_params": {"continuation": gen_params_meta, "rewrite": rw_meta},
                    "augmentation": ai_aug_meta,
                    "num_sentences": len(split_sentences(ai_final)),
                    "num_words": word_count(ai_final),
                    "split": split,
                    "span_sentence_indices": None,
                    "original_text_hash": ai_hash,
                    "stress_mode": stress_mode,
                }
                w = writer_for(split, "ai") if layout == "by_label" else writer_for(split)
                w.write(ai_record)
                seen_hashes.add(ai_hash)
                record_norm(hard_base)
                balancer.record(doc.coarse_domain)
                bucket_counts["hard_ai"] += 1
                ai_rows += 1
                writes_since_save += 1
                if writes_since_save >= sync_every:
                    save_counts_json(output_dir / "sample_counts.json", bucket_counts, targets)
                    writes_since_save = 0
                continue

            # mixed
            span, i0, i1 = sample_consecutive_span(sents, span_min, span_max)
            human_base = join_span(span)
            human_final, human_aug_meta = augmentor.augment(human_base)
            ok_h, reason_h = passes_quality(human_final, qdict, is_ai=False)
            if not ok_h:
                continue
            ai_final_part, ai_aug_meta = augmentor.augment(ai_base)
            ok_a, reason_a = passes_quality(ai_final_part, qdict, is_ai=True, prefix=prefix)
            if not ok_a:
                continue
            joined = human_final.strip() + mixed_join + ai_final_part.strip()
            ok_m, reason_m = passes_quality(
                joined,
                mixed_qdict,
                is_ai=False,
                skip_prefix_overlap_check=True,
            )
            if not ok_m:
                logger.debug("quota skip mixed: %s", reason_m)
                continue
            mixed_base = human_base + mixed_join + ai_base
            mixed_hash = text_sha256(mixed_base)
            if mixed_hash in seen_hashes:
                continue
            if is_norm_dup(mixed_base):
                continue
            mid = next_id(split)
            mixed_record = {
                "id": mid,
                "label": "ai",
                "label_int": 1,
                "sample_kind": "mixed",
                "domain": doc.coarse_domain,
                "source_doc_id": doc.doc_id,
                "source_type": "pile",
                "pile_subset": doc.pile_subset,
                "prompt_text": prefix,
                "base_text": mixed_base,
                "final_text": joined,
                "mixed_joiner": mixed_join,
                "generator_provider": gen_entry.provider,
                "generator_model": gen_entry.record_model_id(),
                "generation_params": gen_params_meta,
                "augmentation": human_aug_meta + ai_aug_meta,
                "num_sentences": len(split_sentences(joined)),
                "num_words": word_count(joined),
                "split": split,
                "span_sentence_indices": [i0, i1],
                "original_text_hash": mixed_hash,
                "stress_mode": stress_mode,
            }
            if layout == "by_label":
                writer_for(split, "mixed").write(mixed_record)
            else:
                writer_for(split).write(mixed_record)
            seen_hashes.add(mixed_hash)
            record_norm(mixed_base)
            balancer.record(doc.coarse_domain)
            bucket_counts["mixed"] += 1
            mixed_rows += 1
            writes_since_save += 1
            if writes_since_save >= sync_every:
                save_counts_json(output_dir / "sample_counts.json", bucket_counts, targets)
                writes_since_save = 0

        save_counts_json(output_dir / "sample_counts.json", bucket_counts, targets)

    for w in writers.values():
        w.__exit__(None, None, None)

    build_stats: dict[str, Any] = {
        "num_pairs_target": num_pairs,
        "pairs_written": pairs_done,
        "human_rows_written": human_rows,
        "ai_rows_written": ai_rows,
        "mixed_rows_written": mixed_rows,
        "use_sample_quotas": quota_mode,
        "sample_targets": targets,
        "bucket_counts_final": dict(bucket_counts) if quota_mode else None,
        "output_layout": layout,
        "documents_seen": docs_seen,
        "doc_split_documents_cached": doc_split_registry.as_dict().get("cached_documents"),
        "dedup_normalize_text": dedup_norm,
        "outputs": {},
    }
    if layout == "by_label":
        build_stats["outputs"] = {"human": {}, "ai": {}, "mixed": {}}
        for lab in ("human", "ai", "mixed"):
            for name in ("train", "val", "test", "stress"):
                p = output_dir / lab / f"{name}.jsonl"
                if p.is_file() and p.stat().st_size > 0:
                    build_stats["outputs"][lab][name] = summarize_jsonl(p)
    else:
        for name in ("train", "val", "test", "stress"):
            p = output_dir / f"{name}.jsonl"
            if p.is_file() and p.stat().st_size > 0:
                build_stats["outputs"][name] = summarize_jsonl(p)

    write_json(output_dir / "build_stats.json", build_stats)
    logger.info(
        "Finished. pairs=%s human=%s ai=%s mixed=%s quotas=%s layout=%s stats=%s",
        pairs_done,
        human_rows,
        ai_rows,
        mixed_rows,
        bucket_counts if quota_mode else None,
        layout,
        output_dir / "build_stats.json",
    )


@click.command()
@click.option(
    "--config-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory containing YAML configs (default: subnet32_dataset/configs)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write JSONL splits and build_stats.json",
)
@click.option(
    "--num-pairs",
    type=int,
    default=50,
    help="Target count: with AI, matched human+ai pairs; with --skip-ai, human rows only.",
)
@click.option("--stress-fraction", type=float, default=0.05, help="Fraction assigned to stress split")
@click.option(
    "--skip-ai",
    is_flag=True,
    help="Human dataset only: sample Pile spans and write label=human rows (no LLM calls).",
)
@click.option(
    "--output-layout",
    type=click.Choice(["mixed", "by_label"], case_sensitive=False),
    default=None,
    help="mixed: {train,val,test,stress}.jsonl with human+ai rows. "
    "by_label: human/{split}.jsonl and ai/{split}.jsonl (clear 3M+3M file layout). "
    "Default: datasets.yaml output_layout or mixed.",
)
@click.option("--ollama-url", type=str, default=None, help="Override Ollama base URL")
@click.option(
    "--append",
    is_flag=True,
    help="Keep existing split JSONLs, continue IDs, and skip rows whose original_text_hash already exists.",
)
@click.option("--verbose", is_flag=True)
def main(
    config_dir: Path | None,
    output_dir: Path | None,
    num_pairs: int,
    stress_fraction: float,
    skip_ai: bool,
    ollama_url: str | None,
    append: bool,
    output_layout: str | None,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg_dir = config_dir or (PACKAGE_ROOT / "configs")
    out = output_dir or (PACKAGE_ROOT / "outputs")
    try:
        run_build(cfg_dir, out, num_pairs, stress_fraction, skip_ai, ollama_url, append, output_layout)
    except OllamaUnreachableError as e:
        raise click.ClickException(str(e)) from e
    except OpenAINotConfiguredError as e:
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
