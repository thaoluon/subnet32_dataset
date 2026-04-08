from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from .ai_generator import (
    OllamaCompletionClient,
    combine_prefix_completion,
    is_text_model_name,
    sample_generation_params,
    strip_leading_boilerplate,
    trim_to_sentence_boundary,
)
from .augmentor import Subnet32Augmentor
from .metadata_writer import JsonlWriter
from .pile_loader import PileLoader
from .prompt_builder import build_prefix_span
from .quality_filter import passes_quality
from .sentence_splitter import is_garbage_document, split_sentences
from .span_sampler import join_span, sample_consecutive_span
from .text_cleaner import clean_text
from .utils import load_yaml, set_global_seed, text_sha256, word_count, write_json
from .validator_matcher import summarize_jsonl

logger = logging.getLogger(__name__)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


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


def _pick_split(ratios: dict[str, float], stress_fraction: float) -> tuple[str, bool]:
    if random.random() < stress_fraction:
        return "stress", True
    t = float(ratios.get("train", 0.8))
    v = float(ratios.get("val", 0.1))
    te = float(ratios.get("test", 0.1))
    s = t + v + te or 1.0
    t, v, te = t / s, v / s, te / s
    r = random.random()
    if r < t:
        return "train", False
    if r < t + v:
        return "val", False
    return "test", False


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
) -> None:
    ds_cfg = load_yaml(config_dir / "datasets.yaml")
    gen_cfg = load_yaml(config_dir / "generation.yaml")
    aug_cfg = load_yaml(config_dir / "augmentation.yaml")
    model_cfg = load_yaml(config_dir / "models.yaml")

    seed = int(ds_cfg.get("random_seed", 42))
    set_global_seed(seed)

    base_url = (ollama_url or model_cfg.get("ollama_base_url", "http://127.0.0.1:11434")).rstrip("/")
    train_models = list(model_cfg.get("train_models") or [])
    stress_models = list(model_cfg.get("stress_models") or [])
    default_model = model_cfg.get("default_model")
    if not train_models and default_model:
        train_models = [str(default_model)]
    if not train_models and not skip_ai:
        raise ValueError("configs/models.yaml: define train_models or default_model")

    quotas = dict(ds_cfg.get("domain_quotas") or {})
    balancer = DomainBalancer(quotas)
    loader = PileLoader(ds_cfg)
    ollama_client = OllamaCompletionClient(base_url, gen_cfg) if not skip_ai else None

    span_min = int(gen_cfg.get("sentence_span_min", 4))
    span_max = int(gen_cfg.get("sentence_span_max", 8))
    pmin = int(gen_cfg.get("prefix_sentence_min", 2))
    pmax = int(gen_cfg.get("prefix_sentence_max", 4))
    sys_msg = str(gen_cfg.get("continuation_system", ""))
    use_raw = bool(gen_cfg.get("use_raw_completion_for_text_models", True))

    qdict = _quality_dict(gen_cfg)
    split_ratios = dict(ds_cfg.get("split_ratios") or {"train": 0.8, "val": 0.1, "test": 0.1})

    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, JsonlWriter] = {}
    counters: dict[str, int] = defaultdict(int)

    def writer_for(split: str) -> JsonlWriter:
        if split not in writers:
            writers[split] = JsonlWriter(output_dir / f"{split}.jsonl")
            writers[split].__enter__()
        return writers[split]

    def next_id(split: str) -> str:
        counters[split] += 1
        return f"s32_{split}_{counters[split]:06d}"

    pairs_done = 0
    docs_seen = 0

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

        split, stress_mode = _pick_split(split_ratios, stress_fraction)
        augmentor = Subnet32Augmentor(aug_cfg, stress=stress_mode)

        span, i0, i1 = sample_consecutive_span(sents, span_min, span_max)
        human_base = join_span(span)
        human_final, human_aug_meta = augmentor.augment(human_base)
        ok_h, reason_h = passes_quality(human_final, qdict, is_ai=False)
        if not ok_h:
            logger.debug("skip human: %s", reason_h)
            continue

        split_writer = writer_for(split)
        hid = next_id(split)
        human_record = {
            "id": hid,
            "label": "human",
            "domain": doc.coarse_domain,
            "source_doc_id": doc.doc_id,
            "source_type": "pile",
            "pile_subset": doc.pile_subset,
            "prompt_text": None,
            "base_text": human_base,
            "final_text": human_final,
            "generator_model": None,
            "generation_params": None,
            "augmentation": human_aug_meta,
            "num_sentences": len(split_sentences(human_final)),
            "num_words": word_count(human_final),
            "split": split,
            "span_sentence_indices": [i0, i1],
            "original_text_hash": text_sha256(human_base),
            "stress_mode": stress_mode,
        }

        if skip_ai:
            split_writer.write(human_record)
            balancer.record(doc.coarse_domain)
            pairs_done += 1
            continue

        assert ollama_client is not None
        model_pool = stress_models if stress_mode and stress_models else train_models
        model_name = random.choice(model_pool)
        raw_completion = ""
        gen_params_meta: dict[str, Any] = {}
        prefix, _pk = build_prefix_span(sents, pmin, pmax)
        if not prefix.strip():
            logger.debug("skip ai: empty prefix")
            continue

        try:
            params = sample_generation_params(gen_cfg)
            raw_mode = use_raw and is_text_model_name(model_name)
            raw_completion, gen_params_meta = ollama_client.generate(
                model_name,
                prefix,
                params=params,
                raw_completion=raw_mode,
                system=sys_msg if not raw_mode else None,
            )
        except Exception as e:
            logger.warning("Ollama failed for doc %s: %s", doc.doc_id, e)
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

        split_writer.write(human_record)
        balancer.record(doc.coarse_domain)

        aid = next_id(split)
        ai_record = {
            "id": aid,
            "label": "ai",
            "domain": doc.coarse_domain,
            "source_doc_id": doc.doc_id,
            "source_type": "pile",
            "pile_subset": doc.pile_subset,
            "prompt_text": prefix,
            "base_text": ai_base,
            "final_text": ai_final,
            "generator_model": model_name,
            "generation_params": gen_params_meta,
            "augmentation": ai_aug_meta,
            "num_sentences": len(split_sentences(ai_final)),
            "num_words": word_count(ai_final),
            "split": split,
            "span_sentence_indices": None,
            "original_text_hash": text_sha256(ai_base),
            "stress_mode": stress_mode,
        }
        split_writer.write(ai_record)
        balancer.record(doc.coarse_domain)
        pairs_done += 1

        if pairs_done % 20 == 0:
            logger.info("pairs=%s docs_seen=%s", pairs_done, docs_seen)

    for w in writers.values():
        w.__exit__(None, None, None)

    build_stats: dict[str, Any] = {
        "num_pairs_target": num_pairs,
        "pairs_written": pairs_done,
        "documents_seen": docs_seen,
        "outputs": {},
    }
    for name in ("train", "val", "test", "stress"):
        p = output_dir / f"{name}.jsonl"
        if p.is_file() and p.stat().st_size > 0:
            build_stats["outputs"][name] = summarize_jsonl(p)

    write_json(output_dir / "build_stats.json", build_stats)
    logger.info("Finished. pairs_written=%s stats=%s", pairs_done, output_dir / "build_stats.json")


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
@click.option("--num-pairs", type=int, default=50, help="Number of human+ai pairs to write")
@click.option("--stress-fraction", type=float, default=0.05, help="Fraction assigned to stress split")
@click.option("--skip-ai", is_flag=True, help="Only write human samples (no Ollama)")
@click.option("--ollama-url", type=str, default=None, help="Override Ollama base URL")
@click.option("--verbose", is_flag=True)
def main(
    config_dir: Path | None,
    output_dir: Path | None,
    num_pairs: int,
    stress_fraction: float,
    skip_ai: bool,
    ollama_url: str | None,
    verbose: bool,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg_dir = config_dir or (PACKAGE_ROOT / "configs")
    out = output_dir or (PACKAGE_ROOT / "outputs")
    run_build(cfg_dir, out, num_pairs, stress_fraction, skip_ai, ollama_url)


if __name__ == "__main__":
    main()
