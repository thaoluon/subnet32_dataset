from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .sentence_splitter import split_sentences
from .utils import word_count


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_jsonl(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    labels: list[str] = []
    words_h: list[int] = []
    words_a: list[int] = []
    sents_h: list[int] = []
    sents_a: list[int] = []
    aug_h = 0
    aug_a = 0
    n_h = n_a = 0
    domains: Counter[str] = Counter()
    generators: Counter[str] = Counter()
    gen_providers: Counter[str] = Counter()
    sample_kinds: Counter[str] = Counter()
    n_mixed = 0

    for row in _iter_jsonl(path):
        lab = row.get("label")
        labels.append(str(lab))
        ft = row.get("final_text") or ""
        dom = row.get("domain") or "unknown"
        domains[dom] += 1
        sk = row.get("sample_kind")
        if not sk:
            sk = "human" if lab == "human" else ("direct_ai" if lab == "ai" else str(lab or "unknown"))
        sample_kinds[str(sk)] += 1
        if sk == "mixed":
            n_mixed += 1
        wc = row.get("num_words")
        if wc is None:
            wc = word_count(ft)
        ns = row.get("num_sentences")
        if ns is None:
            ns = len(split_sentences(ft))
        aug = row.get("augmentation") or []
        has_aug = 1 if aug else 0
        if lab == "human":
            n_h += 1
            words_h.append(int(wc))
            sents_h.append(int(ns))
            aug_h += has_aug
        elif lab == "ai":
            n_a += 1
            words_a.append(int(wc))
            sents_a.append(int(ns))
            aug_a += has_aug
            gm = row.get("generator_model") or "unknown"
            generators[str(gm)] += 1
            gp = row.get("generator_provider") or "unknown"
            gen_providers[str(gp)] += 1

    def avg(xs: list[int]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    total = n_h + n_a
    kind_total = sum(sample_kinds.values()) or 1
    dom_total = sum(domains.values()) or 1
    gen_total = sum(generators.values()) or 1
    prov_total = sum(gen_providers.values()) or 1

    return {
        "path": str(path),
        "num_human": n_h,
        "num_ai": n_a,
        "num_mixed": n_mixed,
        "sample_kind_share": {k: round(v / kind_total, 4) for k, v in sample_kinds.most_common()},
        "avg_words_human": avg(words_h),
        "avg_words_ai": avg(words_a),
        "avg_sentences_human": avg(sents_h),
        "avg_sentences_ai": avg(sents_a),
        "augmentation_rate_human": round(aug_h / n_h, 4) if n_h else 0.0,
        "augmentation_rate_ai": round(aug_a / n_a, 4) if n_a else 0.0,
        "domains": {k: round(v / dom_total, 4) for k, v in domains.most_common()},
        "generators": {k: round(v / gen_total, 4) for k, v in generators.most_common()},
        "generator_providers": {k: round(v / prov_total, 4) for k, v in gen_providers.most_common()},
        "label_balance_ai_share": round(n_a / total, 4) if total else 0.0,
        "label_balance_mixed_share": round(n_mixed / total, 4) if total else 0.0,
    }


def compare_summaries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "avg_words_human",
        "avg_words_ai",
        "augmentation_rate_human",
        "augmentation_rate_ai",
        "label_balance_ai_share",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in a and k in b:
            out[k] = {"a": a[k], "b": b[k], "diff": round(float(a[k]) - float(b[k]), 4)}
    return out
