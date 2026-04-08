from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .sentence_splitter import split_sentences
from .utils import word_count


@dataclass
class QualityConfig:
    word_count_min: int = 80
    word_count_max: int = 300
    min_sentences: int = 3
    max_non_alnum_ratio: float = 0.18
    max_line_repetition: float = 0.35


_ASSISTANT_START = re.compile(
    r"^(sure[,! ]|here is|here's|as an ai|i'm an ai|i cannot|#+ |\*\s)",
    re.IGNORECASE,
)


def non_alnum_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    return bad / len(text)


def repetition_score(text: str) -> float:
    words = text.lower().split()
    if len(words) < 10:
        return 0.0
    uniq = len(set(words))
    return 1.0 - (uniq / len(words))


def prefix_overlap_ratio(prefix: str, full_text: str) -> float:
    """High value → completion barely adds new content (bad for AI samples)."""
    pw = set(prefix.lower().split())
    fw = full_text.lower().split()
    if not fw:
        return 1.0
    new = [w for w in fw if w not in pw]
    return 1.0 - (len(new) / len(fw))


def passes_quality(
    text: str,
    cfg: QualityConfig | dict[str, Any],
    *,
    is_ai: bool = False,
    prefix: str | None = None,
) -> tuple[bool, str]:
    if isinstance(cfg, dict):
        qc = QualityConfig(
            word_count_min=int(cfg.get("word_count_min", 80)),
            word_count_max=int(cfg.get("word_count_max", 300)),
            min_sentences=int(cfg.get("min_sentences", 3)),
            max_non_alnum_ratio=float(cfg.get("max_non_alnum_ratio", 0.18)),
            max_line_repetition=float(cfg.get("max_line_repetition", 0.35)),
        )
    else:
        qc = cfg

    t = text.strip()
    if not t:
        return False, "empty"
    wc = word_count(t)
    if wc < qc.word_count_min:
        return False, "too_short"
    if wc > qc.word_count_max:
        return False, "too_long"

    sents = split_sentences(t)
    if len(sents) < qc.min_sentences:
        return False, "too_few_sentences"

    if non_alnum_ratio(t) > qc.max_non_alnum_ratio:
        return False, "too_many_symbols"

    if repetition_score(t) > qc.max_line_repetition:
        return False, "repetitive"

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if len(lines) > 5:
        bulletish = sum(1 for ln in lines if ln.startswith(("-", "*", "•", "1.", "2.", "3.")))
        if bulletish / len(lines) > 0.4:
            return False, "list_heavy"

    first_line = lines[0] if lines else t[:120]
    if _ASSISTANT_START.search(first_line):
        return False, "assistant_boilerplate"

    code_hits = sum(
        t.count(x)
        for x in (
            "def ",
            "import ",
            "SELECT ",
            "#include",
            "func ",
            "package ",
        )
    )
    if code_hits >= 3 and wc < 200:
        return False, "code_heavy"

    if is_ai and prefix:
        # High ratio → most tokens already appeared in prefix (weak continuation).
        if prefix_overlap_ratio(prefix, t) > 0.88:
            return False, "ai_too_similar_to_prefix"

    return True, "ok"
