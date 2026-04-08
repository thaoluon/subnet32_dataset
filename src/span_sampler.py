from __future__ import annotations

import random
from typing import Tuple


def sample_consecutive_span(
    sentences: list[str],
    span_min: int,
    span_max: int,
) -> Tuple[list[str], int, int]:
    """Return (span_sentences, start_idx, end_idx_exclusive)."""
    n = len(sentences)
    lo = max(1, min(span_min, n))
    hi = max(lo, min(span_max, n))
    length = random.randint(lo, hi)
    if length > n:
        length = n
    start = random.randint(0, n - length)
    end = start + length
    return sentences[start:end], start, end


def join_span(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences if s.strip())
