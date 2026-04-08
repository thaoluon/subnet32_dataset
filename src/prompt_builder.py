from __future__ import annotations

import random
from typing import Tuple


def build_prefix_span(
    sentences: list[str],
    prefix_min: int,
    prefix_max: int,
) -> Tuple[str, int]:
    """
    First k sentences as continuation prefix (same document as validator-style mining).
    Returns (prefix_text, num_prefix_sentences).
    """
    n = len(sentences)
    k = random.randint(
        max(1, min(prefix_min, n)),
        max(1, min(prefix_max, n)),
    )
    k = min(k, n)
    prefix_sents = sentences[:k]
    return " ".join(s.strip() for s in prefix_sents if s.strip()), k
