from __future__ import annotations

import random
from typing import Any


def pick_split(ratios: dict[str, float], stress_fraction: float) -> tuple[str, bool]:
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


class DocSplitRegistry:
    """
    Assign train/val/test/stress once per source document so all derived rows
    (human, direct_ai, hard_ai, mixed) stay in the same split — reduces leakage.
    """

    def __init__(self, split_ratios: dict[str, float], stress_fraction: float):
        self.split_ratios = split_ratios
        self.stress_fraction = stress_fraction
        self._cache: dict[str, tuple[str, bool]] = {}

    def split_for(self, doc_id: str) -> tuple[str, bool]:
        if doc_id not in self._cache:
            self._cache[doc_id] = pick_split(self.split_ratios, self.stress_fraction)
        return self._cache[doc_id]

    def as_dict(self) -> dict[str, Any]:
        return {"cached_documents": len(self._cache)}
