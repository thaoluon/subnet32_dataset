from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load configs. pip install -r subnet32_dataset/requirements.txt"
        ) from e
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace — catches near-identical copies for dedup."""
    t = (text or "").lower().strip()
    return re.sub(r"\s+", " ", t)


def normalized_text_hash(text: str) -> str:
    return text_sha256(normalize_for_dedup(text))


def word_count(text: str) -> int:
    return len(text.split())


def count_sentences(sentences: list[str]) -> int:
    return len([s for s in sentences if s.strip()])


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
