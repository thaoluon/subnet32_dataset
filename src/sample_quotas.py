from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BUCKET_KEYS = ("human", "direct_ai", "hard_ai", "mixed")


def parse_sample_targets(ds_cfg: dict[str, Any]) -> dict[str, int] | None:
    """Return integer targets if quota mode is enabled, else None."""
    if not ds_cfg.get("use_sample_quotas"):
        return None
    raw = dict(ds_cfg.get("sample_targets") or {})
    out: dict[str, int] = {}
    for k in BUCKET_KEYS:
        if k not in raw:
            raise ValueError(f"use_sample_quotas requires sample_targets.{k} in datasets.yaml")
        out[k] = int(raw[k])
        if out[k] < 0:
            raise ValueError(f"sample_targets.{k} must be >= 0")
    return out


def pick_bucket(counts: dict[str, int], targets: dict[str, int]) -> str | None:
    """Uniformly pick an undershot bucket, or None if all filled."""
    choices = [k for k in BUCKET_KEYS if counts[k] < targets[k]]
    if not choices:
        return None
    return random.choice(choices)


def quotas_complete(counts: dict[str, int], targets: dict[str, int]) -> bool:
    return all(counts[k] >= targets[k] for k in BUCKET_KEYS)


def load_counts_json(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s: %s", path, e)
        return None
    out = {k: int(data.get(k, 0)) for k in BUCKET_KEYS}
    return out


def save_counts_json(path: Path, counts: dict[str, int], targets: dict[str, int]) -> None:
    payload = {k: counts[k] for k in BUCKET_KEYS}
    payload["targets"] = {k: targets[k] for k in BUCKET_KEYS}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_hard_transform_pool(model_cfg: dict[str, Any]) -> list[Any]:
    """Optional dedicated models for paraphrase; else reuse train_generators."""
    from .generator_pools import generators_from_yaml_list, resolve_train_pool

    raw = model_cfg.get("hard_transform_generators")
    if raw:
        return generators_from_yaml_list(list(raw))
    return resolve_train_pool(model_cfg)
