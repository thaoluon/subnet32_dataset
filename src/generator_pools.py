from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Any

from .ai_generator import assert_all_ollama_reachable
from .openai_generator import OpenAINotConfiguredError, is_openai_credential_ready

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratorEntry:
    """One row in configs/models.yaml train_generators / stress_generators."""

    provider: str
    model: str
    weight: float = 1.0

    def record_model_id(self) -> str:
        """Stable id for JSONL (model tag only; use generator_provider for backend)."""
        return self.model


def _as_float(x: Any, default: float = 1.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_entry(raw: dict[str, Any]) -> GeneratorEntry:
    prov = str(raw.get("provider") or "").strip().lower()
    model = str(raw.get("model") or "").strip()
    if prov not in ("ollama", "openai"):
        raise ValueError(f"generator entry has invalid provider {raw!r} (use ollama or openai)")
    if not model:
        raise ValueError(f"generator entry missing model: {raw!r}")
    w = _as_float(raw.get("weight"), 1.0)
    if w < 0:
        raise ValueError(f"generator entry weight must be >= 0: {raw!r}")
    return GeneratorEntry(provider=prov, model=model, weight=w)


def generators_from_yaml_list(items: list[Any] | None) -> list[GeneratorEntry]:
    out: list[GeneratorEntry] = []
    for item in items or []:
        if not isinstance(item, dict):
            raise ValueError(f"generator entry must be a mapping, got {type(item)}")
        out.append(_parse_entry(item))
    pos = [e for e in out if e.weight > 0]
    return pos if pos else out


def legacy_train_entries(model_cfg: dict[str, Any]) -> list[GeneratorEntry]:
    names = list(model_cfg.get("train_models") or [])
    default_model = model_cfg.get("default_model")
    if not names and default_model:
        names = [str(default_model)]
    return [GeneratorEntry(provider="ollama", model=str(m), weight=1.0) for m in names]


def legacy_stress_entries(model_cfg: dict[str, Any]) -> list[GeneratorEntry]:
    names = list(model_cfg.get("stress_models") or [])
    return [GeneratorEntry(provider="ollama", model=str(m), weight=1.0) for m in names]


def resolve_train_pool(model_cfg: dict[str, Any]) -> list[GeneratorEntry]:
    if "train_generators" in model_cfg and model_cfg["train_generators"] is not None:
        g = generators_from_yaml_list(list(model_cfg["train_generators"]))
        if g:
            return g
        raise ValueError(
            "configs/models.yaml: train_generators is empty. "
            "Add provider entries or remove train_generators to use legacy train_models."
        )
    return legacy_train_entries(model_cfg)


def resolve_stress_pool(model_cfg: dict[str, Any]) -> list[GeneratorEntry]:
    if "stress_generators" in model_cfg and model_cfg["stress_generators"] is not None:
        return generators_from_yaml_list(list(model_cfg["stress_generators"]))
    return legacy_stress_entries(model_cfg)


def pick_generator_entry(pool: list[GeneratorEntry]) -> GeneratorEntry:
    if not pool:
        raise ValueError("empty generator pool")
    weights = [max(e.weight, 0.0) for e in pool]
    s = sum(weights)
    if s <= 0:
        return random.choice(pool)
    r = random.random() * s
    acc = 0.0
    for e, w in zip(pool, weights, strict=True):
        acc += w
        if r <= acc:
            return e
    return pool[-1]


def providers_used(pools: list[list[GeneratorEntry]]) -> set[str]:
    s: set[str] = set()
    for pool in pools:
        for e in pool:
            s.add(e.provider)
    return s


def apply_ai_throughput_phase(
    train_pool: list[GeneratorEntry],
    stress_pool: list[GeneratorEntry],
    *,
    phase: str,
    ollama_phase_model: str | None,
) -> tuple[list[GeneratorEntry], list[GeneratorEntry]]:
    """
    Reduce provider mixing so Ollama keeps one model loaded (or skip Ollama entirely).

    - ``all``: no change.
    - ``openai_only``: only ``provider: openai`` rows (Ollama idle for this run).
    - ``ollama_locked``: only ``provider: ollama`` rows whose ``model`` equals ``ollama_phase_model``.
      If the tag exists in only one of train/stress, the other pool reuses those entries so both pools stay valid.
    """
    ph = (phase or "all").strip().lower()
    if ph in ("all", "default", ""):
        return train_pool, stress_pool
    if ph == "openai_only":
        tr = [e for e in train_pool if e.provider == "openai"]
        st = [e for e in stress_pool if e.provider == "openai"]
        if not tr or not st:
            raise ValueError(
                "ai_phase=openai_only needs at least one openai generator in both train_generators and "
                f"stress_generators (got train_openai={len(tr)} stress_openai={len(st)})."
            )
        logger.info(
            "ai_phase=openai_only: using %s train + %s stress OpenAI generators (Ollama unused this run).",
            len(tr),
            len(st),
        )
        return tr, st
    if ph == "ollama_locked":
        tag = (ollama_phase_model or "").strip()
        if not tag:
            raise ValueError("ai_phase=ollama_locked requires --ollama-phase-model (exact Ollama model tag).")
        tr = [e for e in train_pool if e.provider == "ollama" and e.model == tag]
        st = [e for e in stress_pool if e.provider == "ollama" and e.model == tag]
        if not tr and not st:
            avail = sorted({e.model for e in train_pool + stress_pool if e.provider == "ollama"})
            raise ValueError(f"ollama_locked: no generator with model={tag!r}. Ollama tags in yaml: {avail}")
        if not tr:
            tr = list(st)
        if not st:
            st = list(tr)
        logger.info(
            "ai_phase=ollama_locked model=%s: %s train + %s stress Ollama generators (OpenAI unused this run).",
            tag,
            len(tr),
            len(st),
        )
        return tr, st
    raise ValueError(f"Unknown ai_phase={phase!r} (use all, openai_only, ollama_locked).")


def filter_pool_for_ai_phase(
    pool: list[GeneratorEntry],
    *,
    phase: str,
    ollama_phase_model: str | None,
    fallback: list[GeneratorEntry],
) -> list[GeneratorEntry]:
    """Filter an auxiliary pool (e.g. hard_transform) to match ``apply_ai_throughput_phase`` semantics."""
    ph = (phase or "all").strip().lower()
    if ph in ("all", "default", ""):
        return pool
    if ph == "openai_only":
        out = [e for e in pool if e.provider == "openai"]
        if out:
            return out
        out = [e for e in fallback if e.provider == "openai"]
        if out:
            return out
        raise ValueError("openai_only: hard_transform_generators has no openai entries and train pool has none.")
    if ph == "ollama_locked":
        tag = (ollama_phase_model or "").strip()
        out = [e for e in pool if e.provider == "ollama" and e.model == tag]
        if out:
            return out
        out = [e for e in fallback if e.provider == "ollama" and e.model == tag]
        if out:
            return out
        return [GeneratorEntry(provider="ollama", model=tag, weight=1.0)]
    return pool


def strip_openai_if_key_missing(
    model_cfg: dict[str, Any], train_pool: list[GeneratorEntry], stress_pool: list[GeneratorEntry]
) -> tuple[list[GeneratorEntry], list[GeneratorEntry]]:
    """
    When openai.omit_if_no_api_key is true and the API key env is empty, drop OpenAI rows
    so local Ollama-only runs still work (dataset will not contain GPT-class text).
    """
    oa = dict(model_cfg.get("openai") or {})
    if not oa.get("omit_if_no_api_key"):
        return train_pool, stress_pool
    if is_openai_credential_ready(oa):
        return train_pool, stress_pool
    env_name = str(oa.get("api_key_env") or "OPENAI_API_KEY")
    logger.warning(
        "OpenAI/Azure credentials incomplete (%s or Azure endpoint/deployment); "
        "removing openai generators because openai.omit_if_no_api_key is true",
        env_name,
    )
    return (
        [e for e in train_pool if e.provider != "openai"],
        [e for e in stress_pool if e.provider != "openai"],
    )


def assert_generators_configured(
    *,
    skip_ai: bool,
    ollama_base_urls: list[str],
    model_cfg: dict[str, Any],
    gen_cfg: dict[str, Any],
    train_pool: list[GeneratorEntry],
    stress_pool: list[GeneratorEntry],
) -> None:
    if skip_ai:
        return
    prov = providers_used([train_pool, stress_pool])
    health_timeout = float(gen_cfg.get("ollama_health_timeout_sec", 5))
    if "ollama" in prov:
        assert_all_ollama_reachable(ollama_base_urls, timeout_sec=health_timeout)
    if "openai" in prov:
        oa = dict(model_cfg.get("openai") or {})
        if not is_openai_credential_ready(oa):
            env_name = str(oa.get("api_key_env") or "OPENAI_API_KEY")
            if oa.get("use_azure"):
                raise OpenAINotConfiguredError(
                    f"Azure OpenAI not fully configured ({env_name}, AZURE_OPENAI_ENDPOINT, "
                    f"AZURE_OPENAI_DEPLOYMENT_NAME, and optionally AZURE_OPENAI_API_VERSION) "
                    f"but configs/models.yaml includes openai generators. "
                    f"Set env vars / openai.azure_* keys, remove openai rows, or use --skip-ai."
                )
            raise OpenAINotConfiguredError(
                f"{env_name} is not set but configs/models.yaml includes openai generators. "
                f"Export your API key, remove openai rows, or use --skip-ai."
            )


__all__ = [
    "GeneratorEntry",
    "apply_ai_throughput_phase",
    "assert_generators_configured",
    "filter_pool_for_ai_phase",
    "generators_from_yaml_list",
    "legacy_stress_entries",
    "legacy_train_entries",
    "pick_generator_entry",
    "providers_used",
    "resolve_stress_pool",
    "resolve_train_pool",
    "strip_openai_if_key_missing",
]
