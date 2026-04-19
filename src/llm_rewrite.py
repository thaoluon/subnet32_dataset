from __future__ import annotations

import logging
import random
from typing import Any

from .ai_generator import (
    OllamaCompletionClient,
    is_text_model_name,
    sample_generation_params,
    strip_leading_boilerplate,
    trim_to_sentence_boundary,
)
from .generator_pools import GeneratorEntry
from .openai_generator import OpenAIRouter

logger = logging.getLogger(__name__)


def pick_hard_rewrite_system(gen_cfg: dict[str, Any]) -> tuple[str, str]:
    """
    Random system prompt for hard_ai diversity (paraphrase / tone / length variants).
    Returns (system_prompt, style_name).
    """
    raw = gen_cfg.get("hard_transform_systems")
    if isinstance(raw, list) and raw:
        entry = random.choice(raw)
        if isinstance(entry, dict):
            name = str(entry.get("name", "variant")).strip()
            sys = str(entry.get("system", "")).strip()
            if sys:
                return sys, name or "variant"
        elif isinstance(entry, str) and entry.strip():
            return entry.strip(), "line"
    legacy = str(gen_cfg.get("hard_transform_system", "")).strip()
    if legacy:
        return legacy, "default"
    return (
        "Rewrite the passage in fresh wording while preserving meaning and tone. "
        "Output only the rewritten text, no preamble.",
        "fallback",
    )


def _rewrite_params(gen_cfg: dict[str, Any]) -> dict[str, Any]:
    """Slightly lower temperature for paraphrase / hard transforms."""
    t0, t1 = gen_cfg.get("hard_transform_temperature_range", [0.2, 0.65])
    p0, p1 = gen_cfg.get("hard_transform_top_p_range", [0.75, 1.0])
    m0 = int(gen_cfg.get("hard_transform_max_tokens_min", 120))
    m1 = int(gen_cfg.get("hard_transform_max_tokens_max", 320))
    return {
        "temperature": float(random.uniform(float(t0), float(t1))),
        "top_p": float(random.uniform(float(p0), float(p1))),
        "num_predict": int(random.randint(m0, m1)),
        "repeat_penalty": float(
            random.uniform(
                float(gen_cfg.get("repeat_penalty_range", [1.0, 1.2])[0]),
                float(gen_cfg.get("repeat_penalty_range", [1.0, 1.2])[1]),
            )
        ),
    }


def llm_paraphrase(
    source_text: str,
    *,
    gen_entry: GeneratorEntry,
    system: str | None,
    ollama_client: OllamaCompletionClient | None,
    openai_router: OpenAIRouter | None,
    gen_cfg: dict[str, Any],
    use_raw: bool,
) -> tuple[str, dict[str, Any]]:
    """
    Rewrite `source_text` (paraphrase / light edit) using the same provider routing as generation.
    If ``system`` is None, a random prompt is chosen from ``hard_transform_systems`` (or legacy single string).
    """
    params = _rewrite_params(gen_cfg)
    style_name = "override"
    sys_prompt: str
    if system is None:
        sys_prompt, style_name = pick_hard_rewrite_system(gen_cfg)
    else:
        sys_prompt = system
    meta: dict[str, Any] = {
        "rewrite_stage": "hard_ai",
        "provider": gen_entry.provider,
        "rewrite_style": style_name,
    }

    if gen_entry.provider == "ollama":
        if ollama_client is None:
            raise RuntimeError("ollama client required for hard transform")
        raw_mode = use_raw and is_text_model_name(gen_entry.model)
        text, gen_meta = ollama_client.generate(
            gen_entry.model,
            source_text.strip(),
            params=params,
            raw_completion=raw_mode,
            system=sys_prompt if not raw_mode else None,
        )
        meta.update(gen_meta)
    elif gen_entry.provider == "openai":
        if openai_router is None:
            raise RuntimeError("openai router required for hard transform")
        oa_client = openai_router.client_for_entry(gen_entry)
        text, gen_meta = oa_client.generate(
            gen_entry.model,
            source_text.strip(),
            system=sys_prompt,
            params=params,
        )
        meta.update(gen_meta)
    else:
        raise RuntimeError(f"unknown provider {gen_entry.provider!r}")

    out = strip_leading_boilerplate(trim_to_sentence_boundary(text))
    return out, meta
