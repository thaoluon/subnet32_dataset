from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OllamaUnreachableError(RuntimeError):
    """Raised when the Ollama HTTP API cannot be reached before generation starts."""


def assert_ollama_reachable(base_url: str, *, timeout_sec: float = 5.0) -> None:
    """Fail fast with a clear message if Ollama is not listening (e.g. connection refused)."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=timeout_sec)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise OllamaUnreachableError(
            f"Ollama is not reachable at {base_url} ({e}). "
            "Start the Ollama app (or run `ollama serve`), ensure models are pulled, "
            "or build human-only JSONL with: python -m src.dataset_builder --skip-ai ..."
        ) from e


_ASSISTANT_PATTERNS = re.compile(
    r"^(sure[,! ]|here is|here's|okay[,! ]|as an ai|i cannot|i'm an ai)",
    re.IGNORECASE | re.MULTILINE,
)


def sample_generation_params(gen_cfg: dict[str, Any]) -> dict[str, Any]:
    t0, t1 = gen_cfg.get("temperature_range", [0.3, 1.2])
    p0, p1 = gen_cfg.get("top_p_range", [0.7, 1.0])
    m0, m1 = gen_cfg.get("max_tokens_range", [80, 220])
    r0, r1 = gen_cfg.get("repeat_penalty_range", [1.0, 1.2])
    return {
        "temperature": float(random.uniform(t0, t1)),
        "top_p": float(random.uniform(p0, p1)),
        "num_predict": int(random.randint(int(m0), int(m1))),
        "repeat_penalty": float(random.uniform(r0, r1)),
    }


class OllamaCompletionClient:
    def __init__(self, base_url: str, gen_cfg: dict[str, Any]):
        self.base_url = base_url.rstrip("/")
        self.gen_cfg = gen_cfg
        self.timeout = float(gen_cfg.get("request_timeout_sec", 240))

    def generate(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any] | None = None,
        raw_completion: bool = False,
        system: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        p = sample_generation_params(self.gen_cfg) if params is None else dict(params)
        num_predict = p.get("num_predict") or self.gen_cfg.get("num_predict", 256)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": p["temperature"],
                "top_p": p["top_p"],
                "repeat_penalty": p["repeat_penalty"],
                "num_predict": int(num_predict),
            },
        }
        if not raw_completion and system:
            payload["system"] = system

        last_err = None
        for attempt in range(5):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()
                text = (data.get("response") or "").strip()
                meta = {
                    "temperature": p["temperature"],
                    "top_p": p["top_p"],
                    "repeat_penalty": p["repeat_penalty"],
                    "max_tokens": int(num_predict),
                }
                return text, meta
            except Exception as e:
                last_err = e
                logger.warning("Ollama generate failed (%s), retry %s", e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama generation failed after retries: {last_err}")


def trim_to_sentence_boundary(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    matches = list(re.finditer(r'[.!?][\'"]?(?=\s|$)', text))
    if not matches:
        return text
    end = matches[-1].end()
    return text[:end].strip()


def strip_leading_boilerplate(text: str) -> str:
    lines = text.split("\n")
    out = []
    started = False
    for line in lines:
        s = line.strip()
        if not started:
            if _ASSISTANT_PATTERNS.search(s):
                continue
            started = True
        if started:
            out.append(line)
    return "\n".join(out).strip()


def combine_prefix_completion(prefix: str, completion: str) -> str:
    prefix = prefix.rstrip()
    completion = completion.lstrip()
    if not completion:
        return prefix
    if (
        prefix
        and completion
        and not prefix[-1].isspace()
        and completion[0] not in " \n\t.,;:!?'\")}]}"
    ):
        return f"{prefix} {completion}"
    return f"{prefix}{completion}"


def is_text_model_name(model: str) -> bool:
    return ":text" in model.lower() or model.lower().endswith("-text")
