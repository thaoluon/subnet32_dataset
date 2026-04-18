from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OpenAINotConfiguredError(RuntimeError):
    """Raised when OpenAI-backed generators are enabled but credentials are missing."""


class OpenAIChatCompletionClient:
    """
    Minimal OpenAI Chat Completions client (HTTPS + JSON).
    Set model id in configs/models.yaml (e.g. gpt-5, gpt-4.1, gpt-4o); verify names in OpenAI docs.
    """

    def __init__(self, gen_cfg: dict[str, Any], openai_cfg: dict[str, Any]):
        self.gen_cfg = gen_cfg
        self.openai_cfg = openai_cfg
        self.base_url = str(openai_cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = str(openai_cfg.get("api_key_env") or "OPENAI_API_KEY")
        self.timeout = float(
            openai_cfg.get("request_timeout_sec", gen_cfg.get("request_timeout_sec", 240))
        )
        self.max_retries = int(openai_cfg.get("max_retries", 5))

    def generate(
        self,
        model: str,
        user_content: str,
        *,
        system: str | None,
        params: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        api_key = (os.environ.get(self.api_key_env) or "").strip()
        if not api_key:
            raise OpenAINotConfiguredError(f"Missing environment variable {self.api_key_env}")

        p = dict(params or {})
        temperature = float(p.get("temperature", 0.7))
        top_p = float(p.get("top_p", 0.95))
        max_tokens = int(p.get("num_predict") or self.gen_cfg.get("num_predict", 256))

        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        org = self.openai_cfg.get("organization")
        if org:
            headers["OpenAI-Organization"] = str(org)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        # Newer OpenAI models may expect max_completion_tokens; try both shapes on failure.
        body["max_tokens"] = max_tokens
        extra = self.openai_cfg.get("extra_request_json")
        if isinstance(extra, dict):
            body.update(extra)

        last_err: Exception | None = None
        retried_token_param = False
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429:
                    wait = float(self.openai_cfg.get("retry_backoff_base_sec", 2.0)) * (attempt + 1)
                    logger.warning("OpenAI rate limited (429), retry in %ss", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenAI empty choices: {data!r}"[:500])
                msg = (choices[0].get("message") or {}) if isinstance(choices[0], dict) else {}
                text = (msg.get("content") or "").strip()
                meta: dict[str, Any] = {
                    "provider": "openai",
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                }
                return text, meta
            except requests.exceptions.HTTPError as e:
                last_err = e
                detail = ""
                if e.response is not None:
                    try:
                        detail = str(e.response.json())[:400]
                    except ValueError:
                        detail = (e.response.text or "")[:400]
                # Some models reject max_tokens; retry once with max_completion_tokens
                if (
                    not retried_token_param
                    and e.response is not None
                    and e.response.status_code == 400
                ):
                    err_txt = (detail + str(e)).lower()
                    if "max_tokens" in err_txt and "max_completion_tokens" in err_txt:
                        body.pop("max_tokens", None)
                        body["max_completion_tokens"] = max_tokens
                        retried_token_param = True
                        logger.info("OpenAI rejected max_tokens; retrying with max_completion_tokens")
                        continue
                logger.warning("OpenAI HTTP error (%s), retry %s: %s", e, attempt + 1, detail)
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                logger.warning("OpenAI request failed (%s), retry %s", e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"OpenAI generation failed after retries: {last_err}")


__all__ = ["OpenAIChatCompletionClient", "OpenAINotConfiguredError"]
