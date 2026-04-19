from __future__ import annotations

import logging
import os
import time
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .generator_pools import GeneratorEntry
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

_DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"


def _str_from_cfg_or_env(
    openai_cfg: dict[str, Any], *, yaml_key: str, env_var: str, default: str = ""
) -> str:
    raw = openai_cfg.get(yaml_key)
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return (os.environ.get(env_var) or default).strip()


def azure_chat_completions_url(openai_cfg: dict[str, Any]) -> str:
    """Build Azure OpenAI chat/completions URL (deployment + api-version query)."""
    endpoint = _str_from_cfg_or_env(
        openai_cfg, yaml_key="azure_endpoint", env_var="AZURE_OPENAI_ENDPOINT"
    ).rstrip("/")
    deployment = _str_from_cfg_or_env(
        openai_cfg, yaml_key="azure_deployment", env_var="AZURE_OPENAI_DEPLOYMENT_NAME"
    )
    ver = _str_from_cfg_or_env(
        openai_cfg,
        yaml_key="azure_api_version",
        env_var="AZURE_OPENAI_API_VERSION",
        default=_DEFAULT_AZURE_API_VERSION,
    )
    if not endpoint or not deployment:
        return ""
    dep_enc = quote(deployment, safe="")
    return f"{endpoint}/openai/deployments/{dep_enc}/chat/completions?api-version={quote(ver, safe='')}"


def _parse_retry_after_sec(response: requests.Response) -> float | None:
    """Return seconds to wait from ``Retry-After`` (seconds or HTTP-date), or None if absent/invalid."""
    ra = (response.headers.get("Retry-After") or "").strip()
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        try:
            dt = parsedate_to_datetime(ra)
            if dt is None:
                return None
            return max(0.0, dt.timestamp() - time.time())
        except (TypeError, ValueError, OSError):
            return None


def is_openai_credential_ready(openai_cfg: dict[str, Any]) -> bool:
    """True if API key env is set and, for Azure mode, endpoint + deployment are available."""
    oa = dict(openai_cfg or {})
    env_name = str(oa.get("api_key_env") or "OPENAI_API_KEY")
    if not (os.environ.get(env_name) or "").strip():
        return False
    if not oa.get("use_azure"):
        return True
    return bool(azure_chat_completions_url(oa))


class OpenAINotConfiguredError(RuntimeError):
    """Raised when OpenAI-backed generators are enabled but credentials are missing."""


class OpenAIChatCompletionClient:
    """
    Minimal OpenAI Chat Completions client (HTTPS + JSON).
    Set model id in configs/models.yaml (e.g. gpt-5, gpt-4.1, gpt-4o); verify names in OpenAI docs.
    With ``openai.use_azure: true``, calls Azure OpenAI (deployment URL + ``api-key`` header);
    set ``AZURE_OPENAI_*`` env vars or the matching ``azure_*`` keys in YAML.
    """

    def __init__(self, gen_cfg: dict[str, Any], openai_cfg: dict[str, Any]):
        self.gen_cfg = gen_cfg
        self.openai_cfg = openai_cfg
        self.use_azure = bool(openai_cfg.get("use_azure"))
        self.base_url = str(openai_cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.api_key_env = str(openai_cfg.get("api_key_env") or "OPENAI_API_KEY")
        self.timeout = float(
            openai_cfg.get("request_timeout_sec", gen_cfg.get("request_timeout_sec", 240))
        )
        self.max_retries = int(openai_cfg.get("max_retries", 12))

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
        # gpt-5 / o-series on Azure often reject any temperature except the default (1).
        if bool(self.openai_cfg.get("force_default_sampling")):
            temperature = 1.0
            top_p = 1.0

        if self.use_azure:
            url = azure_chat_completions_url(self.openai_cfg)
            if not url:
                raise OpenAINotConfiguredError(
                    "Azure OpenAI: set azure_endpoint / AZURE_OPENAI_ENDPOINT and "
                    "azure_deployment / AZURE_OPENAI_DEPLOYMENT_NAME (or add them under openai: in models.yaml)."
                )
            headers: dict[str, str] = {
                "api-key": api_key,
                "Content-Type": "application/json",
            }
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {
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
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if not self.use_azure:
            body["model"] = model
        # Newer OpenAI models may expect max_completion_tokens; try both shapes on failure.
        body["max_tokens"] = max_tokens
        extra = self.openai_cfg.get("extra_request_json")
        if isinstance(extra, dict):
            body.update(extra)

        last_err: Exception | None = None
        retried_token_param = False
        retried_sampling_defaults = False
        cap = float(self.openai_cfg.get("retry_after_max_sec", 180))
        base = float(self.openai_cfg.get("retry_backoff_base_sec", 2.0))

        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 429:
                    parsed = _parse_retry_after_sec(r)
                    if parsed is not None:
                        wait = min(max(parsed, 1.0), cap)
                    else:
                        wait = min(base * (2 ** min(attempt, 10)), cap)
                    try:
                        err_body = str(r.json())[:400]
                    except ValueError:
                        err_body = (r.text or "")[:400]
                    last_err = requests.HTTPError(
                        f"429 Too Many Requests: {err_body or r.reason}",
                        response=r,
                    )
                    logger.warning(
                        "OpenAI rate limited (429), sleeping %.1fs then retry (%s/%s)%s",
                        wait,
                        attempt + 1,
                        self.max_retries,
                        f" (Retry-After header: {parsed}s)" if parsed is not None else "",
                    )
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
                if self.use_azure:
                    meta["backend"] = "azure_openai"
                    meta["azure_deployment"] = _str_from_cfg_or_env(
                        self.openai_cfg,
                        yaml_key="azure_deployment",
                        env_var="AZURE_OPENAI_DEPLOYMENT_NAME",
                    )
                return text, meta
            except requests.exceptions.HTTPError as e:
                last_err = e
                detail = ""
                if e.response is not None:
                    try:
                        detail = str(e.response.json())[:400]
                    except ValueError:
                        detail = (e.response.text or "")[:400]
                err_txt = (detail + str(e)).lower()
                # Some models reject max_tokens; retry once with max_completion_tokens
                if (
                    not retried_token_param
                    and e.response is not None
                    and e.response.status_code == 400
                ):
                    if "max_tokens" in err_txt and "max_completion_tokens" in err_txt:
                        body.pop("max_tokens", None)
                        body["max_completion_tokens"] = max_tokens
                        retried_token_param = True
                        logger.info("OpenAI rejected max_tokens; retrying with max_completion_tokens")
                        continue
                if (
                    not retried_sampling_defaults
                    and e.response is not None
                    and e.response.status_code == 400
                    and "temperature" in err_txt
                    and ("unsupported" in err_txt or "default" in err_txt)
                ):
                    body["temperature"] = 1.0
                    body["top_p"] = 1.0
                    temperature = 1.0
                    top_p = 1.0
                    retried_sampling_defaults = True
                    logger.info(
                        "OpenAI rejected non-default sampling; retrying with temperature=1, top_p=1"
                    )
                    continue
                logger.warning("OpenAI HTTP error (%s), retry %s: %s", e, attempt + 1, detail)
                time.sleep(1.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                logger.warning("OpenAI request failed (%s), retry %s", e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))

        tail = repr(last_err) if last_err is not None else "no error captured (unexpected)"
        raise RuntimeError(
            f"OpenAI generation failed after {self.max_retries} attempts (often 429 rate limits): {tail}"
        )


class OpenAIRouter:
    """
    One logical OpenAI Chat Completions integration per distinct (base_url, api_key_env, Azure flag).

    ``train_generators`` rows can set ``openai_base_url`` / ``openai_api_key_env`` to hit
    DeepSeek, DashScope compatible-mode, OpenRouter, etc., alongside the default ``openai:`` block.
    """

    def __init__(self, gen_cfg: dict[str, Any], global_openai_cfg: dict[str, Any]):
        self.gen_cfg = gen_cfg
        self.global_openai_cfg = dict(global_openai_cfg or {})
        self._cache: dict[tuple[str, str, bool], OpenAIChatCompletionClient] = {}

    def client_for_entry(self, entry: "GeneratorEntry") -> OpenAIChatCompletionClient:
        from .generator_pools import merge_openai_cfg_for_entry

        merged = merge_openai_cfg_for_entry(self.global_openai_cfg, entry)
        base = str(merged.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        envn = str(merged.get("api_key_env") or "OPENAI_API_KEY")
        key = (base, envn, bool(merged.get("use_azure")))
        if key not in self._cache:
            self._cache[key] = OpenAIChatCompletionClient(self.gen_cfg, merged)
        return self._cache[key]


__all__ = [
    "OpenAIChatCompletionClient",
    "OpenAINotConfiguredError",
    "OpenAIRouter",
    "azure_chat_completions_url",
    "is_openai_credential_ready",
]
