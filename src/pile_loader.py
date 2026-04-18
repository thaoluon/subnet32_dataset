from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


def _is_retryable_hf_stream_error(exc: BaseException) -> bool:
    """True when a streaming shard read fails in a way that often succeeds after reconnect."""
    if _is_retryable_hub_error(exc):
        return True
    if isinstance(exc, FileNotFoundError):
        s = str(exc).lower()
        return "hf://" in s or "zstd://" in s or "huggingface.co" in s
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if any(x in msg for x in ("errno 110", "timed out", "connection reset", "broken pipe")):
            return True
    return False


def _is_retryable_hub_error(exc: BaseException) -> bool:
    """True for transient Hub/network errors (504 Gateway Timeout, rate limits, etc.)."""
    msg = str(exc).lower()
    if any(
        x in msg
        for x in (
            "504",
            "503",
            "502",
            "429",
            "gateway time-out",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "bad gateway",
        )
    ):
        return True
    cur: BaseException | None = exc
    seen = 0
    while cur is not None and seen < 8:
        seen += 1
        if type(cur).__name__ == "HfHubHTTPError":
            resp = getattr(cur, "response", None)
            code = getattr(resp, "status_code", None) if resp is not None else None
            if code in (429, 502, 503, 504):
                return True
        cur = cur.__cause__ or getattr(cur, "__context__", None)  # type: ignore[assignment]
    if isinstance(exc, (TimeoutError, ConnectionError, BrokenPipeError)):
        return True
    return False


def load_dataset_streaming_with_retry(
    name: str,
    split: str,
    *,
    max_retries: int,
    base_delay_sec: float,
    max_delay_sec: float,
):
    """Call ``datasets.load_dataset(..., streaming=True)`` with backoff on transient Hub errors."""
    from datasets import load_dataset

    last: BaseException | None = None
    for attempt in range(max(1, max_retries)):
        try:
            return load_dataset(name, split=split, streaming=True)
        except Exception as e:
            last = e
            if attempt >= max_retries - 1 or not _is_retryable_hub_error(e):
                raise
            exp = min(max_delay_sec, base_delay_sec * (2**attempt))
            jitter = random.uniform(0, min(10.0, base_delay_sec))
            wait = exp + jitter
            logger.warning(
                "Hugging Face load failed (%s); retry %s/%s in %.0fs "
                "(transient Hub errors are common; or use source: local + local_jsonl_path).",
                e,
                attempt + 1,
                max_retries,
                wait,
            )
            time.sleep(wait)
    assert last is not None
    raise last


@dataclass
class PileDocument:
    text: str
    doc_id: str
    pile_subset: str | None
    coarse_domain: str


def _parse_meta(meta: Any) -> dict[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return {}


def _subset_name(meta: dict[str, Any]) -> str | None:
    for key in ("pile_set_name", "subset", "pile_subset"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def map_to_coarse_domain(
    subset: str | None,
    mapping: dict[str, str],
    default: str = "web",
) -> str:
    if subset and subset in mapping:
        return mapping[subset]
    if subset:
        sl = subset.lower()
        for k, v in mapping.items():
            if k.lower() == sl:
                return v
    return default


class PileLoader:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.source = config.get("source", "huggingface")
        self.text_field = config.get("text_field", "text")
        self.meta_field = config.get("meta_field", "meta")
        self.subset_map: dict[str, str] = dict(config.get("pile_subset_to_domain") or {})
        self.default_coarse_domain: str = str(config.get("default_coarse_domain", "web"))
        self._counter = 0

    def iter_documents(self) -> Iterator[PileDocument]:
        if self.source == "local":
            yield from self._iter_local()
        else:
            yield from self._iter_hf()

    def _iter_local(self) -> Iterator[PileDocument]:
        path = self.config.get("local_jsonl_path")
        if not path:
            raise ValueError("datasets.yaml: local_jsonl_path required when source=local")
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(path)
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get(self.text_field) or row.get("text")
                if not text:
                    continue
                meta = _parse_meta(row.get(self.meta_field))
                subset = _subset_name(meta)
                cid = row.get("id", f"local_{self._counter}")
                self._counter += 1
                coarse = map_to_coarse_domain(subset, self.subset_map, self.default_coarse_domain)
                yield PileDocument(
                    text=str(text),
                    doc_id=str(cid),
                    pile_subset=subset,
                    coarse_domain=coarse,
                )

    def _iter_hf(self) -> Iterator[PileDocument]:
        import os

        name = self.config.get("huggingface_dataset", "monology/pile-uncopyrighted")
        split = self.config.get("split", "train")
        buf = int(self.config.get("shuffle_buffer_size", 100_000))
        seed = int(self.config.get("random_seed", int(time.time())))
        max_retries = int(self.config.get("hf_load_max_retries", 15))
        base_delay = float(self.config.get("hf_load_retry_base_sec", 20))
        max_delay = float(self.config.get("hf_load_retry_max_sec", 180))
        stream_reconnects = int(self.config.get("hf_stream_max_reconnects", 30))
        stream_base = float(self.config.get("hf_stream_reconnect_base_sec", 15))
        stream_max = float(self.config.get("hf_stream_reconnect_max_sec", 120))
        timeout_cfg = self.config.get("hf_hub_download_timeout_sec")
        if timeout_cfg is not None and "HF_HUB_DOWNLOAD_TIMEOUT" not in os.environ:
            os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(int(timeout_cfg))

        num_shards = max(1, int(self.config.get("stream_num_shards", 1)))
        shard_index = int(self.config.get("stream_shard_index", 0))
        if not (0 <= shard_index < num_shards):
            raise ValueError(
                f"datasets.yaml: stream_shard_index must be in [0, stream_num_shards), got {shard_index}/{num_shards}"
            )

        logger.info(
            "Loading streaming dataset %s split=%s shuffle_buffer=%s shard=%s/%s",
            name,
            split,
            buf,
            shard_index,
            num_shards,
        )

        for attempt in range(max(1, stream_reconnects)):
            try:
                ds = load_dataset_streaming_with_retry(
                    name,
                    split,
                    max_retries=max_retries,
                    base_delay_sec=base_delay,
                    max_delay_sec=max_delay,
                )
                ds = ds.shuffle(seed=seed, buffer_size=buf)
                for row in ds:
                    stream_pos = self._counter
                    self._counter += 1
                    if num_shards > 1 and (stream_pos % num_shards) != shard_index:
                        continue
                    text = row.get(self.text_field)
                    if not text:
                        continue
                    meta = _parse_meta(row.get(self.meta_field))
                    subset = _subset_name(meta)
                    cid = f"hf_s{shard_index}_{stream_pos}" if num_shards > 1 else f"hf_{stream_pos}"
                    coarse = map_to_coarse_domain(subset, self.subset_map, self.default_coarse_domain)
                    yield PileDocument(
                        text=str(text),
                        doc_id=cid,
                        pile_subset=subset,
                        coarse_domain=coarse,
                    )
            except Exception as e:
                if attempt >= stream_reconnects - 1 or not _is_retryable_hf_stream_error(e):
                    raise
                exp = min(stream_max, stream_base * (2**attempt))
                jitter = random.uniform(0, min(10.0, stream_base))
                wait = exp + jitter
                logger.warning(
                    "HF streaming read failed (%s); reconnect %s/%s in %.0fs "
                    "(unstable links: raise hf_hub_download_timeout_sec or use source: local).",
                    e,
                    attempt + 1,
                    stream_reconnects,
                    wait,
                )
                time.sleep(wait)
