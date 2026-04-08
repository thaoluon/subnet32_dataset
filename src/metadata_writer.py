from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO

from .utils import ensure_dir


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._fp: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        self._fp = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def write(self, record: dict[str, Any]) -> None:
        assert self._fp is not None
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
