from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_PUNKT_READY = False


def _ensure_nltk_punkt() -> None:
    global _PUNKT_READY
    if _PUNKT_READY:
        return
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    _PUNKT_READY = True


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences; keeps trailing space/newline glue minimal."""
    text = text.strip()
    if not text:
        return []
    _ensure_nltk_punkt()
    try:
        import nltk

        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception as e:
        logger.warning("nltk sentence split failed (%s); using regex fallback", e)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]


def is_garbage_document(sentences: list[str], min_sentences: int = 4) -> bool:
    if len(sentences) < min_sentences:
        return True
    joined = " ".join(sentences)
    if len(joined) < 200:
        return True
    alpha_ratio = sum(c.isalpha() or c.isspace() for c in joined) / max(len(joined), 1)
    if alpha_ratio < 0.55:
        return True
    code_markers = sum(joined.count(x) for x in ("```", "{", "}", "def ", "import ", "SELECT ", "#include"))
    if code_markers > 12 and alpha_ratio < 0.65:
        return True
    return False
