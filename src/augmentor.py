from __future__ import annotations

import logging
import random
import re
import string
from typing import Any

logger = logging.getLogger(__name__)

KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfxc",
    "e": "rdsw",
    "f": "drtgcv",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiol",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"


def _pick_word_indices(text: str) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\b[a-zA-Z]{3,}\b", text))
    return [(m.start(), m.end()) for m in words]


def _random_char_in_word(word: str) -> int:
    return random.randint(0, len(word) - 1)


def aug_swap(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b]
    if len(word) < 2:
        return s
    i = random.randint(0, len(word) - 2)
    arr = list(word)
    arr[i], arr[i + 1] = arr[i + 1], arr[i]
    return s[:a] + "".join(arr) + s[b:]


def aug_delete(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b]
    i = _random_char_in_word(word)
    return s[: a + i] + s[a + i + 1 :]


def aug_insert(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b]
    i = random.randint(0, len(word))
    ch = random.choice(string.ascii_lowercase)
    return s[: a + i] + ch + s[a + i :]


def aug_replace(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b].lower()
    i = _random_char_in_word(word)
    ch = random.choice(string.ascii_lowercase)
    nw = list(word)
    nw[i] = ch
    return s[:a] + "".join(nw) + s[b:]


def aug_duplicate(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b]
    i = _random_char_in_word(word)
    return s[: a + i + 1] + word[i] + s[a + i + 1 :]


def aug_keyboard_typo(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = list(s[a:b].lower())
    for _ in range(8):
        i = random.randint(0, len(word) - 1)
        c = word[i]
        neigh = KEYBOARD_NEIGHBORS.get(c, "")
        if neigh:
            word[i] = random.choice(neigh)
            return s[:a] + "".join(word) + s[b:]
    return aug_replace(s)


def aug_case_flip(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = s[a:b]
    i = _random_char_in_word(word)
    ch = word[i]
    flip = ch.lower() if ch.isupper() else ch.upper()
    return s[: a + i] + flip + s[a + i + 1 :]


def aug_vowel_sub(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = list(s[a:b].lower())
    vowel_idx = [i for i, c in enumerate(word) if c in VOWELS]
    if not vowel_idx:
        return aug_replace(s)
    i = random.choice(vowel_idx)
    opts = [x for x in VOWELS if x != word[i]] or list(VOWELS)
    word[i] = random.choice(opts)
    return s[:a] + "".join(word) + s[b:]


def aug_consonant_sub(s: str) -> str:
    words = _pick_word_indices(s)
    if not words:
        return s
    a, b = random.choice(words)
    word = list(s[a:b].lower())
    cons_idx = [i for i, c in enumerate(word) if c in CONSONANTS]
    if not cons_idx:
        return aug_replace(s)
    i = random.choice(cons_idx)
    opts = [x for x in CONSONANTS if x != word[i]] or list(CONSONANTS)
    word[i] = random.choice(opts)
    return s[:a] + "".join(word) + s[b:]


def aug_repeat_drop(s: str) -> str:
    for m in re.finditer(r"(.)\1{1,3}", s):
        if random.random() < 0.5:
            a, b = m.start(), m.end()
            return s[: a + 1] + s[b:]
    return aug_delete(s)


CHAR_AUG_FUNCS: dict[str, Any] = {
    "swap": aug_swap,
    "delete": aug_delete,
    "insert": aug_insert,
    "replace": aug_replace,
    "duplicate": aug_duplicate,
    "keyboard_typo": aug_keyboard_typo,
    "case_flip": aug_case_flip,
    "vowel_sub": aug_vowel_sub,
    "consonant_sub": aug_consonant_sub,
    "repeat_drop": aug_repeat_drop,
}


def remove_random_adjective(text: str) -> tuple[str, dict[str, Any] | None]:
    try:
        import nltk

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            try:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            except Exception:
                nltk.download("averaged_perceptron_tagger", quiet=True)
        tokens = nltk.word_tokenize(text)
        if len(tokens) < 3:
            return text, None
        tagged = nltk.pos_tag(tokens)
        adj_idx = [i for i, (_, t) in enumerate(tagged) if t in ("JJ", "JJR", "JJS")]
        if not adj_idx:
            return text, None
        i = random.choice(adj_idx)
        word = tokens[i]
        # remove token and following space in reconstruction
        new_tokens = tokens[:i] + tokens[i + 1 :]
        new_text = " ".join(new_tokens)
        new_text = re.sub(r"\s+([.,;:!?])", r"\1", new_text)
        return new_text, {"type": "adj_remove", "word": word}
    except Exception as e:
        logger.debug("adjective removal skipped: %s", e)
        return text, None


class Subnet32Augmentor:
    def __init__(self, aug_cfg: dict[str, Any], stress: bool = False):
        self.cfg = aug_cfg
        self.stress = stress
        self.char_types = list(aug_cfg.get("char_augs") or CHAR_AUG_FUNCS.keys())

    def _num_augmentations(self) -> int:
        if self.stress:
            r = random.random()
            n0 = float(self.cfg.get("stress_augment_prob_none", 0.1))
            n1 = float(self.cfg.get("stress_augment_prob_one", 0.4))
            if r < n0:
                return 0
            if r < n0 + n1:
                return 1
            return 2
        r = random.random()
        n0 = float(self.cfg.get("augment_prob_none", 0.2))
        n1 = float(self.cfg.get("augment_prob_one", 0.5))
        if r < n0:
            return 0
        if r < n0 + n1:
            return 1
        return 2

    def augment(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        text = text.strip()
        if not text:
            return text, []
        n = self._num_augmentations()
        meta: list[dict[str, Any]] = []
        out = text
        for _ in range(n):
            if random.random() < 0.75:
                kind = random.choice(self.char_types)
                fn = CHAR_AUG_FUNCS.get(kind)
                if fn:
                    before = out
                    out = fn(out)
                    if out != before:
                        meta.append({"type": kind})
            else:
                if "adjective_remove" in (self.cfg.get("word_augs") or []):
                    out2, m = remove_random_adjective(out)
                    if m:
                        out = out2
                        meta.append(m)
        return out, meta
