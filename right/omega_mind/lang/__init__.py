"""The language core: every language-specific rule as versioned data.

The engine stays language-agnostic. Segmentation, expansion and stemming are
data tables under ``lang/<language>/``; a persona package names the language and,
where a language has more than one rule set, the profile inside it.

Rules here are pure functions over plain strings so they can be shared by the
retrieval index, the query path and any tool that needs the same segmentation.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Tuple

_DATA_DIR = Path(__file__).resolve().parent

SUPPORTED = ("zh", "en")


class LanguageRuleError(ValueError):
    """A language or profile has no rule file, or its file is unusable."""


class LanguageCore:
    """Segmentation and query expansion for one language profile."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data
        self.language: str = data["language"]
        self.profile: str = data.get("profile", "default")
        self.version: int = int(data.get("version", 1))
        self._ascii: Pattern[str] = re.compile(data["ascii"])
        self._cjk: Optional[Pattern[str]] = (
            re.compile(data["cjk"]) if "cjk" in data else None)
        self._stop_bigrams = frozenset(data.get("stop_bigrams", ()))
        self._unigram_stop = frozenset(data.get("unigram_stop", ""))
        self._stop_words = frozenset(data.get("stop", ()))
        self._stem_rules: Dict[str, Any] = data.get("stem", {})
        self._intents: Tuple[Tuple[Pattern[str], str], ...] = tuple(
            (self._compile(row), row["tokens"]) for row in data.get("intents", ()))
        self._terms: Tuple[Tuple[str, str], ...] = tuple(
            (key, tokens) for key, tokens in data.get("terms", ()))
        self._bridge: Tuple[Tuple[str, str], ...] = tuple(
            (key, tokens) for key, tokens in data.get("bridge", ()))

    @staticmethod
    def _compile(row: Dict[str, Any]) -> Pattern[str]:
        flags = re.I if row.get("ignore_case") else 0
        return re.compile(row["pattern"], flags)

    # -- segmentation -----------------------------------------------------

    def tokens(self, text: str) -> List[str]:
        """Content tokens: the units the index and the query are built from."""
        lowered = text.lower()
        if self.language == "zh":
            return self._tokens_zh(lowered)
        return self._tokens_en(lowered)

    def _tokens_zh(self, lowered: str) -> List[str]:
        out: List[str] = self._ascii.findall(lowered)
        for run in self._cjk.findall(lowered):
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[i:i + 2] for i in range(len(run) - 1))
        return [tok for tok in out if tok not in self._stop_bigrams]

    def _tokens_en(self, lowered: str) -> List[str]:
        out: List[str] = []
        for tok in self._ascii.findall(lowered):
            if tok.isdigit():
                out.append(tok)
                continue
            if len(tok) < 2 or tok in self._stop_words:
                continue
            out.append(self.stem(tok))
        return out

    def unigrams(self, text: str) -> List[str]:
        """Single CJK characters as weak topical evidence (zh only)."""
        if self._cjk is None:
            return []
        out: List[str] = []
        for run in self._cjk.findall(text):
            out.extend(ch for ch in run if ch not in self._unigram_stop)
        return out

    # -- morphology -------------------------------------------------------

    def stem(self, word: str) -> str:
        """Light, deterministic English stemming.

        It normalises derivational pairs that matter in concept answers
        ("deviation"/"deviates", "conservation"/"conserved"), not just plurals
        and tense. Languages without a stem table return the word unchanged.
        """
        rules = self._stem_rules
        if not rules:
            return word
        ies = rules.get("ies_suffix", "ies")
        ied = rules.get("ied_suffix", "ied")
        if word.endswith(ies) and len(word) > 4:
            return word[:-len(ies)] + "y"
        if word.endswith(ied) and len(word) > 4:
            return word[:-len(ied)] + "y"
        remainder = rules.get("min_stem_remainder", 2)
        for _ in range(int(rules.get("passes", 1))):
            for suffix in rules.get("suffixes", ()):
                if len(word) > len(suffix) + remainder and word.endswith(suffix):
                    word = word[: -len(suffix)]
                    break
        min_e = rules.get("trailing_e_min_length", 3)
        if len(word) > min_e - 1 and len(word) > 3 and word.endswith("e"):
            word = word[:-1]
        return word

    # -- query expansion --------------------------------------------------

    def expand(self, text: str) -> str:
        """Add subject vocabulary a question phrase hides."""
        extra = [tokens for pattern, tokens in self._intents if pattern.search(text)]
        extra += [tokens for key, tokens in self._terms if key in text]
        return text + (" " + " ".join(extra) if extra else "")

    def bridge(self, text: str) -> str:
        """Add the corpus's own vocabulary for everyday words."""
        lower = text.lower()
        extra = [tokens for key, tokens in self._bridge if key in lower]
        return text + (" " + " ".join(extra) if extra else "")


def _rule_path(language: str, profile: Optional[str]) -> Path:
    name = "core.json" if profile in (None, "", "default") else f"{profile}.json"
    return _DATA_DIR / language / name


@lru_cache(maxsize=None)
def load(language: str, profile: Optional[str] = None) -> LanguageCore:
    """Load a language rule set; ``profile`` selects a named variant.

    Languages with a single rule set use ``core.json``. A named profile is an
    overlay on it, so a variant only states what it changes.
    """
    if language not in SUPPORTED:
        raise LanguageRuleError(f"unsupported language: {language!r}")
    base_path = _rule_path(language, None)
    try:
        data = json.loads(base_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LanguageRuleError(f"missing language rules: {base_path}") from error
    except json.JSONDecodeError as error:
        raise LanguageRuleError(f"invalid language rules: {base_path}: {error}") from error
    if profile not in (None, "", "default"):
        overlay_path = _rule_path(language, profile)
        try:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise LanguageRuleError(f"missing language profile: {overlay_path}") from error
        except json.JSONDecodeError as error:
            raise LanguageRuleError(f"invalid language profile: {overlay_path}: {error}") from error
        data = _merge(data, overlay)
        data["profile"] = profile
    return LanguageCore(data)


def _merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay one rule file on another; list tables replace, maps merge."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def profiles(language: str) -> List[str]:
    """Rule profiles shipped for a language, ``default`` first."""
    directory = _DATA_DIR / language
    names = sorted(path.stem for path in directory.glob("*.json"))
    return ["default" if name == "core" else name for name in names]
