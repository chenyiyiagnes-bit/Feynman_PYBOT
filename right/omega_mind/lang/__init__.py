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
        self._lexicon = frozenset(self._build_lexicon(data))
        self._lexicon_max = max((len(w) for w in self._lexicon), default=1)

    @staticmethod
    def _compile(row: Dict[str, Any]) -> Pattern[str]:
        flags = re.I if row.get("ignore_case") else 0
        return re.compile(row["pattern"], flags)

    @staticmethod
    def _build_lexicon(data: Dict[str, Any]) -> set:
        """Words the segmenter knows, from the rule file's lexicon and bridge
        table. Expansion ``terms`` are left out on purpose: they carry whole
        question phrases ("费曼是谁") that are useful for retrieval but are not
        words."""
        words = set(data.get("lexicon", ()))
        words.update(key for key, _ in data.get("bridge", ()))
        return {word for word in words if len(word) >= 2}

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

    # -- word segmentation ------------------------------------------------

    def segment(self, text: str) -> List[str]:
        """Cut text into words without an outside dependency.

        This is the one entry point a caller should use when it wants *words*
        rather than index tokens: the left brain's typesetter, the vault
        miner and any tool that has to line up with the index. It is a pure
        function of the rule file, so two callers on the same profile cannot
        disagree, and a missing word table surfaces as a rule error instead of
        a silent fallback.
        """
        text = (text or "").strip()
        if not text:
            return []
        if self._cjk is None:
            return self._tokens_en(text.lower())
        out: List[str] = []
        pos = 0
        length = len(text)
        while pos < length:
            ch = text[pos]
            if self._cjk.fullmatch(ch):
                word = self._longest_word(text, pos)
                if word:
                    out.append(word)
                    pos += len(word)
                else:
                    out.append(ch)
                    pos += 1
                continue
            match = self._ascii.match(text.lower(), pos)
            if match:
                out.append(match.group(0))
                pos = match.end()
                continue
            pos += 1
        return self._join_unknown_runs(out)

    def _join_unknown_runs(self, tokens: List[str]) -> List[str]:
        """Glue dictionary misses back into words.

        Forward maximum matching leaves an unseen name as one character per
        token ("木","块"). A run of characters that are neither particles nor
        lexicon words is most likely a word the table does not have yet, so it
        is kept whole up to six characters; particles (``unigram_stop``) break
        the run, which is what keeps "浮在水面上" from becoming one token.
        Longer runs are cut into two-character pieces so no unbounded string
        reaches a caller.
        """
        out: List[str] = []
        run: List[str] = []

        def flush() -> None:
            if not run:
                return
            text = "".join(run)
            if len(text) <= 6:
                out.append(text)
            else:
                out.extend(text[i:i + 2] for i in range(0, len(text), 2))
            run.clear()

        for token in tokens:
            if (len(token) == 1 and self._cjk is not None
                    and self._cjk.fullmatch(token) and token not in self._unigram_stop):
                run.append(token)
                continue
            flush()
            out.append(token)
        flush()
        return out

    def _longest_word(self, text: str, pos: int) -> str:
        """Longest dictionary word starting at ``pos`` (forward maximum match)."""
        limit = min(self._lexicon_max, len(text) - pos)
        for size in range(limit, 1, -1):
            candidate = text[pos:pos + size]
            if candidate in self._lexicon:
                return candidate
        return ""

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


def load(language: str, profile: Optional[str] = None) -> LanguageCore:
    """Load a language rule set; ``profile`` selects a named variant.

    Languages with a single rule set use ``core.json``. A named profile is an
    overlay on it, so a variant only states what it changes. ``None``,
    ``""`` and ``"default"`` name the same rule set and return the same object.
    """
    if profile in (None, "", "default"):
        profile = None
    return _load(language, profile)


@lru_cache(maxsize=None)
def _load(language: str, profile: Optional[str]) -> LanguageCore:
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
    """Overlay one rule file on another.

    Maps merge; lists (stop tables, lexicon, expansion rows) concatenate, so a
    profile adds vocabulary and can never silently drop what the language
    already knew; strings and scalars replace, because a profile that changes
    the particle list means to change it.
    """
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = {**current, **value}
        elif isinstance(value, list) and isinstance(current, list):
            combined = list(current)
            for item in value:
                if item not in combined:
                    combined.append(item)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


def profiles(language: str) -> List[str]:
    """Rule profiles shipped for a language, ``default`` first."""
    directory = _DATA_DIR / language
    names = sorted(path.stem for path in directory.glob("*.json"))
    return ["default" if name == "core" else name for name in names]
