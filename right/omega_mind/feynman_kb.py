"""Retrieval layer over a persona package's knowledge cards.

One instance holds one package in one language: the cards come from
``personas/<id>/knowledge/*.jsonl`` — lecture cards, teachable units, physics
guardrails, persona cards, biography — and the language (and its rule set)
is the one that package declares in ``persona.json``. This module turns the
cards into a small TF-IDF index, pure standard library.

Rows are addressed by ``id`` and the reply is composed from the row's own
fields — nothing is invented, and a low score returns no answer instead of a
bad one.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import (assoc, dialogue, evidence, lang, ranking, review,
                 scales, spreading)
from .learner import LearnedStore, card_row, normalize
from .paths import learned_dir_for

#: Shipped persona packages, discovered from ``personas/*/knowledge``. The
#: right brain names no persona: a caller that wants a specific mind passes
#: its ``knowledge`` directory (``app/runtime.build()`` does). The discovery
#: only lets a fresh checkout, the CLI and the tests find cards without one.
PACKAGES_ROOT = Path(__file__).resolve().parents[2] / "personas"


def package_language(package_dir: Path) -> Optional[str]:
    """The language a package declares in its own ``persona.json``, if any.

    ``package_dir`` may be the package root or its ``knowledge`` subdirectory.
    A package that says nothing is treated as unclassified; the caller then
    has to name the language.
    """
    root = Path(package_dir)
    if root.name == "knowledge":
        root = root.parent
    try:
        data = json.loads((root / "persona.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    language = data.get("language")
    return language if language in ("zh", "en") else None


def _wants_recall_expansion(package_dir: Path) -> bool:
    """Whether a package opted into the derived term graph.

    ``recall_expansion`` is a persona-level switch, read from the package's
    own ``persona.json``. Unreadable or absent means off: a rebuilt graph is
    never allowed to change answers unless the package asked for it.
    """
    try:
        data = json.loads((Path(package_dir) / "persona.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("recall_expansion"))


def package_profile(package_dir: Path) -> Optional[str]:
    """The segmentation profile a package declares (``default``/``classical``)."""
    try:
        data = json.loads((Path(package_dir) / "persona.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    profile = data.get("language_profile")
    return str(profile) if profile else None


def sibling_package(name: str, language: str,
                    root: Path = PACKAGES_ROOT) -> Optional[Path]:
    """The same family's package in another language (``x-zh`` → ``x-en``).

    Used when a mind without a supplied package changes language: switching
    should keep the character, not jump to whichever package happens to be
    first in the directory listing.
    """
    base = re.sub(r"-(zh|en)$", "", str(name or ""))
    if not base:
        return None
    candidate = Path(root) / f"{base}-{language}"
    if (candidate / "persona.json").is_file():
        return candidate
    return None


def _shipped_knowledge_dirs() -> Tuple[Path, ...]:
    return tuple(sorted(
        path for path in PACKAGES_ROOT.glob("*/knowledge")
        if path.is_dir() and any(path.glob("*.jsonl"))
    ))


#: Every pack the repo ships, one language each. Never a single hardcoded id.
DEFAULT_DIRS = _shipped_knowledge_dirs()

# Files whose rows carry a usable "text" (or equivalent) body. Rows without
# prose (relations, people, manifest) are skipped automatically.
SKIP_FILES = {"MANIFEST.json"}

# Rows sometimes omit "type"; the file name then names the kind. Kinds have
# one spelling across packages: a package that names its files after its own
# section headings still yields the same kind the engine reasons about, so a
# card's behaviour never depends on the file layout of the package it came
# from. Kinds are a property of the card schema, not of any persona.
STEM_KIND = {
    "teachable_units": "teachable_unit",
    "physics_edges": "physics_edge",
    "method_cards": "method_card",
    "feynman_biography": "biography",
    "feynman_lectures": "lecture_card",
    "feynman_life_art": "persona_memory",
    "feynman_math_comp": "lecture_card",
    "feynman_physics": "physics_card",
    "facts": "fact",
    "verified_facts": "fact",
    "persona_memories": "persona_memory",
    "physics_guardrails": "physics_guardrail",
    "thinking_moves": "thinking_move",
    "taste_discriminants": "taste_discriminant",
}

_ASCII = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_CJK = re.compile(r"[一-鿿]+")

# Language rules are data under ``lang/<language>/``. Nothing is loaded at
# import time: each KB instance loads exactly the core its package speaks, and
# the module-level helpers below only exist for callers that work on one
# language at a time (challenge mode, scripts). They load lazily on first use.
#
# The helpers keep the loader that was bound at import time. Tools that swap
# ``lang.load`` to try another segmentation scheme mean to change the KB under
# test, not the baseline the scheme is compared against — going through the
# live name would make the baseline call the candidate, and recurse.
# Question scaffolding that says nothing about the subject. Left in, these
# bigrams let any card that happens to contain "什么" win the query.
_STOP_BIGRAMS = {
    "什么", "么是", "是什", "为什", "怎么", "么样", "有没", "多少", "哪个", "哪些",
    "这个", "那个", "就是", "一个", "我们", "你们", "他们", "可以", "是不",
    "不是", "没有", "一下", "知道", "告诉", "为会", "么会", "子是",
    # Conversational scaffolding around a real question: "能不能聊聊X到底
    # 是怎么回事". Left in, these bigrams dilute the subject tokens and push
    # an on-topic card below the answer gate.
    "聊聊", "回事", "到底", "能不",
    # Magnitude scaffolding: "原子有多大" is a question about atoms and a
    # scale, not about the bigrams 多大/有多.
    "多大", "多小", "多快", "多慢", "多高", "多重", "多远", "多长", "多久",
    "有多", "几倍",
}

_BASELINE_LOAD = lang.load


def _core(language: str, profile: Optional[str] = None):
    """The package core for a live instance; follows ``lang.load`` swaps."""
    return lang.load(language, profile)


def _baseline_core(language: str):
    return _BASELINE_LOAD(language)


def _stem_en(word: str) -> str:
    """Light, deterministic English stemming for retrieval and grading."""
    return _baseline_core("en").stem(word)


def _tokens_en(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed, light stemming."""
    return _baseline_core("en").tokens(text)


def _expand_en(text: str) -> str:
    return _baseline_core("en").expand(text)


def _expand_zh(text: str) -> str:
    return _baseline_core("zh").expand(text)


def _bridge_zh(text: str) -> str:
    return _baseline_core("zh").bridge(text)


def _unigrams(text: str) -> List[str]:
    """Single CJK characters as weak topical evidence.

    Bigrams carry the subject signal; unigrams only widen recall for short
    or everyday phrasings. They are weighted low enough that they can never
    pass the precision gate on their own.
    """
    return _baseline_core("zh").unigrams(text)


def _tokens(text: str) -> List[str]:
    return _baseline_core("zh").tokens(text)


# Questions about the speaker's own memory are not questions about
# Feynman's biography. The card layer stays out of them instead of matching
# "remembered" in an obituary sentence.
_EN_EPISODIC = re.compile(
    r"\b(?:why\s+(?:can'?t|cannot|can\s+not|don'?t)\s+(?:i\s+)?remember"
    r"|(?:do|can|could)\s+you\s+remember"
    r"|what\s+did\s+(?:we|i|you)\s+(?:talk|say|discuss)"
    r"|(?:my|your)\s+memory"
    r"|remember\s+(?:yesterday|last\s+time|our\s+conversation))\b",
    re.I,
)
_EN_MEMORY_QUERY = re.compile(r"\bmemory\b", re.I)

# Requests for entertainment belong to the chat layer (PyBot), not to a card
# about Feynman's family joking at dinner. Without this guard "tell me a joke"
# retrieves the dinner-table memory.
_EN_SMALLTALK = re.compile(
    r"\b(?:tell\s+(?:me\s+)?(?:a\s+)?joke|make\s+me\s+laugh"
    r"|tell\s+(?:me\s+)?(?:a\s+)?story|sing\s+(?:me\s+)?(?:a\s+)?song"
    r"|what'?s\s+up)\b",
    re.I,
)

# The same problem in Chinese: "我昨天吃了什么" is a question about the
# speaker's own episode. It must not be answered from a science memory card.
_ZH_EPISODIC = re.compile(
    r"(?:我|我们|你|咱们)(?:昨天|前天|上次|上回|刚才|刚刚|今早|今天早上)"
    r".{0,8}(?:吃|喝|干|做|聊|说|去|见)(?:了|过)?(?:什么|啥|谁|哪儿|哪里)"
)

# "How big is it?" questions: only these invite a stored magnitude anchor.
# An ordinary "what is X" keeps its answer uncluttered.
_SCALE_ZH = re.compile(
    r"多大|多小|多快|多慢|多高|多重|多远|多长|多久|多少|几倍|数量级|量级|尺度")
_SCALE_EN = re.compile(
    r"how (?:big|large|small|fast|slow|far|heavy|hot|cold|long)|"
    r"what(?:'s| is) the (?:size|scale|mass|speed)|order of magnitude",
    re.IGNORECASE)

# "Why does ice float?" is a physics question, not a request for an anecdote
# from the biography. A biography/persona card may only answer a why-question
# when the question is actually about Feynman.
_EN_THIRD_PERSON_WHY = re.compile(
    r"^\s*why\s+(?:does|do|is|are|did|can|would)\b", re.I)


def _join(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_join(v) for v in value)
    return ""


@dataclass(frozen=True)
class Lookup:
    """Result of one retrieval: text plus the evidence behind it."""
    text: Optional[str]
    source: str
    score: float = 0.0
    margin: Optional[float] = None   # top1 - top2, only when 2+ candidates
    considered: int = 0
    # Query tokens the answering card actually matched. An audit field: it
    # makes a wrong answer diagnosable without re-running the search.
    matched: Tuple[str, ...] = ()
    # Top three (card id, score) pairs, winner included. A miss keeps its
    # near-miss here instead of throwing it away, so a later audit can see
    # which card the question almost reached.
    alternatives: Tuple[Tuple[str, float], ...] = ()
    # One opt-in remark connecting the answering card to a hand-confirmed
    # neighbour (see assoc.py). It is never part of the answer and never
    # changes the source.
    association: Optional[str] = None
    association_edge: Optional[dict] = None
    diagnostics: dict = field(default_factory=dict)
    # A characteristic magnitude on the answering card, spoken only when the
    # question asks for a size / speed / order of magnitude.
    scale: Optional[str] = None
    # One question the right brain may ask after answering, built only from
    # real data: a genuine near-tie between two cards, or the existence of a
    # confirmed premise edge below the answered card.
    question: Optional[str] = None
    # Cards this lookup deliberately demoted because they were just spoken
    # (retrieval inhibition). An audit field: the answer was chosen, not
    # merely found.
    inhibited: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "score": round(self.score, 4),
            "margin": None if self.margin is None else round(self.margin, 4),
            "considered": self.considered,
            "matched": list(self.matched),
            "alternatives": [[card_id, round(score, 4)]
                             for card_id, score in self.alternatives],
            "association": self.association,
            "association_edge": self.association_edge,
            **self.diagnostics,
            "scale": self.scale,
            "question": self.question,
            "inhibited": list(self.inhibited),
        }


# A near-tie this close is not a confident answer: two cards are within a
# few hundredths of each other. The hedge (engine) and the clarifying
# question (below) share the threshold so they cannot disagree.
QUESTION_MARGIN = 0.06

# Retrieval inhibition: a card just spoken is demoted, not banned. Small on
# purpose — it should break near-ties, never outvote clear evidence.
INHIBIT_FACTOR = 0.94


def _question_for(found: "Lookup", titles: Dict[str, str], language: str,
                  premise_ok: bool) -> Optional[str]:
    """The one question the right brain may ask, or None.

    It names only real things: the two near-tied cards' own titles, or the
    fact that a confirmed premise edge sits below the answered card. A
    missing title is never papered over with a card id.
    """
    if not found.text or ":" not in found.source:
        return None
    card_id = found.source.split(":", 1)[1]
    if (found.margin is not None and found.margin <= QUESTION_MARGIN
            and len(found.alternatives) >= 2):
        top = titles.get(card_id)
        alt = titles.get(found.alternatives[1][0])
        if top and alt:
            if language == "en":
                if _CJK.search(top) or _CJK.search(alt):
                    return None
                return (f"I'm not sure which you meant — “{top}”, "
                        f"or “{alt}”?")
            return f"我拿不太准：你想问的是「{top}」，还是「{alt}」那边？"
        return None
    if premise_ok:
        if language == "en":
            return ("There is another layer under this one; ask “why” "
                    "to go down.")
        return "这一层底下还有一层；想往下走，就再问一句“为什么”。"
    return None


@dataclass
class Doc:
    id: str
    kind: str
    title: str
    answer: str
    weights: Dict[str, float] = field(default_factory=dict)
    row: Optional[dict] = None
    language: str = "zh"

    # Challenge mode needs the structured fields of a teachable unit.
    unit: Optional[dict] = None

    def answer_for(self, detail: str = "normal") -> str:
        if self.row is None:
            return self.answer
        text = _answer_for(self.kind, self.row, language=self.language,
                           detail=detail)
        if self.kind == "persona_memory":
            text = _strip_persona_note(text)
        return text or self.answer


# Persona cards sometimes append a note telling the assistant when to recall
# them. It is useful metadata, but it is not part of the answer and reads as a
# leaked instruction once the card is spoken aloud.
_PERSONA_NOTE = re.compile(
    r"(?:这个记忆应在[^。]*。|在吃饭和闲聊时不要[^。]*。"
    r"|This memory should be triggered[^.]*\.|"
    r"[Dd]uring meals and casual conversation,[^.]*\.)"
)


def _strip_persona_note(text: str) -> str:
    return _PERSONA_NOTE.sub("", text).strip()


def _answer_for(kind: str, row: dict, language: str = "zh",
                detail: str = "normal") -> str:
    if detail not in ("brief", "normal", "rich"):
        detail = "normal"
    rich = detail == "rich"
    if language == "en":
        if kind == "teachable_unit":
            parts = [row.get("force_invariant", "")]
            if detail != "brief" and row.get("invariant"):
                parts.append(f"The invariant that pins it down: {row['invariant']}")
            if rich and row.get("check_yourself"):
                parts.append(f"Check yourself: {row['check_yourself']}")
            if rich and row.get("boundary"):
                parts.append(f"Boundary: {row['boundary']}")
            return "\n\n".join(p for p in parts if p)
        if kind == "physics_edge":
            parts = [row.get("phenomenon", "")]
            if detail != "brief" and row.get("invariant"):
                parts.append(f"Invariant: {row['invariant']}")
            if detail != "brief" and row.get("operator"):
                parts.append(f"Operator: {row['operator']}")
            transfer = row.get("transfer")
            if rich and transfer:
                parts.append("The same shape shows up in: " + "; ".join(transfer[:3]))
            return "\n\n".join(p for p in parts if p)
        if kind == "method_card":
            # Method cards have no single body field; their reply is the
            # pattern, one use, and (when the mind has room) the guardrail.
            parts = [row.get("pattern", "")]
            if detail != "brief":
                parts.append(row.get("novel_use", ""))
            if rich:
                parts.append(row.get("guardrail", ""))
            return "\n\n".join(p for p in parts if p)
        return row.get("text", "") or row.get("move", "")
    if kind == "teachable_unit":
        parts = [row.get("force_invariant", "")]
        if detail != "brief" and row.get("invariant"):
            parts.append(f"卡死的那条不变量：{row['invariant']}")
        if rich and row.get("check_yourself"):
            parts.append(f"自己验一下：{row['check_yourself']}")
        if rich and row.get("boundary"):
            parts.append(f"边界：{row['boundary']}")
        return "\n\n".join(p for p in parts if p)
    if kind == "method_card":
        parts = [row.get("pattern", "")]
        if detail != "brief":
            parts.append(row.get("novel_use", ""))
        if rich:
            parts.append(row.get("guardrail", ""))
        return "\n\n".join(p for p in parts if p)
    if kind == "physics_edge":
        parts = [row.get("phenomenon", "")]
        if detail != "brief" and row.get("invariant"):
            parts.append(f"不变量：{row['invariant']}")
        if detail != "brief" and row.get("operator"):
            parts.append(f"式子：{row['operator']}")
        transfer = row.get("transfer")
        if rich and transfer:
            parts.append("同样的形状也出现在：" + "；".join(transfer[:3]))
        return "\n\n".join(p for p in parts if p)
    return row.get("text", "") or row.get("move", "")


RECALL_TOP = 3          # neighbours offered per question token
RECALL_WEIGHT = 0.35    # share of a real query token's weight a neighbour gets


class FeynmanKB:
    # A query must clear this normalised score to be answered from the cards.
    MIN_SCORE = 0.30
    # Below the confident gate a mind may still answer, but only when the
    # best card is clearly ahead of the runner-up. This is the "curiosity
    # loosens the gate" branch; it never applies to a near-tie.
    SOFT_SCORE = 0.24
    SOFT_MARGIN = 0.02

    def __init__(self, knowledge_dir: Optional[str] = None, autoload: bool = True,
                 learned_dir: Optional[str] = None,
                 assoc_path: Optional[str] = None,
                 language: Optional[str] = None,
                 profile: Optional[str] = None,
                 lexicon_dir: Optional[str] = None,
                 source_prefix: Optional[str] = None,
                 spreading: Optional[bool] = None):
        """One instance = one package = one language.

        ``knowledge_dir`` (or ``OMEGA_KB_PATH``) names the package's card
        directory; nothing here assumes a persona id. ``language`` selects the
        rule set and the language of the index. When it is omitted and a
        package is found, the package's own ``persona.json`` decides; a KB
        with no package at all falls back to Chinese so a learned-only store
        still segments its cards.
        """
        if language is not None and language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        self.language = language or "zh"
        self._language_explicit = language is not None
        self.docs: List[Doc] = []
        self.by_id: Dict[str, Doc] = {}
        self._bm25 = None
        self._review_stamp = None
        self._reviewed_ids = set()
        self._review_aliases = set()
        self._corpus_weights = {}
        self.idf: Dict[str, float] = {}
        # Off unless asked for: OMEGA_SPREADING=1 or an explicit argument.
        # The bonus is small and gated, but it still needs golden-net proof
        # before it may become the default.
        self.spreading = (os.environ.get("OMEGA_SPREADING") == "1"
                          if spreading is None else bool(spreading))
        self._spread: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self._assoc: Optional[assoc.AssocGraph] = None
        self._scales: Optional[scales.ScaleBook] = None
        self.dir: Optional[Path] = None
        self.learned_dir: Optional[Path] = Path(learned_dir) if learned_dir else None
        # Judgement about this mind's own cards belongs to the persona, not to
        # the engine: two personas may share the engine and never borrow each
        # other's associations. None keeps the engine default (single-persona
        # CLI use).
        self.assoc_path: Optional[Path] = Path(assoc_path) if assoc_path else None
        # Which rule set inside the language reads this package's text. The
        # 老庄 package reads classical Chinese; everything else is modern.
        self.profile = profile or "default"
        self._core_cache = None
        if profile:
            # An unknown profile is a caller error, and the caller deserves to
            # hear about it here, not on the first query three layers down.
            self._validate_profile()
        # Optional derived term graph (tools/build_lexicon.py). Off unless a
        # caller names the directory: the file is rebuilt data, not shipped
        # source, so its absence must not change any answer.
        self.lexicon_dir: Optional[Path] = Path(lexicon_dir) if lexicon_dir else None
        self._edge_cache: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
        self.managed_package = source_prefix is not None
        self.source_prefix = source_prefix or "feynman_kb"
        self.learned: Optional[LearnedStore] = None
        self.aliases: Dict[str, List[str]] = {}
        self._source_rows: List[Tuple[str, dict]] = []
        self.mode = "unloaded"
        if autoload:
            path = self._find_dir(knowledge_dir, self.language)
            if path is not None:
                self.load(path)
        # A learned overlay can exist with no corpus at all: the user may teach
        # the mind before any cards are installed.
        if self.learned_dir is not None and self.learned is None:
            self._load_learned()

    @property
    def core(self):
        """This package's language rule set; loaded once, on first use."""
        if self._core_cache is None:
            self._core_cache = lang.load(self.language, self.profile)
        return self._core_cache

    @property
    def source_root(self) -> str:
        """The prefix every answer source from this package starts with.

        The package names its own source: an instance holds one package in one
        language, so ``feynman-zh:flp-i-01`` needs no separate language field.
        Callers that mix packages pass a distinct ``source_prefix``.
        """
        return self.source_prefix

    @property
    def core_zh(self):
        """The Chinese rule set, for callers that name the language explicitly.

        A KB that holds another package's language has no Chinese core to
        hand out; ask ``core`` for the rules this package actually reads.
        """
        if self.language != "zh":
            raise ValueError(
                f"this mind holds the {self.language!r} package; "
                f"its language core is {self.language!r}, not 'zh'")
        return self.core

    def _validate_profile(self) -> None:
        """Refuse a profile the package's language does not ship."""
        available = lang.profiles(self.language)
        if self.profile not in available:
            raise lang.LanguageRuleError(
                f"unknown language profile: {self.profile!r}; "
                f"{self.language!r} ships {available}")

    def _set_language(self, language: str) -> None:
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        if language != self.language:
            self.language = language
            self._core_cache = None
            self._validate_profile()

    def _adopt_package_language(self, path: Path) -> None:
        """Take the language from the package's persona.json unless pinned."""
        if self._language_explicit:
            return
        declared = package_language(path.parent)
        if declared is not None:
            self._set_language(declared)

    def _find_dir(self, explicit: Optional[str],
                  language: Optional[str]) -> Optional[Path]:
        """Caller path, then env, then a shipped package in that language.

        ``language=None`` means "no preference": the first shipped package is
        used, unclassified or not.
        """
        env = os.environ.get("OMEGA_KB_PATH")
        for candidate in (explicit, env):
            if not candidate:
                continue
            path = Path(candidate)
            if path.is_dir() and any(path.glob("*.jsonl")):
                return path
        for path in DEFAULT_DIRS:
            if language is None or package_language(path.parent) == language:
                return path
        return None

    @classmethod
    def disabled(cls) -> "FeynmanKB":
        kb = cls(autoload=False)
        kb.mode = "disabled"
        return kb

    @classmethod
    def for_package(cls, name: str, root: Path | str = PACKAGES_ROOT,
                    language: Optional[str] = None,
                    profile: Optional[str] = None,
                    learned_dir: Optional[str] = None) -> Optional["FeynmanKB"]:
        """A KB for one named package, or None when it is not installed.

        This is the only way to build a package's KB outside ``app.runtime``:
        everything a package declares (language, segmentation profile, term
        graph switch) comes from its own ``persona.json``, and learned cards
        go to the shared state directory — never into the package, which is
        what keeps the CLI and the app teaching the same mind.
        """
        package = Path(root) / str(name)
        if not (package / "persona.json").is_file():
            return None
        knowledge = package / "knowledge"
        if not knowledge.is_dir():
            return None
        language = language or package_language(package) or "zh"
        profile = profile or package_profile(package)
        # Recall expansion is the package's choice, same as in app.runtime:
        # a machine that once built the term graph must not answer differently
        # from a fresh clone unless the package asked for the graph.
        lexicon = (str(package / "data" / "lexicon")
                   if _wants_recall_expansion(package) else None)
        return cls(
            knowledge_dir=str(knowledge),
            learned_dir=str(learned_dir or learned_dir_for(package.name)),
            assoc_path=str(package / "data" / "assoc" / "assoc_edges.jsonl"),
            lexicon_dir=lexicon,
            language=language,
            profile=profile,
            source_prefix=package.name,
        )

    @classmethod
    def for_language(cls, language: str,
                     profile: Optional[str] = None) -> Optional["FeynmanKB"]:

        """A KB for the installed package of ``language``, or None.

        Used when a mind changes language at runtime: the new language needs
        its own package, its own cards and its own derived state. Nothing from
        the old package carries over, which is why this builds a fresh
        instance instead of reloading one.
        """
        if language not in ("zh", "en"):
            raise ValueError("language must be 'zh' or 'en'")
        path = cls(autoload=False)._find_dir(None, language)
        if path is None:
            return None
        return cls.for_package(path.parent.name, root=path.parent.parent,
                               language=language, profile=profile)

    def available(self) -> bool:
        return bool(self.docs)

    def _rebuild_idf(self) -> None:
        self.idf = self._build_idf(self.docs)
        self.by_id = {doc.id: doc for doc in self.docs}
        self._bm25 = None

    def topic_for(self, text: str, language: Optional[str] = None) -> str:
        """Coarse topic of a user utterance: the kind of the nearest card.

        Deliberately not a tag — tags are fine-grained and half of them are
        proper nouns. The card kind is stable, visible in the corpus, and
        enough to protect whole areas of memory from a single busy subject.
        Learned cards are skipped: a new metaphor should join the topic of the
        subject it explains, not the topic "learned_card".
        """
        hits = self.search(text, language=language, limit=5)
        for _score, doc in hits:
            if not doc.id.startswith("learned-"):
                return doc.kind
        return "general"

    def load(self, path: Path) -> None:
        """Load one package. The language was fixed before the first card."""
        self._adopt_package_language(path)
        if self.language == "en":
            self._load_en_package(path)
            return
        docs: List[Doc] = []
        source_rows: List[Tuple[str, dict]] = []
        for file in sorted(path.glob("*.jsonl")):
            if file.name in SKIP_FILES:
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_rows.append((file.stem, row))
                doc = self._row_to_doc(file.stem, row)
                if doc is not None:
                    docs.append(doc)
        self.docs = docs
        self._corpus_weights = {doc.id: dict(doc.weights) for doc in docs}
        self._source_rows = source_rows
        self.dir = path
        self._rebuild_idf()
        self._spread = None
        self._load_learned()
        self.mode = f"{self.source_prefix}:{len(docs)}"
        if len(self.docs) > len(docs):
            self.mode += f"+learned:{len(self.docs) - len(docs)}"

    def _load_en_package(self, path: Path) -> None:
        """Load a package whose cards are already English.

        One directory = one language. Rows that still carry Chinese are
        dropped, not translated: an English package that leaks Chinese would
        break the monolingual contract the engine enforces on replies.
        """
        docs: List[Doc] = []
        dropped = 0
        for file in sorted(path.glob("*.jsonl")):
            if file.name in SKIP_FILES:
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not self._already_english(
                        row, STEM_KIND.get(file.stem, file.stem)):
                    dropped += 1
                    continue
                doc = self._row_to_doc(file.stem, row, language="en")
                if doc is not None:
                    docs.append(doc)
        self.docs = docs
        self._corpus_weights = {doc.id: dict(doc.weights) for doc in docs}
        self._rebuild_idf()
        self.dir = path
        self._spread = None
        self.dropped_rows = dropped
        self._load_learned()
        self.mode = f"{self.source_prefix}:en:{len(docs)}"
        if dropped:
            self.mode += f"-drop:{dropped}"
        learned = len(self.docs) - len(docs)
        if learned:
            self.mode += f"+learned:{learned}"

    # -- learned overlay ---------------------------------------------------

    def _learned_store(self) -> Optional[LearnedStore]:
        """The learned overlay that belongs to this KB's own package.

        Derived state is per package: what one mind was taught must never
        appear in another package's index. An explicit ``learned_dir`` wins;
        otherwise the store sits under the package that owns the cards. A KB
        with neither has no overlay at all rather than a global one.
        """
        directory = self.learned_dir
        if directory is None and self.dir is not None:
            directory = self.dir.parent / "data" / "kb_learned"
        if directory is None:
            return None
        return LearnedStore(directory)

    def uses_review(self) -> bool:
        default = "1" if self.managed_package else "0"
        return os.getenv("OMEGA_REVIEW_LEARNING", default) != "0"

    def _load_learned(self) -> None:
        """Append user-taught cards and aliases on top of the read-only corpus.

        Anything retrieved from here is still a normal row: same fields, same
        scoring, same answer builder. Removing ``data/kb_learned`` restores
        the exact pre-learning behaviour.
        """
        store = self._learned_store()
        self.learned = store
        if store is None:
            return
        docs = self.docs
        known = {doc.id for doc in docs}
        added = 0
        reviewed = self._review_decisions(store)
        for row in [*store.cards, *(d["row"] for d in reviewed if "row" in d)]:
            if row.get("id") in known:
                continue
            doc = self._row_to_doc("learned_cards", row,
                                   language=self.language)
            if doc is None:
                continue
            docs.append(doc)
            known.add(doc.id)
            added += 1
        if added:
            self._rebuild_idf()
        if not self.uses_review():
            for card_id, phrase in store.alias_items():
                self._apply_alias(card_id, phrase)
        self._review_aliases = {tuple(d["alias"]) for d in reviewed if "alias" in d}
        for card_id, phrase in sorted(self._review_aliases):
            self._apply_alias(card_id, phrase)

    def _apply_alias(self, card_id: str, phrase: str) -> None:
        docs = self.docs
        doc = next((d for d in docs if d.id == card_id), None)
        if doc is None:
            return
        phrases = self.aliases.setdefault(card_id, [])
        if phrase not in phrases:
            phrases.append(phrase)
        n = max(1, len(docs))
        idf = self.idf
        for tok in self.core.tokens(phrase):
            if doc.weights.get(tok, 0.0) < 3.0:
                doc.weights[tok] = 3.0
            if tok not in idf:
                idf[tok] = math.log(1.0 + n)

    def learn_alias(self, card_id: str, phrase: str,
                    persist: bool = True) -> bool:
        """Attach a phrasing to a card after the mind was praised for it."""
        if not phrase or not any(d.id == card_id for d in self.docs):
            return False
        self._apply_alias(card_id, phrase)
        if persist and self.learned is not None:
            self.learned.add_alias(card_id, phrase)
        return True

    def learn_card(self, text: str, origin: str = "conversation",
                   persist: bool = True, topic: str = "general") -> bool:
        """Turn a metaphor the user taught into a retrievable card.

        The gate lives in ``learner.card_row``: only comparisons become cards.
        A card that closely resembles an existing one becomes a revision, so
        the new wording replaces the old one without losing history. The new
        card joins ``self.docs`` immediately for later turns; it is not
        retroactively used to answer the turn that taught it.

        ``topic`` is coarser than a tag: it is the kind of corpus card the
        comparison is closest to, so a burst of physics metaphors cannot
        crowd art out of memory at the card cap.
        """
        row = card_row(text, origin=origin, topic=topic)
        if row is None:
            return False
        if persist and self.learned is not None:
            changed = self.learned.add_card(row)
            if not changed:
                return False
            row = next((c for c in self.learned.cards
                        if c.get("id") == row["id"]), row)
            # A revision may have merged into a *different* id.
            row = self._merged_row(row, text)
        doc = self._row_to_doc("learned_cards", row, language=self.language)
        if doc is None:
            return False
        docs = self.docs
        for i, existing in enumerate(docs):
            if existing.id == doc.id:
                docs[i] = doc
                break
        else:
            docs.append(doc)
        self._drop_archived_docs()
        self._rebuild_idf()
        return True

    def _merged_row(self, row: Dict[str, Any], text: str) -> Dict[str, Any]:
        """After a merge the new text lives on whichever card it resembled."""
        body = text.strip()
        for card in (self.learned.cards if self.learned else []):
            if card.get("body") == body or card.get("text") == row.get("text"):
                return card
        return row

    def _drop_archived_docs(self) -> None:
        if self.learned is None:
            return
        active = {c.get("id") for c in self.learned.cards}
        self.docs = [d for d in self.docs
                     if not d.id.startswith("learned-") or d.id in active]

    def reload_learned(self) -> None:
        """Rebuild the learned overlay after a batch change (sleep merges).

        Corpus docs are untouched; learned docs are re-derived from the store
        and every learned alias is re-applied. Aliases added in memory only
        (``persist=False``) are kept as they are.
        """
        if self.learned is None:
            return
        keep = self.unreviewed_aliases()
        self.docs = [doc for doc in self.docs if doc.id in self._corpus_weights]
        for doc in self.docs:
            doc.weights = dict(self._corpus_weights[doc.id])
        self.aliases = {}
        self._rebuild_idf()
        self._load_learned()
        for card_id, phrases in keep.items():
            for phrase in phrases:
                self._apply_alias(card_id, phrase)
        self._spread = None

    def unreviewed_aliases(self) -> dict:
        return {cid: [phrase for phrase in phrases if (cid, phrase) not in self._review_aliases]
                for cid, phrases in self.aliases.items()}

    def note_hit(self, card_id: str) -> None:
        if self.learned is not None and card_id.startswith("learned-"):
            self.learned.record_hit(card_id)

    def note_praise(self, card_id: str) -> None:
        if self.learned is not None and card_id.startswith("learned-"):
            self.learned.record_praise(card_id)

    @staticmethod
    def _already_english(row: dict, kind: str = "") -> bool:
        body = " ".join(str(row.get(k) or "") for k in ("text", "move"))
        if not body.strip():
            # Structured cards keep their prose in per-kind fields
            # (phenomenon, force_invariant, pattern), not in "text".
            body = _answer_for(kind, row, language="en")
        return bool(body.strip()) and not _CJK.search(body)

    def _row_to_doc(self, stem: str, row: dict,
                    language: Optional[str] = None) -> Optional[Doc]:
        language = language or self.language
        kind = str(row.get("type") or STEM_KIND.get(stem, stem)).replace(" ", "_")
        answer = _answer_for(kind, row, language=language)
        if kind == "persona_memory":
            answer = _strip_persona_note(answer)
        if not answer:
            return None
        title = _join(row.get("title") or row.get("name_zh") or "")
        doc = Doc(
            id=str(row.get("id") or f"{stem}:{len(self.docs)}"),
            kind=kind,
            title=title or _join(row.get("word") or ""),
            answer=answer,
            row=row,
            language=language,
            unit=row if kind == "teachable_unit" else None,
        )
        # Title/tag/alias tokens are the card's subject; body tokens are
        # context. The weight ratio is what keeps "原子是什么" off a biography
        # that merely mentions atoms.
        self._add_weight(doc, title, 4.0, language=language)
        self._add_weight(doc, answer, 1.0, language=language)
        # The invariant is the card's concept sentence. In the English
        # overlay, physics_edge rows have no translated title, so without
        # this the concept word ("damping") would sit at body weight and fail
        # the precision gate.
        self._add_weight(doc, _join(row.get("invariant")), 2.5,
                         language=language)
        self._add_weight(doc, _join(row.get("tags")), 3.0, language=language)
        self._add_weight(doc, _join(row.get("aliases")), 3.0, language=language)
        entry_weight = 3.5 if language == "en" else 2.5
        self._add_weight(doc, _join(row.get("entry_question")), entry_weight,
                         language=language)
        self._add_weight(doc, _join(row.get("transfer")), 0.5, language=language)
        boost = row.get("retrieval_boost")
        if isinstance(boost, (int, float)) or (isinstance(boost, str) and boost.isdigit()):
            for tok in list(doc.weights):
                doc.weights[tok] *= 1.0 + min(float(boost), 50.0) / 100.0
        if language != "en":
            # Weak single-character evidence, below the 2.5 precision gate.
            for text, weight in ((title, 1.0), (_join(row.get("tags")), 0.8),
                                 (_join(row.get("aliases")), 1.0),
                                 (answer, 0.25)):
                for tok in self.core.unigrams(text):
                    if doc.weights.get(tok, 0.0) < weight:
                        doc.weights[tok] = weight
        if not doc.weights:
            return None
        return doc

    def _add_weight(self, doc: Doc, text: str, weight: float,
                    language: Optional[str] = None) -> None:
        if not text:
            return
        # One core per instance: a card is tokenised by the rules of the
        # package it came from, never by another language's segmenter.
        for tok in self.core.tokens(text):
            if doc.weights.get(tok, 0.0) < weight:
                doc.weights[tok] = weight

    def _build_idf(self, docs: Optional[List[Doc]] = None) -> Dict[str, float]:
        docs = self.docs if docs is None else docs
        df: Dict[str, int] = {}
        for doc in docs:
            for tok in doc.weights:
                df[tok] = df.get(tok, 0) + 1
        n = max(1, len(docs))
        return {tok: math.log(1.0 + n / d) for tok, d in df.items()}

    def _recall_neighbours(self, query: Iterable[str]) -> Dict[str, float]:
        """Terms this question's terms are attested with, and how strongly.

        Returns nothing when no lexicon was named or its edges file is absent,
        which is the normal case on a fresh clone.
        """
        edges = self._edges()
        if not edges:
            return {}
        out: Dict[str, float] = {}
        for tok in query:
            for neighbour, npmi in edges.get(tok, ())[:RECALL_TOP]:
                if neighbour == tok:
                    continue
                weight = RECALL_WEIGHT * max(0.0, min(1.0, npmi))
                if weight > out.get(neighbour, 0.0):
                    out[neighbour] = weight
        return out

    def _edges(self) -> Dict[str, List[Tuple[str, float]]]:
        if self.lexicon_dir is None:
            return {}
        key = str(self.lexicon_dir)
        if key in self._edge_cache:
            return self._edge_cache[key]
        path = self.lexicon_dir / "edges.jsonl"
        table: Dict[str, List[Tuple[str, float]]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for source, target in ((row["source"], row["target"]),
                                       (row["target"], row["source"])):
                    table.setdefault(source, []).append((target, float(row["npmi"])))
            for source in table:
                table[source].sort(key=lambda pair: (-pair[1], pair[0]))
        self._edge_cache[key] = table
        return table

    def search(self, text: str, limit: int = 3,
               language: Optional[str] = None,
               min_score: Optional[float] = None,
               prefer_kinds: Iterable[str] = (), *, public_only: bool = False) -> List[Tuple[float, Doc]]:
        """Search this package's index.

        A request in another language is a miss, not a translation: the
        package holds one language, and there is no second index to fall
        through to.
        """
        if language is not None and language != self.language:
            return []
        language = self.language
        docs = self.docs
        idf = self.idf
        if public_only:
            docs = [doc for doc in docs if self.public_doc(doc)]
            if len(docs) != len(self.docs):
                idf = self._build_idf(docs)
        if not docs:
            return []
        if language == "en" and (_EN_EPISODIC.search(text)
                                 or _EN_SMALLTALK.search(text)):
            return []
        if language != "en" and _ZH_EPISODIC.search(text):
            return []
        query_text = self.core.query_text(text)
        expanded = self.core.bridge(self.core.expand(query_text))
        query = set(self.core.tokens(expanded))
        if not query:
            return []
        uni_query = set(self.core.unigrams(expanded))
        # Only tokens the corpus actually contains can match a card. Counting
        # out-of-vocabulary query tokens in the denominator punished natural
        # paraphrases ("费曼怎么看待权威") for words no card could ever have.
        qweight = {tok: idf[tok] ** 2 for tok in query if tok in idf}
        # Tokens that appear in more than a quarter of the cards carry no
        # subject information ("energy" in a physics corpus, "father" in a bio).
        idf_cut = math.log(1.0 + len(docs) / max(1.0, 0.25 * len(docs)))
        qweight = {t: w for t, w in qweight.items() if idf.get(t, 0.0) >= idf_cut}
        # Recall expansion from the derived term graph: a question that names
        # one half of a well-attested pair ("超导" -> "库珀对") reaches the card
        # that talks about the other half. Only tokens the corpus holds can be
        # added, at a damped weight so a neighbour can lift a card but never
        # outvote what was actually asked; the precision gate below still
        # requires the card to match a token of the *question*.
        if language != "en":
            for tok, factor in self._recall_neighbours(query).items():
                if tok in qweight or tok not in idf or idf[tok] < idf_cut:
                    continue
                qweight[tok] = factor * idf[tok] ** 2
        # The tokens the question itself supplied; neighbours are not among
        # them, so evidence for a neighbour is not evidence for the question.
        subject = set(qweight) & set(query)
        # Neighbours add evidence; they must not dilute how much of the
        # question a card accounts for, or a lookup that used to reach its card
        # would lose to one that never had a neighbour to offer.
        total = sum(qweight[tok] for tok in subject)
        if not total:
            return []
        wants_cjk = bool(_CJK.search(text))
        third_person_why = (language == "en" and "feynman" not in text.lower()
                            and bool(_EN_THIRD_PERSON_WHY.match(text)))
        prefer = set(prefer_kinds)
        scored: List[Tuple[float, Doc]] = []
        for doc in docs:
            if wants_cjk and not _CJK.search(doc.answer):
                continue  # English-only cards never answer a Chinese question
            if language == "en" and _CJK.search(doc.answer):
                continue  # a leaked Chinese row is not an English answer
            if (third_person_why
                    and doc.kind in ("biography", "persona_memory")):
                continue  # a story about Feynman is not a physics answer
            common = set(qweight) & doc.weights.keys()
            if not common:
                continue
            # Precision gate: the card must match the query on its own
            # subject (title / tags / aliases / entry question, weight 2.5+).
            # One stray shared body bigram is how "天气" used to reach a
            # cosmology card.
            if not any(doc.weights[tok] >= 2.5 for tok in common & subject):
                continue
            # A second shape of the same failure: a question that names
            # several things but touches the card through exactly one token.
            # "为什么彩虹总是七种颜色呢" must not be answered by a van
            # anecdote merely because its tags include 颜色, nor by a damped-
            # oscillator card merely because its title contains the glue
            # bigram 总是. Only a one-subject question (query tokens all on
            # one subject, as in "What is damping?") may be answered on one
            # token. OOV words count here: 彩虹 is exactly the evidence that
            # the shared token is not what was asked about. Two-token
            # questions are exempt because a short compound gets split into
            # overlapping bigrams ("费曼图" -> 费曼 + 曼图).
            if len(common & subject) == 1 and len(query) > 2:
                continue
            # Coverage: how much of the query the card accounts for. A long
            # card is not penalised for being long; a stray bigram is.
            weights = [qweight[tok] / total for tok in common]
            best = max(
                w * min(doc.weights[tok], 4.0) / 4.0
                for w, tok in zip(weights, common)
            )
            coverage = sum(
                w * min(doc.weights[tok], 4.0) / 4.0 for w, tok in zip(weights, common)
            )
            score = 0.7 * best + 0.3 * coverage
            # Two independent title tokens is a much stronger subject signal
            # than one, and it offsets the dilution from cross-word bigrams
            # ("曼是" in "费曼是谁") that no card can match.
            title_hits = sum(1 for tok in common if doc.weights[tok] >= 3.9)
            if title_hits >= 2:
                score = min(1.0, score + 0.06 * (title_hits - 1))
            if doc.kind in prefer:
                # A small, deterministic nudge: the right brain can prefer a
                # kind of card, but it cannot make an irrelevant card win.
                score *= 1.08
            scored.append((score, doc))
        # Weak unigram evidence tops the bigram score up; it can never open
        # the gate on its own because doc unigram weights stay below 2.5.
        if uni_query and scored:
            uni_idf = {t: idf.get(t, 0.0) for t in uni_query if t in idf}
            uni_total = sum(uni_idf.values()) or 1.0
            lifted: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                common_uni = set(uni_idf) & doc.weights.keys()
                if common_uni:
                    cov = sum(uni_idf[t] * min(doc.weights[t], 1.0)
                              for t in common_uni) / uni_total
                    score = min(1.0, score + 0.12 * cov)
                lifted.append((score, doc))
            scored = lifted
        if self.aliases:
            norm_text = normalize(text)
            boosted: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                if doc.id in self.aliases and any(
                        normalize(p) and normalize(p) in norm_text
                        for p in self.aliases[doc.id]):
                    score = min(1.0, score + 0.10)
                boosted.append((score, doc))
            scored = boosted
        if self.spreading and scored:
            index = (self._spread_index(docs) if len(docs) == len(self.docs)
                     else spreading.npmi_neighbors(docs))
            subject = set(qweight)
            spread: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                if spreading.precision_open(subject, doc):
                    score = min(1.0, score + spreading.spread_bonus(
                        subject, doc, index))
                spread.append((score, doc))
            scored = spread
        if language == "en" and _EN_MEMORY_QUERY.search(text):
            # A general memory question may only use a card whose subject is
            # memory; a body mention is not enough.
            scored = [(score, doc) for score, doc in scored
                      if doc.weights.get("memory", 0.0) >= 2.5]
        if min_score is not None:
            scored = [(score, doc) for score, doc in scored
                      if score >= min_score]
        # On an exact score tie, a physics card beats a story about Feynman.
        # "部分子是什么" should get the parton model, not the one-line
        # biography that merely names it; the biography still wins whenever
        # it scores higher or the question is about Feynman himself.
        narrative = ("biography", "persona_memory")
        # A one-word question ("what is polarization") scores 1.0 on every
        # card with that word in its title, so the vacuum-polarization card
        # tied with the optics card and won on file order. On an exact tie the
        # card that lists the question itself as an alias is the one written
        # to answer it. This is a tie-break only: it never lifts a lower score.
        asked = normalize(text)

        def _asked_verbatim(doc: Doc) -> int:
            aliases = doc.row.get("aliases") or ()
            if isinstance(aliases, str):
                aliases = (aliases,)
            return 1 if asked and any(
                isinstance(a, str) and normalize(a) == asked
                for a in aliases) else 0

        scored.sort(key=lambda pair: (pair[0], _asked_verbatim(pair[1]),
                                      0 if pair[1].kind in narrative else 1),
                    reverse=True)
        return scored[:max(1, limit)]

    def _spread_index(self, docs: List["Doc"]
                      ) -> Dict[str, List[Tuple[str, float]]]:
        if self._spread is None:
            self._spread = spreading.npmi_neighbors(docs)
        return self._spread

    def _review_decisions(self, store):
        self._review_stamp = self._review_mtime(store)
        try:
            decisions = review.active(review.read(store))
        except (OSError, ValueError, KeyError, TypeError):
            decisions = []
        self._reviewed_ids = {str(decision["row"]["id"])
                              for decision in decisions if "row" in decision}
        return decisions

    @staticmethod
    def _review_mtime(store):
        if store is not None:
            try:
                return (store.dir / review.FILE).stat().st_mtime_ns
            except OSError:
                pass
        return None

    def refresh_review(self) -> None:
        if self._review_mtime(self.learned) != self._review_stamp:
            self.reload_learned()

    def public_doc(self, doc: Doc) -> bool:
        row = doc.row or {}
        if (row.get("source_class") == "user_claim"
                or row.get("origin") in ("dream_gap", "dream_correction")
                or (doc.kind == "learned_card" and doc.id.startswith("learned-"))):
            return doc.id in self._reviewed_ids
        return True

    def rank_candidates(self, text: str, limit: int, *, public_only: bool = False):
        legacy = self.search(text, limit=limit, min_score=0, public_only=public_only)
        if self._bm25 is None:
            self._bm25 = ranking.BM25(self.docs, self.core)
        terms = self.core.tokens(self.core.bridge(self.core.expand(self.core.query_text(text))))
        allowed = {doc.id for doc in self.docs if self.public_doc(doc)} if public_only else None
        bm25 = self._bm25.rank(terms, limit, allowed)
        return legacy, bm25, ranking.fuse(legacy, bm25, limit)

    def lookup(self, text: str, language: Optional[str] = None,
               min_score: Optional[float] = None, limit: int = 3,
               prefer_kinds: Iterable[str] = (),
               detail: str = "normal",
               associate: bool = False, *, public_only: bool = False,
               mode: str = "legacy", structured: bool = False,
               prefer_examples: bool = False,
               previous_card: Optional[str] = None,
               scale: bool = False,
               curiosity: bool = False,
               inhibit: Iterable[str] = ()) -> Lookup:
        """Retrieve one answer plus the score, runner-up margin and how many
        candidates were considered. Admission always compares the two best
        eligible cards, even when the caller asks for a single candidate.

        ``associate=True`` appends at most one hand-confirmed association
        remark. It is chosen *after* the answer is fixed and never touches
        the source or the score. ``previous_card`` is the card answered on
        the previous turn; if an edge joins it to this answer, the remark
        continues that thought instead of introducing a new card cold.

        ``scale=True`` adds the card's verified magnitude anchor, but only
        when the question itself asks for a size, speed or order of
        magnitude. The association never changes the source or the score; a
        curated scale question may pick its anchored card on purpose.
        """
        if (language is not None and language != self.language) or (
                self.language == "en" and _CJK.search(text)):
            return Lookup(None, f"{self.source_root}:no_match")
        if public_only and evidence.hypothetical_measurement(text, self.language):
            return Lookup(None, f"{self.source_root}:no_match",
                          diagnostics={"decision": "hypothetical_measurement"})
        found = self._lookup(text, language=language, min_score=min_score,
                             limit=limit, prefer_kinds=prefer_kinds,
                             detail=detail, public_only=public_only,
                             inhibit=inhibit)
        if scale:
            # A curated magnitude question goes straight to the card its
            # anchor was written for; ordinary retrieval loses these
            # ("原子有多大") because the noun is too common to discriminate.
            routed = self._scale_route(text, language, detail)
            if routed is not None:
                found = routed
        intent = evidence.intent_for(text, self.language)
        diagnostics = {"intent": intent, "mode": mode}
        ranked = []
        if mode in ("shadow", "rrf"):
            legacy, bm25, fused = self.rank_candidates(text, max(2, limit), public_only=public_only)
            diagnostics["candidates"] = {
                "legacy": [doc.id for _, doc in legacy],
                "bm25": [doc.id for _, doc in bm25], "rrf": fused}
            ranked = [self.by_id[cid] for cid in fused]
            # Fusion may choose between audited candidates that also clear the
            # lexical gate. It cannot promote a BM25-only card into speech.
            eligible = {doc.id: score for score, doc in legacy
                        if score >= (self.MIN_SCORE if min_score is None else min_score)}
            if mode == "rrf" and found.text and ranked:
                for doc in ranked:
                    if (doc.id in eligible and eligible[doc.id] >= found.score - self.SOFT_MARGIN
                            and evidence.compose(doc, intent, detail, prefer_examples)[0]):
                        rivals = [score for score, other in legacy if other.id != doc.id]
                        margin = eligible[doc.id] - max(rivals) if rivals else None
                        found = Lookup(doc.answer_for(detail), f"{self.source_root}:{doc.id}",
                                       eligible[doc.id], margin, found.considered,
                                       self._matched_terms(text, doc), found.alternatives)
                        break
        doc = self.by_id.get(found.source.partition(":")[2]) if found.text else None
        if doc is not None:
            revision = dialogue.card_revision(doc)
            diagnostics.update(card_id=doc.id, revision=revision)
            start = doc.answer_for("rich").find(found.text)
            if start >= 0:
                diagnostics["card_evidence"] = {
                    "span": [start, start + len(found.text)],
                    "provenance": ("reviewed_local" if doc.id in self._reviewed_ids else "card_summary"),
                }
        if structured and doc is not None:
            viable = [self.by_id[cid] for cid, score in found.alternatives
                      if cid in self.by_id and score >= found.score - self.SOFT_MARGIN]
            if evidence.conflict_among([doc, *viable]):
                diagnostics["decision"] = "conflict"
                return replace(found, text=None, source=f"{self.source_root}:no_match",
                               diagnostics=diagnostics)
            if evidence.structured_card(doc):
                plan, reason = evidence.compose(doc, intent, detail, prefer_examples)
                if not plan:
                    diagnostics["decision"] = reason
                    return replace(found, text=None, source=f"{self.source_root}:no_match",
                                   diagnostics=diagnostics)
                diagnostics.pop("card_evidence", None)
                diagnostics["evidence_plan"] = plan
                found = replace(found, text=" ".join(part["text"] for part in plan))
        if not found.text:
            diagnostics.setdefault("decision", "no_evidence")
        if scale and found.text and not found.scale:
            anchor = self._scale_for(text, found, language)
            if anchor:
                found = replace(found, scale=anchor)
        if curiosity and found.text and ":" in found.source:
            card_id = found.source.split(":", 1)[1]
            titles = {doc.id: doc.title for doc in self.docs}
            graph = self._assoc_graph()
            premise_ok = bool(graph and graph.has_premise(card_id))
            question = _question_for(found, titles, language, premise_ok)
            if question:
                found = replace(found, question=question)
        if associate and found.text:
            matched = self._association_for(found, language,
                                            previous_card=previous_card)
            if matched is not None:
                found = replace(found, association=matched.text,
                                association_edge=matched.provenance())
        return replace(found, diagnostics=diagnostics)

    def _scale_route(self, text: str, language: str, detail: str
                     ) -> Optional[Lookup]:
        """Answer a curated magnitude question from the anchored card."""
        pattern = _SCALE_EN if language == "en" else _SCALE_ZH
        if not pattern.search(text or ""):
            return None
        book = self._scale_book()
        hit = book.anchor_for(text, language)
        if hit is None:
            return None
        card_id, ask = hit
        docs = self.docs
        doc = next((d for d in docs if d.id == card_id), None)
        if doc is None:
            return None
        body = doc.answer_for(detail)
        if not body:
            return None
        if language == "en" and _CJK.search(body):
            return None
        anchor = book.line(card_id, language)
        prefix = self.source_root
        # The score is a fixed evidence mark for "the curated phrasing
        # matched", not a TF-IDF value: there is exactly one candidate.
        return Lookup(text=body, source=f"{prefix}:{card_id}", score=1.0,
                      considered=1, matched=(ask,), scale=anchor)

    def _scale_for(self, text: str, found: Lookup, language: str
                   ) -> Optional[str]:
        """The card's magnitude anchor, only for a magnitude question."""
        pattern = _SCALE_EN if language == "en" else _SCALE_ZH
        if not pattern.search(text or ""):
            return None
        if ":" not in found.source:
            return None
        card_id = found.source.split(":", 1)[1]
        if card_id.startswith("learned-") or card_id == "no_match":
            return None
        anchor = self._scale_book().line(card_id, language)
        if not anchor or anchor in (found.text or ""):
            return None
        return anchor

    def follow_premise(self, card_id: str, language: str = "zh",
                       detail: str = "normal",
                       avoid: Iterable[str] = ()) -> Optional[Lookup]:
        """Answer a bare "why?" by walking one confirmed premise hop.

        The body is the premise card's own answer; the sentence in front of
        it is the hand-written link. Nothing is generated here: the link is
        confirmed data and the prose comes from the row. None means the card
        has no confirmed premise, so the caller answers normally instead.
        """
        if (not card_id or card_id.startswith("learned-")
                or card_id == "no_match"):
            return None
        graph = self._assoc_graph()
        if graph is None or not graph.edges:
            return None
        step = graph.premise_step(card_id, language=language, avoid=avoid)
        if step is None:
            return None
        premise_id, link = step
        docs = self.docs
        doc = next((d for d in docs if d.id == premise_id), None)
        if doc is None:
            return None
        body = doc.answer_for(detail)
        if not body:
            return None
        if language == "en" and _CJK.search(body):
            return None
        lead = (f"One step down: {link}. " if language == "en"
                else f"往下追一层：{link}。")
        prefix = self.source_root
        return Lookup(text=lead + body, source=f"{prefix}:{premise_id}")

    def recast(self, card_id: str, language: str = "zh"
               ) -> Optional[Tuple[str, str]]:
        """Say the card a second way, or None.

        Wraps the association graph's confirmed analogy: the mind may recast
        a card only through an edge that already exists, and only when both
        halves (the shared mechanism and where it breaks) are written.
        """
        if (not card_id or card_id.startswith("learned-")
                or card_id == "no_match"):
            return None
        graph = self._assoc_graph()
        if graph is None or not graph.edges:
            return None
        docs = self.docs
        titles = {doc.id: doc.title for doc in docs}
        hit = graph.recast(card_id, language=language, titles=titles)
        if hit is None:
            return None
        line, key = hit
        return line, f"analogy:{key}"

    def rethink(self, card_id: str, language: str = "zh",
                spoken: Iterable[str] = ()) -> Optional[Tuple[str, str]]:
        """Look at the same card from a face that has not been said yet.

        Reflection, not retrieval: the card stays the same, only the angle
        changes. Faces come from confirmed edges (see ``AssocGraph.facets``)
        and ``spoken`` names the ones already used, so a second reflection
        never repeats the first. Returns ``(line, facet key)`` or None when
        the card has nothing new to show — the caller then says the card
        again rather than inventing an angle.
        """
        if (not card_id or card_id.startswith("learned-")
                or card_id == "no_match"):
            return None
        graph = self._assoc_graph()
        if graph is None or not graph.edges:
            return None
        docs = self.docs
        titles = {doc.id: doc.title for doc in docs}
        used = {str(key) for key in spoken}
        for key, line in graph.facets(card_id, language=language,
                                      titles=titles):
            if key in used:
                continue
            return line, key
        return None

    def card_body(self, card_id: str, language: str = "zh",
                  detail: str = "normal") -> Optional[str]:
        """The card's own answer, for saying it again plainly.

        Used when a reflection has no unused face: repeating the card is
        better than manufacturing a new angle. None means the card does not
        exist in this language (or is Chinese-only in English mode), so the
        caller must not speak it.
        """
        if not card_id or card_id == "no_match":
            return None
        docs = self.docs
        doc = next((d for d in docs if d.id == card_id), None)
        if doc is None:
            return None
        body = doc.answer_for(detail)
        if not body:
            return None
        if language == "en" and _CJK.search(body):
            return None
        return body

    def _association_for(self, found: Lookup,
                         language: Optional[str],
                         previous_card: Optional[str] = None
                         ) -> Optional[assoc.Match]:
        prefix = self.source_root + ":"
        if not found.source.startswith(prefix):
            return None
        card_id = found.source[len(prefix):]
        if card_id.startswith("learned-") or card_id == "no_match":
            return None
        graph = self._assoc_graph()
        if graph is None or not graph.edges:
            return None
        language = language or self.language
        titles = {doc.id: doc.title for doc in self.docs}
        # Never explain a card the answer already names: the user asked
        # about it, and the runner-up may be the same subject restated.
        skip = {card_id}
        skip.update(other for other, _score in found.alternatives)
        return graph.select(card_id, language=language, titles=titles,
                            exclude=skip, continuity_with=previous_card)

    def _assoc_graph(self) -> Optional[assoc.AssocGraph]:
        if self._assoc is None:
            path = self.assoc_path
            if path is None and self.dir is not None:
                path = self.dir.parent / "data" / "assoc" / "assoc_edges.jsonl"
            self._assoc = assoc.load(path) if path is not None else assoc.AssocGraph([])
        return self._assoc

    def follow_up(self, card_id: str, kind: str, target: str,
                  spoken: Tuple[str, ...]) -> Optional[str]:
        passage, _ = self.follow_up_evidence(card_id, kind, target, spoken)
        return passage

    def follow_up_evidence(self, card_id: str, kind: str, target: str,
                           spoken: Tuple[str, ...]):
        """Select a passage or an audited relation for a grounded follow-up."""
        card = self.by_id.get(card_id)
        if card is None:
            return None, None
        core = self.core
        cue = {"why": core.explanation, "boundary": core.condition,
               "rephrase": core.example}[kind]
        body = card.answer_for("rich")
        passages = re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+", body)
        passages = [passage.strip() for passage in passages if passage.strip()]
        relevant = None
        if target:
            # For a conditional, short Chinese nouns may be a single
            # character. Use the condition itself, not the generic tail
            # ("那该怎么办"), as weak evidence alongside the bigrams.
            if kind == "boundary":
                target = re.split(r"[，,；;]", target, maxsplit=1)[0]
            terms = set(core.tokens(target))
            chars = set(core.unigrams(target)) if kind == "boundary" else set()

            def overlap(passage: str) -> Tuple[int, int]:
                matched = len(terms.intersection(core.tokens(passage)))
                if matched >= 2 and matched * 2 >= len(terms):
                    return matched, 0
                intro, colon, _ = passage.partition("：")
                if chars and colon and target[0] in intro and core.example.search(intro):
                    # A one-character reference must name the example's
                    # subject (苗), not just share its condition (干、水).
                    shared = len(chars.intersection(core.unigrams(passage)))
                    if shared >= 2:
                        return 0, shared
                return 0, 0

            passages = [passage for passage in passages
                        if any(overlap(passage)) and cue.search(passage)]
            passages.sort(key=overlap, reverse=True)
            if kind == "boundary":
                # An example can introduce a conditional after a colon.
                # Quote the condition itself when the introduction is extra.
                for i, passage in enumerate(passages):
                    _, colon, condition_text = passage.partition("：")
                    if colon and cue.search(condition_text):
                        passages[i] = condition_text.strip()
            selected = next((p for p in passages if p not in spoken), None)
            if selected:
                return selected, None

            def relevant(line: str) -> bool:
                return overlap(line)[0] > 0

        edge_kind = {"why": "premise", "boundary": "boundary",
                     "rephrase": "same_mechanism"}[kind]
        graph = self._assoc_graph()
        titles = {doc.id: doc.title for doc in self.docs}
        match = graph.select(
            card_id, language=self.language, titles=titles,
            kinds=(edge_kind,), spoken=spoken, relevant=relevant)
        if match is not None:
            return match.text, match.provenance()
        if target:
            return None, None
        return next((p for p in passages if cue.search(p) and p not in spoken
                     and (kind != "rephrase" or not any(p in s for s in spoken))), None), None

    def _scale_book(self) -> scales.ScaleBook:
        if self._scales is None:
            self._scales = scales.load()
        return self._scales

    def _lookup(self, text: str, language: Optional[str] = None,
                min_score: Optional[float] = None, limit: int = 3,
                prefer_kinds: Iterable[str] = (),
                detail: str = "normal", public_only: bool = False,
                inhibit: Iterable[str] = ()) -> Lookup:
        # A package names its own source; there is no cross-package index to
        # borrow a prefix from.
        prefix = self.source_root
        threshold = self.MIN_SCORE if min_score is None else min_score
        # Search's own min_score would drop the soft band before lookup can
        # consider it. An explicit gate above the default is a deliberate
        # withholding ("be careful / say nothing"), so it stays strict.
        soft_ok = min_score is None or min_score <= self.MIN_SCORE
        floor = min(threshold, self.SOFT_SCORE) if soft_ok else threshold
        hits = self.search(text, language=language, limit=max(2, limit),
                           min_score=floor, prefer_kinds=prefer_kinds,
                           public_only=public_only)
        # Retrieval inhibition: a card spoken in the last few turns is less
        # likely to be selected again, so a near-tie resolves to the other
        # card instead of circling. This never turns a confident answer into
        # a miss: the demotion only applies when the demoted top still clears
        # the gate, so the worst case is the runner-up taking over — exactly
        # what the phenomenon says should happen.
        blocked = {str(card_id) for card_id in inhibit}
        inhibited: Tuple[str, ...] = ()
        if blocked and len(hits) > 1 and hits[0][0] * INHIBIT_FACTOR >= threshold:
            demoted = [(score * INHIBIT_FACTOR if doc.id in blocked else score,
                        doc) for score, doc in hits]
            demoted.sort(key=lambda pair: pair[0], reverse=True)
            inhibited = tuple(doc.id for _score, doc in hits if doc.id in blocked)
            hits = demoted
        alternatives = tuple((doc.id, score) for score, doc in hits[:3])
        if not hits or hits[0][0] < threshold:
            if hits:
                score, doc = hits[0]
                margin = score - hits[1][0] if len(hits) > 1 else None
                if (score >= self.SOFT_SCORE
                        and (margin is None or margin >= self.SOFT_MARGIN)):
                    return Lookup(
                        text=doc.answer_for(detail),
                        source=f"{prefix}_soft:{doc.id}",
                        score=score,
                        margin=margin,
                        considered=len(hits),
                        matched=self._matched_terms(text, doc, language),
                        alternatives=alternatives,
                        inhibited=inhibited,
                    )
                # A user-taught restatement must not veto the corpus card it
                # restates: a near-tie across the learned/corpus boundary is
                # wording overlap, not subject doubt. Two cards of the same
                # class tied is still a near-tie, and still refused.
                tie = self._class_tie(hits, prefix, detail, text, language,
                                      alternatives=alternatives)
                if tie is not None:
                    return tie
            return Lookup(None, f"{prefix}:no_match", considered=len(hits),
                          alternatives=alternatives, inhibited=inhibited)
        score, doc = hits[0]
        margin = score - hits[1][0] if len(hits) > 1 else None
        return Lookup(
            text=doc.answer_for(detail),
            source=f"{prefix}:{doc.id}",
            score=score,
            margin=margin,
            considered=len(hits),
            matched=self._matched_terms(text, doc, language),
            alternatives=alternatives,
            inhibited=inhibited,
        )

    def _matched_terms(self, text: str, doc: Doc,
                       language: Optional[str] = None) -> Tuple[str, ...]:
        """Query tokens that hit this card — the evidence behind the answer."""
        core = self.core
        tokens = set(core.tokens(core.bridge(core.expand(core.query_text(text)))))
        shared = sorted(tok for tok in tokens if tok in doc.weights)
        return tuple(shared[:10])

    def _class_tie(self, hits: List[Tuple[float, Doc]], prefix: str,
                   detail: str, text: str = "",
                   language: Optional[str] = None,
                   alternatives: Tuple[Tuple[str, float], ...] = ()
                   ) -> Optional[Lookup]:
        """Resolve a learned/corpus near-tie in favour of the corpus card.

        The margin guard exists to avoid answering the wrong *subject*. A
        user-taught card that scores within the margin of a corpus card on
        the same question is a restatement of that subject, so the corpus
        answer is safe. Same-class ties are untouched and still refused.
        """
        if len(hits) < 2:
            return None
        (score0, doc0), (score1, doc1) = hits[0], hits[1]
        if score0 - score1 > self.SOFT_MARGIN:
            return None
        learned0 = doc0.id.startswith("learned-")
        learned1 = doc1.id.startswith("learned-")
        if learned0 == learned1:
            return None
        corpus, corpus_score = ((doc1, score1) if learned0
                                else (doc0, score0))
        if corpus_score < self.SOFT_SCORE:
            return None
        return Lookup(
            text=corpus.answer_for(detail),
            source=f"{prefix}_soft:{corpus.id}",
            score=corpus_score,
            margin=None,
            considered=len(hits),
            matched=self._matched_terms(text, corpus, language),
            alternatives=alternatives,
        )

    def answer(self, text: str, language: Optional[str] = None,
               min_score: Optional[float] = None, limit: int = 3,
               prefer_kinds: Iterable[str] = (),
               detail: str = "normal") -> Tuple[Optional[str], str]:
        found = self.lookup(text, language=language, min_score=min_score,
                            limit=limit, prefer_kinds=prefer_kinds,
                            detail=detail)
        return found.text, found.source

    def teachable_units(self, language: Optional[str] = None) -> List[Doc]:
        """Cards this package can teach.

        A package holds one language; asking for another language yields
        nothing rather than the other package's cards.
        """
        if language is not None and language != self.language:
            return []
        return [doc for doc in self.docs if doc.unit is not None]
