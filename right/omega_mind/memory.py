"""Three-tier memory: working (recent), episodic (salient events), semantic (facts).

The engine writes an episode every turn; the dreamer reads episodes and
promotes durable facts into semantic memory. Everything here is plain data
with explicit caps so a long-running mind cannot grow without bound.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, List, Optional

WORKING_SIZE = 16
EPISODIC_CAP = 500


@dataclass
class Episode:
    turn: int
    now: float
    user: str
    text: str
    reply: str
    source: str
    tags: List[str] = field(default_factory=list)
    affect: Dict[str, float] = field(default_factory=dict)
    salience: float = 0.0
    dreamed: bool = False
    # Retrieval evidence for this turn (score, margin, alternatives). The
    # dream-time audit reads it after the episode is no longer in the event
    # log; an old save without it simply has no diagnostics.
    retrieval: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_EPISODE_REQUIRED = ("turn", "now", "user", "text", "reply", "source")
_EPISODE_FIELDS = tuple(Episode.__dataclass_fields__)


def _episode_from(entry: Any) -> Optional[Episode]:
    """Build an Episode from a save entry, or None if it cannot be used.

    Extra keys are ignored (a save from another version); an entry without
    the required fields is skipped rather than crashing the load.
    """
    if not isinstance(entry, dict):
        return None
    if any(key not in entry for key in _EPISODE_REQUIRED):
        return None
    return Episode(**{k: entry[k] for k in _EPISODE_FIELDS if k in entry})


# Semantic facts are extracted, not generated: each pattern either matches or
# it does not, and the quote is stored whole.
_FACT_PATTERNS = [
    (re.compile(r"我叫([一-鿿A-Za-z·]{2,8})"), "name", r"用户的名字是 {0}"),
    (re.compile(r"我的名字是([一-鿿A-Za-z·]{2,8})"), "name", r"用户的名字是 {0}"),
    (re.compile(r"我(?:很|特别|最)?(?:喜欢|爱)([一-鿿A-Za-z0-9]{1,12})"), "likes", r"用户喜欢 {0}"),
    (re.compile(r"我(?:很|特别|最)?(?:讨厌|不喜欢|受不了)([一-鿿A-Za-z0-9]{1,12})"), "dislikes", r"用户不喜欢 {0}"),
    (re.compile(r"我住在([一-鿿A-Za-z]{2,10})"), "lives_in", r"用户住在 {0}"),
    (re.compile(r"我(?:在|是)([一-鿿]{2,6})(?:工作|上班)"), "work", r"用户的工作与 {0} 有关"),
]

# English mode speaks a different language, so the same five facts need their
# own patterns. The value is cut at the first clause boundary; a pronoun or
# filler ("I love it") is not a fact and is dropped.
_EN_FACT_PATTERNS = [
    (re.compile(r"\bmy name(?:'s| is) ([A-Za-z][\w .'\-]{0,29})", re.I),
     "name", r"User's name is {0}"),
    (re.compile(r"\bcall me ([A-Za-z][\w .'\-]{0,29})", re.I),
     "name", r"User's name is {0}"),
    (re.compile(r"\bi (?:really |especially |also )?(?:like|love) "
                r"([A-Za-z][\w .'\-]{0,29})", re.I),
     "likes", r"User likes {0}"),
    (re.compile(r"\bi (?:really )?(?:hate|dislike|can't stand|don't like|"
                r"do not like) ([A-Za-z][\w .'\-]{0,29})", re.I),
     "dislikes", r"User dislikes {0}"),
    (re.compile(r"\bi (?:live|am living) in ([A-Za-z][\w .'\-]{0,29})", re.I),
     "lives_in", r"User lives in {0}"),
    (re.compile(r"\bi work (?:at|for) ([A-Za-z][\w .'\-]{0,29})", re.I),
     "work", r"User works at {0}"),
    (re.compile(r"\bi work as ([A-Za-z][\w .'\-]{0,29})", re.I),
     "work", r"User works as {0}"),
]

_EN_FACT_STOP = {
    "it", "this", "that", "you", "him", "her", "them", "us", "everything",
    "nothing", "anything", "something", "people", "things", "stuff", "life",
    "sorry", "fine", "okay", "ok", "sure", "too", "a", "an", "the",
    # Interrogatives: "What do I like?" must not store likes="what".
    "what", "where", "who", "whom", "whose", "which", "why", "how", "when",
}
_CLAUSE_SPLIT = re.compile(r"[,.;:!?\n]|\s+(?:and|but|because|so)\s+", re.I)

# A Chinese capture containing one of these is a question, not a fact:
# "我喜欢什么" captured "什么", "我住在哪儿" captured "哪儿".
_ZH_INTERROGATIVE = ("什么", "啥", "谁", "哪", "几", "多少", "怎么", "怎样",
                     "如何", "为什么", "干嘛")


def _zh_value_ok(value: str) -> bool:
    return bool(value) and not any(mark in value for mark in _ZH_INTERROGATIVE)


def _clean_en_value(raw: str) -> str:
    value = _CLAUSE_SPLIT.split(raw, maxsplit=1)[0]
    return value.strip().strip(" .'\"-")


def _en_value_ok(value: str) -> bool:
    return len(value) >= 2 and value.lower() not in _EN_FACT_STOP


class MemoryTiers:
    def __init__(self) -> None:
        self.working: Deque[Episode] = deque(maxlen=WORKING_SIZE)
        self.episodes: List[Episode] = []
        # key -> {"value": str, "source_turn": int, "quote": str}
        self.semantic: Dict[str, Dict[str, Any]] = {}
        self.fact_log: List[Dict[str, Any]] = []

    # -- writing -----------------------------------------------------------

    def record(self, episode: Episode) -> None:
        self.working.append(episode)
        self.episodes.append(episode)
        if len(self.episodes) > EPISODIC_CAP:
            # Keep every undreamed episode (the dreamer still owes them a pass),
            # then the most salient dreamed ones up to the cap. Turn order is
            # restored afterwards so reads stay chronological. If nobody ever
            # dreams, the undreamed pool is capped too, oldest dropped first:
            # a batch caller must not grow memory without bound.
            undreamed = [e for e in self.episodes if not e.dreamed]
            undreamed = undreamed[-EPISODIC_CAP:]
            dreamed = [e for e in self.episodes if e.dreamed]
            room = max(0, EPISODIC_CAP - len(undreamed))
            dreamed = sorted(dreamed, key=lambda e: e.salience, reverse=True)[:room]
            self.episodes = sorted(undreamed + dreamed, key=lambda e: e.turn)
        self._extract(episode)

    def _extract(self, episode: Episode) -> None:
        for pattern, kind, template in _FACT_PATTERNS:
            match = pattern.search(episode.text)
            if match and _zh_value_ok(match.group(1)):
                self._store_fact(episode, kind, template.format(match.group(1)),
                                 match.group(0), raw=match.group(1))
        for pattern, kind, template in _EN_FACT_PATTERNS:
            match = pattern.search(episode.text)
            if not match:
                continue
            value = _clean_en_value(match.group(1))
            if not _en_value_ok(value):
                continue
            quote = match.group(0).strip().rstrip(" .,;:!?")
            self._store_fact(episode, kind, template.format(value), quote,
                             raw=value)

    def _store_fact(self, episode: Episode, kind: str, value: str,
                    quote: str, raw: Optional[str] = None) -> None:
        key = f"{kind}:{episode.user}"
        entry = {
            "value": value,
            "source_turn": episode.turn,
            "quote": quote,
            "raw": raw if raw is not None else value,
        }
        if self.semantic.get(key) != entry:
            self.semantic[key] = entry
            self.fact_log.append({"key": key, **entry})
            self.fact_log = self.fact_log[-100:]

    # -- reading -----------------------------------------------------------

    def recent(self, n: int = 5) -> List[Episode]:
        return list(self.working)[-n:]

    def undreamed(self) -> List[Episode]:
        return [e for e in self.episodes if not e.dreamed]

    def about(self, user: str) -> List[Dict[str, Any]]:
        return [
            {"key": k, **v}
            for k, v in self.semantic.items()
            if k.endswith(f":{user}")
        ]

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "working": [e.to_dict() for e in self.working],
            "episodes": [e.to_dict() for e in self.episodes],
            "semantic": self.semantic,
            "fact_log": self.fact_log,
        }

    def restore(self, data: Dict[str, Any]) -> None:
        # Unknown keys come from a save written by a different version. A
        # save must never stop the app from starting, so extras are dropped
        # and entries without the required fields are skipped.
        working = [_episode_from(e) for e in data.get("working", [])]
        episodes = [_episode_from(e) for e in data.get("episodes", [])]
        self.working = deque((e for e in working if e is not None),
                             maxlen=WORKING_SIZE)
        self.episodes = [e for e in episodes if e is not None]
        self.semantic = data.get("semantic", {})
        self.fact_log = data.get("fact_log", [])


def salience_of(activations, deltas: Dict[str, float]) -> float:
    """How much this turn is worth remembering: activation strength plus the
    size of the affect change it caused."""
    act = sum(a.strength for a in activations)
    change = sum(abs(v) for v in deltas.values())
    return round(min(10.0, 0.6 * act + 2.0 * change), 3)
