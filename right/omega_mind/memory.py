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
# A Chinese capture containing one of these is a question, not a fact:
# "我喜欢什么" captured "什么", "我住在哪儿" captured "哪儿".
_ZH_INTERROGATIVE = ("什么", "啥", "谁", "哪", "几", "多少", "怎么", "怎样",
                     "如何", "为什么", "干嘛")


def _zh_value_ok(value: str) -> bool:
    return bool(value) and not any(mark in value for mark in _ZH_INTERROGATIVE)


def _en_value_ok(value: str) -> bool:
    return len(value) >= 2 and value.lower() not in _EN_FACT_STOP


@dataclass(frozen=True)
class FactChange:
    kind: str
    raw: str
    value: str
    quote: str
    operation: str = "set"
    replaces: str = ""


_MOVE = {
    "zh": re.compile(r"我(?:已经|刚)?(?:从([^，。！？]{2,20}))?搬(?:到了?|去了?)([^，。！？]{2,20}?)(?:了)?"),
    "en": re.compile(r"I (?:have )?moved (?:from (.+?) )?to (.+)", re.I),
}
_RETRACT = {
    "zh": (
        ("lives_in", re.compile(r"我(?:已经|现在|早就)?(?:不再|不)(?:住在|在)([一-鿿A-Za-z· ]{2,20}?)(?:了)?")),
        ("work", re.compile(r"我(?:已经|现在)?不(?:再)?在(.+?)(?:工作|上班)(?:了)?")),
        ("name", re.compile(r"我(?:已经|现在)?不(?:再)?叫(.+?)(?:了)?")),
        ("likes", re.compile(r"我(?:已经|现在)?不再(?:喜欢|爱)(.+?)(?:了)?")),
        ("dislikes", re.compile(r"我(?:已经|现在)?不再(?:讨厌|反感)(.+?)(?:了)?")),
    ),
    "en": (
        ("lives_in", re.compile(r"I (?:(?:no longer|don't|do not) live in|used to live in) (.+)", re.I)),
        ("work", re.compile(r"I (?:no longer|don't|do not) work (?:at|for|as) (.+)", re.I)),
        ("name", re.compile(r"I no longer go by (.+)", re.I)),
        ("likes", re.compile(r"I no longer (?:like|love) (.+)", re.I)),
        ("dislikes", re.compile(r"I no longer (?:dislike|hate) (.+)", re.I)),
    ),
}
_DECLARE = {
    "zh": re.compile(r"^(?:请?记住[：: ]*|其实[，, ]*|我现在(?=住在))"),
    "en": re.compile(r"^(?:please remember(?: that)?[ :]+|actually[, ]+)", re.I),
}
_HYPOTHETICAL = re.compile(
    r"如果|假如|假设|比如|举例|听说|他说|她说|我说过|曾经|以前|过去|可能|也许|"
    r"\b(?:if|suppose|imagine|hypothetically|example|might|maybe)\b|"
    r"\b(?:said|claims?|told me)\b", re.I)
_QUOTES = re.compile(r"""["“”「」『』‘’\x60]|(?:^|\s)'[^']+'(?:$|\s|[.,!?])""")


def fact_changes(text: str, language: Optional[str] = None) -> List[FactChange]:
    """Extract self-reports from complete clauses only."""
    text = re.sub(r"(?<=[A-Za-z])’(?=[A-Za-z])", "'", text)
    if any(mark in text for mark in "?？") or _QUOTES.search(text):
        return []
    language = language or ("zh" if re.search(r"[一-鿿]", text) else "en")
    if _HYPOTHETICAL.search(text) or (
            language == "zh" and re.search(r"(?:说|表示|转述|写道|告诉我)[，：,:]", text)
            and not text.startswith(("我说", "我告诉"))):
        return []
    text = _DECLARE[language].sub(
        lambda match: "我" if match[0].startswith("我现在") else "", text.strip(), count=1)
    patterns = _FACT_PATTERNS if language == "zh" else _EN_FACT_PATTERNS
    value_ok = _zh_value_ok if language == "zh" else _en_value_ok
    changes = {}
    if language == "en":
        text = re.sub(r"\s+(?:and|but)\s+(?=I\s+)", ", ", text, flags=re.I)
        text = re.sub(r"\s+(?:and|but)\s+(?=(?:work|live|like|love|hate|dislike)\s)",
                      ", I ", text, flags=re.I)
    for clause in re.split(r"[，。！,;；!\n]|\.(?:\s+|$)", text):
        clause = clause.strip()
        move = _MOVE[language].fullmatch(clause)
        withdrawals = [(kind, match[1]) for kind, pattern in _RETRACT[language]
                       if (match := pattern.fullmatch(clause)) and value_ok(match[1])]
        if move and value_ok(move[2]):
            raw = move[2]
            value = f"用户住在 {raw}" if language == "zh" else f"User lives in {raw}"
            changes["lives_in"] = FactChange("lives_in", raw, value, clause,
                                             replaces=move[1] or "")
        elif withdrawals:
            kind, raw = withdrawals[0]
            changes[kind] = FactChange(kind, raw, "", clause, "retract")
        else:
            for pattern, kind, template in patterns:
                match = pattern.fullmatch(clause)
                if match and value_ok(match[1]):
                    changes[kind] = FactChange(kind, match[1], template.format(match[1]), clause)
    return list(changes.values())


class MemoryTiers:
    def __init__(self) -> None:
        self.working: Deque[Episode] = deque(maxlen=WORKING_SIZE)
        self.episodes: List[Episode] = []
        # key -> {"value": str, "source_turn": int, "quote": str}
        self.semantic: Dict[str, Dict[str, Any]] = {}
        self.fact_log: List[Dict[str, Any]] = []
        self.withdrawn: Dict[str, Dict[str, Any]] = {}

    # -- writing -----------------------------------------------------------

    def record(self, episode: Episode, changes: Optional[List[FactChange]] = None) -> None:
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
        for change in fact_changes(episode.text) if changes is None else changes:
            self._store_fact(episode, change)

    def _store_fact(self, episode: Episode, change: FactChange) -> None:
        key = f"{change.kind}:{episode.user}"
        prior = self.semantic.get(key)
        latest = prior or self.withdrawn.get(key)
        if latest and latest.get("source_turn", -1) >= episode.turn:
            return
        if (change.operation == "retract" and latest
                and latest.get("raw", "").casefold() != change.raw.casefold()):
            return
        entry = {
            "value": change.value,
            "source_turn": episode.turn,
            "valid_from": episode.now,
            "quote": change.quote,
            "raw": change.raw,
            "operation": change.operation,
        }
        if prior is not None:
            for event in reversed(self.fact_log):
                if event.get("key") == key and event.get("source_turn") == prior.get("source_turn"):
                    event["valid_until"] = episode.now
                    break
        elif change.replaces and latest is None:
            self.fact_log.append({"key": key, "raw": change.replaces,
                                  "operation": "set", "quote": change.quote,
                                  "source_turn": episode.turn, "valid_until": episode.now})
        if change.operation == "retract":
            self.semantic.pop(key, None)
            self.withdrawn[key] = entry
        else:
            self.semantic[key] = entry
            self.withdrawn.pop(key, None)
        self.fact_log.append({"key": key, **entry})
        self.fact_log = self.fact_log[-100:]
        opposite = {"likes": "dislikes", "dislikes": "likes"}.get(change.kind)
        if opposite and change.operation == "set":
            other = self.semantic.get(f"{opposite}:{episode.user}")
            if other and other.get("raw", "").casefold() == change.raw.casefold():
                self._store_fact(episode, FactChange(opposite, change.raw, "", change.quote, "retract"))

    def previous(self, user: str, kind: str) -> Optional[Dict[str, Any]]:
        key = f"{kind}:{user}"
        current = self.semantic.get(key, {}).get("raw")
        for entry in reversed(self.fact_log):
            if (entry.get("key") == key and entry.get("operation", "set") == "set"
                    and entry.get("raw") and entry["raw"] != current):
                return entry
        return self.withdrawn.get(key)

    # -- reading -----------------------------------------------------------

    def recent(self, n: int = 5) -> List[Episode]:
        return list(self.working)[-n:]

    def undreamed(self) -> List[Episode]:
        return [e for e in self.episodes if not e.dreamed]

    def about(self, user: str) -> List[Dict[str, Any]]:
        return [
            {"key": k, **v}
            for k, v in self.semantic.items()
            if k.partition(":")[2] == user
        ]

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "working": [e.to_dict() for e in self.working],
            "episodes": [e.to_dict() for e in self.episodes],
            "semantic": self.semantic,
            "fact_log": self.fact_log,
            "withdrawn": self.withdrawn,
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
        source = data.get("semantic")
        self.semantic = ({key: value for key, value in source.items()
                          if isinstance(value, dict)} if isinstance(source, dict) else {})
        self.fact_log = [entry for entry in (data.get("fact_log") or [])[-100:]
                         if isinstance(entry, dict)]
        source = data.get("withdrawn")
        self.withdrawn = ({key: value for key, value in source.items()
                           if isinstance(value, dict)} if isinstance(source, dict) else {})


def salience_of(activations, deltas: Dict[str, float]) -> float:
    """How much this turn is worth remembering: activation strength plus the
    size of the affect change it caused."""
    act = sum(a.strength for a in activations)
    change = sum(abs(v) for v in deltas.values())
    return round(min(10.0, 0.6 * act + 2.0 * change), 3)
