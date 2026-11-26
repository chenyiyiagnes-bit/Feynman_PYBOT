"""Sleep: offline consolidation of what happened.

A dream takes undreamed episodes, finds the themes with the most weight, writes
a short narrative from the episodes themselves (quoted, not invented), promotes
durable facts to semantic memory, and lets the affect state settle toward its
baseline. It is deterministic: same episodes, same dream.

Nothing here is mystical — it is the maintenance pass a long-running mind needs
so episodic memory does not grow forever and stress does not accumulate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .config import PersonaConfig
from .learner import CORRECTION_MARKERS
from .memory import Episode, MemoryTiers
from .state import AffectState

# How many undreamed episodes make a dream worth having. Ordinary turns count
# 1; a turn the cards could not answer counts more, and a turn where the user
# had to correct the mind counts most. Type a gap twice and the mind sleeps on
# it that same night instead of waiting out eight unrelated turns.
DREAM_THRESHOLD = 8.0
SURPRISE_MISS = 2.5
SURPRISE_CORRECTION = 3.5


def episode_weight(episode: Episode) -> float:
    """How much this turn argues for a dream: 1.0, 2.5 or 3.5."""
    source = str(getattr(episode, "source", "") or "")
    text = str(getattr(episode, "text", "") or "")
    surprise = 0.0
    if (source.endswith(":no_match") or source.startswith("fallback")
            or "soft" in source):
        surprise = max(surprise, SURPRISE_MISS)
    if any(mark in text for mark in CORRECTION_MARKERS):
        surprise = max(surprise, SURPRISE_CORRECTION)
    return 1.0 + surprise
# A dream keeps at most this many episodes in the narrative.
THEMES_EN = {
    "edge_trauma": "the heavy things",
    "corner_grudge": "what I never swallowed",
    "edge_humor": "the parts that made us laugh",
    "corner_empathy": "when you were hurting",
    "core_logic": "the reasoning",
    "core_intuition": "the gut feeling",
    "edge_nostalgia": "the old days",
    "corner_paranoia": "the worries",
}

THEMES = {
    "edge_trauma": "那些沉的东西",
    "corner_grudge": "没咽下去的话",
    "edge_humor": "笑出来的地方",
    "corner_empathy": "对方难受的时候",
    "core_logic": "讲道理的部分",
    "core_intuition": "凭感觉的部分",
    "edge_nostalgia": "以前的事",
    "corner_paranoia": "担心的事",
}


@dataclass
class Dream:
    index: int
    now: float
    episodes: int
    themes: List[str] = field(default_factory=list)
    narrative: str = ""
    promoted: List[str] = field(default_factory=list)
    settled: Dict[str, float] = field(default_factory=dict)
    # Peak salience (0..10) of the batch. The engine reads this to decide how
    # much of the cube's displacement the dream is allowed to settle.
    salience: float = 0.0
    # Post-dream self-audit: findings, bounded repairs, and whether the
    # repair pass was allowed to write. Empty for old saves.
    audit: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


_DREAM_REQUIRED = ("index", "now", "episodes")
_DREAM_FIELDS = tuple(Dream.__dataclass_fields__)


def _dream_from(entry: object) -> Optional[Dream]:
    """Build a Dream from a save entry, or None if it cannot be used.

    A save from another version may carry extra keys or omit a required one;
    neither may stop the app from loading.
    """
    if not isinstance(entry, dict):
        return None
    if any(key not in entry for key in _DREAM_REQUIRED):
        return None
    return Dream(**{k: entry[k] for k in _DREAM_FIELDS if k in entry})


class Dreamer:
    def __init__(self, threshold: float = DREAM_THRESHOLD):
        self.threshold = threshold
        self.dreams: List[Dream] = []

    def weight(self, memory: MemoryTiers) -> float:
        return sum(episode_weight(e) for e in memory.undreamed())

    def due(self, memory: MemoryTiers) -> bool:
        return self.weight(memory) >= self.threshold

    def consolidate(
        self,
        memory: MemoryTiers,
        state: AffectState,
        persona: PersonaConfig,
        now: float,
        force: bool = False,
        language: str = "zh",
    ) -> Optional[Dream]:
        episodes = memory.undreamed()
        if not episodes:
            return None
        if not force and not self.due(memory):
            return None

        # Salience leads; when episodes tie, the one that actually moved the
        # affect state wins, then recency. This is the read path for
        # Episode.affect, which would otherwise be written every turn and
        # never consulted.
        def intensity(e: Episode) -> float:
            return sum(abs(v) for v in e.affect.values()) if e.affect else 0.0

        episodes = sorted(episodes,
                          key=lambda e: (e.salience, intensity(e), e.turn),
                          reverse=True)
        top = episodes[: min(5, len(episodes))]

        themes_map = THEMES_EN if language == "en" else THEMES
        theme_names = []
        for episode in top:
            for tag in episode.tags:
                label = themes_map.get(tag)
                if label and label not in theme_names:
                    theme_names.append(label)
        theme_names = theme_names[:3]

        # The narrative reuses the user's own lines; the dream invents nothing.
        picked = [e for e in top if e.salience > 0][:3]
        if not picked:
            picked = top[:2]
        if language == "en":
            fragments = [f'"{e.text[:28]}"' for e in picked]
        else:
            fragments = [f"「{e.text[:28]}」" for e in picked]
        if language == "en":
            if theme_names:
                narrative = ("The dream kept turning over " + ", ".join(theme_names)
                             + ". I heard again " + ", and ".join(fragments))
            else:
                narrative = ("Nothing heavy in this one — I just replayed the recent "
                             "words: " + ", ".join(fragments))
        elif theme_names:
            narrative = f"梦里翻的是{'、'.join(theme_names)}。" + "又看到" + "，还有".join(fragments)
        else:
            narrative = "梦里没什么大事，只是把最近的话又听了一遍：" + "，".join(fragments)

        promoted: List[str] = []
        for key, fact in memory.semantic.items():
            if fact["source_turn"] not in {e.turn for e in episodes}:
                continue
            if fact["value"] not in promoted:
                promoted.append(fact["value"])

        before = state.snapshot()
        settled: Dict[str, float] = {}
        for dim in ("cortisol", "defensiveness", "arousal"):
            old = state.get(dim)
            base = persona.baseline(dim)
            state.set(dim, base + (old - base) * 0.55)
            settled[dim] = round(state.get(dim) - before.get(dim, old), 4)
        state.apply({"dopamine": 0.03})

        for episode in episodes:
            episode.dreamed = True

        dream = Dream(
            index=len(self.dreams) + 1,
            now=now,
            episodes=len(episodes),
            themes=theme_names,
            narrative=narrative,
            promoted=promoted,
            settled=settled,
            salience=round(max((e.salience for e in episodes), default=0.0), 3),
        )
        self.dreams.append(dream)
        self.dreams = self.dreams[-50:]
        return dream

    def last(self) -> Optional[Dream]:
        return self.dreams[-1] if self.dreams else None

    def snapshot(self) -> Dict:
        return {"dreams": [d.to_dict() for d in self.dreams]}

    def restore(self, data: Dict) -> None:
        self.dreams = []
        for entry in data.get("dreams", []):
            dream = _dream_from(entry)
            if dream is not None:
                self.dreams.append(dream)
