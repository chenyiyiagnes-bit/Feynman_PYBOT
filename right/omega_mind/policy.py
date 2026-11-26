"""Map affect + activations to a structured style directive.

The directive is the only thing a renderer is allowed to see. Facts never pass
through this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .appraisal import Activation
from .config import PersonaConfig
from .state import AffectState


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class StyleDirective:
    tone: str                 # warm | neutral | cold | sharp
    warmth: float
    directness: float
    verbosity: float
    humor: float
    tempo: float              # speech-rate multiplier
    expression: str           # Live2D-style hint
    prefixes: List[str] = field(default_factory=list)
    endings: List[str] = field(default_factory=list)
    active: List[str] = field(default_factory=list)


class Policy:
    def __init__(self, persona: PersonaConfig):
        self.persona = persona

    def compute(self, state: AffectState, activations: List[Activation],
                language: str = "zh") -> StyleDirective:
        v = state.values
        by_name = {a.name: a.strength for a in activations}
        p = self.persona

        warmth = _clamp(
            0.35
            + 0.40 * v["affinity"]
            + 0.30 * v["valence"]
            - 0.50 * v["defensiveness"]
            - 0.40 * max(0.0, v["cortisol"] - p.baseline("cortisol"))
            + 0.05 * p.politeness
        )
        directness = _clamp(
            0.50
            + 0.30 * v["dopamine"]
            + 0.20 * (1.0 - p.politeness)
            - 0.20 * v["defensiveness"]
        )
        verbosity = _clamp(
            0.40
            + 0.50 * v["energy"]
            + 0.20 * v["arousal"]
            - 0.30 * max(0.0, v["cortisol"] - 0.40)
            - 0.20 * v["defensiveness"]
        )
        humor = _clamp(
            0.20
            + 0.50 * v["dopamine"]
            - 0.40 * max(0.0, v["cortisol"] - 0.35)
            + (0.20 if by_name.get("edge_humor", 0) > 0 else 0.0)
        )

        if p.jargon_aversion and by_name.get("corner_pedantry", 0) > 0:
            tone = "sharp"
        elif v["cortisol"] > 0.55 or v["defensiveness"] > 0.40:
            tone = "cold"
        elif warmth > 0.55:
            tone = "warm"
        else:
            tone = "neutral"

        if tone == "sharp":
            expression = "raised_eyebrow"
        elif tone == "cold":
            expression = "angry_pout"
        elif v["dopamine"] > 0.65 and v["arousal"] > 0.50:
            expression = "excited_curious"
        elif v["defensiveness"] > 0.35:
            expression = "blush_defensive"
        elif tone == "warm":
            expression = "soft_smile"
        else:
            expression = "neutral_calm"

        tempo = _clamp(
            1.0
            + 0.25 * (v["arousal"] - 0.30)
            + 0.15 * (v["dopamine"] - 0.30)
            - 0.20 * max(0.0, v["cortisol"] - 0.40),
            0.80,
            1.25,
        )

        if language == "en":
            habit = list(p.catchphrase_prefixes_en)
            shame = ["(looks away) ", "(voice hardens) "]
            sharp = ["Hold on, ", "Now that's the interesting part — "]
            endings = list(p.ending_particles_en)
        else:
            habit = list(p.catchphrase_prefixes)
            shame = ["（别过头）", "（语气硬了一点）"]
            sharp = ["等一下，", "你这话就有意思了——"]
            endings = list(p.ending_particles)

        prefixes: List[str] = []
        if by_name.get("edge_habit", 0) > 0:
            prefixes = habit
        elif by_name.get("corner_shame", 0) > 0:
            prefixes = shame
        elif tone == "sharp":
            prefixes = sharp

        return StyleDirective(
            tone=tone,
            warmth=round(warmth, 4),
            directness=round(directness, 4),
            verbosity=round(verbosity, 4),
            humor=round(humor, 4),
            tempo=round(tempo, 4),
            expression=expression,
            prefixes=prefixes,
            endings=endings,
            active=[a.name for a in activations],
        )
