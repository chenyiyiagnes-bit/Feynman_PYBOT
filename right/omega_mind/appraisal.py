"""Event appraisal with habituation.

The 26 cubies evaluate one message. Repeated triggers of the same cubie within
a short window are damped, which is what stops a persona from turning into a
caricature after three turns.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from .config import PersonaConfig
from .cubies import CUBIES, Cubie, compiled_triggers


@dataclass
class Activation:
    name: str
    strength: float          # 0..3, after habituation
    effects: Dict[str, float]


HABITUATION_WINDOW = 3
HABITUATION_FACTOR = 0.6
MAX_STRENGTH = 3.0


class AppraisalEngine:
    def __init__(self, persona: PersonaConfig):
        self.persona = persona
        self._recent: Dict[str, Deque[int]] = defaultdict(lambda: deque(maxlen=HABITUATION_WINDOW))
        self._override: Dict[str, List[Tuple[re.Pattern, float]]] = {
            name: [(re.compile(p, re.IGNORECASE), 1.0) for p in patterns]
            for name, patterns in persona.override_triggers.items()
        }

    def evaluate(self, text: str, turn: int,
                 gains: Optional[Dict[str, float]] = None) -> List[Activation]:
        """Score the message against the 26 cubies.

        ``gains`` is the cube feedback layer's bounded multiplier for cubies
        that the conversation has moved onto the front face; it never exceeds
        [0.9, 1.2], so it colours appraisal without being able to invent one.
        """
        gains = gains or {}
        result: List[Activation] = []
        for cubie in CUBIES:
            if cubie.name in self.persona.disable_cubies:
                continue
            strength = self._match(cubie, text)
            if strength <= 0:
                continue
            # Only triggers inside the rolling window damp this one; an event
            # that has not fired for three turns is fresh again.
            prior = self._note(cubie.name, turn)
            scale = HABITUATION_FACTOR ** prior
            effects = dict(cubie.effects)
            effects.update(self.persona.extra_effects.get(cubie.name, {}))
            gain = gains.get(cubie.name, 1.0)
            value = min(MAX_STRENGTH, strength * scale * gain)
            result.append(Activation(cubie.name, round(value, 4), effects))

        # "Being corrected is a reward" needs no match on a disabled cubie.
        if self.persona.correction_as_curiosity and re.search(r"说错|错了|不对|纠正|盲区|搞错", text):
            if not any(a.name == "core_intuition" for a in result):
                prior = self._note("core_intuition", turn)
                gain = gains.get("core_intuition", 1.0)
                result.append(Activation(
                    "core_intuition",
                    round(min(MAX_STRENGTH,
                              1.0 * HABITUATION_FACTOR ** prior * gain), 4),
                    {"dopamine": 0.30, "valence": 0.10},
                ))
        return result

    def _note(self, name: str, turn: int) -> int:
        """Drop triggers older than the window, then record this one.

        Returns the number of *prior* triggers still inside the window, which
        is the exponent for habituation. The first trigger is therefore full
        strength, not already halved.
        """
        recent = self._recent[name]
        while recent and turn - recent[0] >= HABITUATION_WINDOW:
            recent.popleft()
        prior = len(recent)
        recent.append(turn)
        return prior

    def _match(self, cubie: Cubie, text: str) -> float:
        patterns = self._override.get(cubie.name) or compiled_triggers(cubie)
        total = 0.0
        for pattern, gain in patterns:
            if pattern.search(text):
                total += gain
        return min(MAX_STRENGTH, total)

    def reset(self) -> None:
        self._recent.clear()
