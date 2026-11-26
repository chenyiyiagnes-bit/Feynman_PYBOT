"""Right-brain state -> left-brain retrieval and composition policy.

The right brain never supplies facts. It only chooses how hard the left brain
should look, which kinds of cards it currently prefers, and how much of a card
to say. Every field here is a bound on retrieval or field selection; no text is
invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .appraisal import Activation
from .config import PersonaConfig
from .state import AffectState


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


# Curiosity is dopamine above the calm midpoint. The gate sits at 0.09 so a
# strong compliment reaches it: with the old 0.20 gate, a peak of 0.72 was
# still treated as "not curious", and the whole explore branch never fired.
CURIOSITY_GATE = 0.09
# Fatigue is measured against a full battery of 0.70, not 0.55: energy only
# drops under sustained effort, and the floor it reaches in normal dialogue is
# around 0.60, so the old cutoff was unreachable.
FATIGUE_REF = 0.70
FATIGUE_GATE = 0.10


@dataclass(frozen=True)
class LeftBrainPolicy:
    # Minimum retrieval score. Stress and defensiveness raise it; curiosity
    # lowers it a little, because a curious mind is willing to follow a
    # looser thread.
    min_score: float = 0.30
    # How many candidates the retriever may consider before choosing one.
    limit: int = 3
    # Card kinds to bias upward when scores are close.
    prefer_kinds: Tuple[str, ...] = ()
    # brief | normal | rich: how many optional fields of a card to speak.
    detail: str = "normal"
    # answer | careful | explore: observable posture for the turn.
    posture: str = "answer"
    # Diagnostic values, not used by retrieval itself.
    stress: float = 0.0
    curiosity: float = 0.0
    fatigue: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "min_score": round(self.min_score, 4),
            "limit": self.limit,
            "prefer_kinds": list(self.prefer_kinds),
            "detail": self.detail,
            "posture": self.posture,
            "stress": round(self.stress, 4),
            "curiosity": round(self.curiosity, 4),
            "fatigue": round(self.fatigue, 4),
        }


def plan_for(state: AffectState, persona: PersonaConfig,
             activations: List[Activation],
             cube_bias=None) -> LeftBrainPolicy:
    """Compute the left-brain policy from the current affect state.

    ``cube_bias`` is the feedback layer's read of the front corners: it can
    widen or narrow the aperture, but the stress gate still wins — a guarded
    mind stays terse even when an inquisitive corner is on stage.
    """
    v = state.values
    baseline_cortisol = persona.baseline("cortisol")
    # A single sharp event can tighten the gate before it has moved the slow
    # state much: being challenged, talked down to, or put on guard.
    by_name = {a.name: a.strength for a in activations}
    alert = max(
        (min(1.0, by_name.get(name, 0.0)) for name in (
            "edge_trauma", "corner_pedantry", "edge_suspicion",
            "corner_paranoia", "core_id",
        )),
        default=0.0,
    )
    stress = _clamp(max(
        0.0,
        v["cortisol"] - max(0.40, baseline_cortisol),
        0.35 * alert,
    ))
    curiosity = _clamp(max(0.0, v["dopamine"] - 0.55))
    fatigue = _clamp(max(0.0, FATIGUE_REF - v["energy"]))
    guard = _clamp(v["defensiveness"])

    # A stressed or defensive mind demands a better match before it asserts
    # anything. Curiosity loosens the gate slightly, but never below 0.26.
    min_score = _clamp(
        0.30 + 0.22 * stress + 0.16 * guard - 0.10 * curiosity,
        0.26, 0.52,
    )

    if fatigue > FATIGUE_GATE:
        limit = 1
    elif curiosity > 0.25:
        limit = 4
    else:
        limit = 3

    if curiosity >= CURIOSITY_GATE:
        prefer: Tuple[str, ...] = ("teachable_unit", "method_card")
    elif stress >= 0.15 or guard >= 0.35:
        prefer = ("physics_edge",)
    else:
        prefer = ()

    if fatigue > FATIGUE_GATE or stress >= 0.20:
        detail = "brief"
    elif curiosity >= CURIOSITY_GATE and v["energy"] >= 0.55:
        detail = "rich"
    else:
        detail = "normal"

    if stress >= 0.20 or guard >= 0.35:
        posture = "careful"
    elif curiosity >= CURIOSITY_GATE:
        posture = "explore"
    else:
        posture = "answer"

    if cube_bias is not None:
        # Cube bias is a per-front-slot average (see CubeFeedback.bias), so a
        # one-step / 0.20 tilt is already a real shift of the whole face.
        if cube_bias.depth >= 1:
            limit = min(4, limit + 1)
        elif cube_bias.depth <= -1:
            limit = max(1, limit - 1)
        if detail != "brief" and cube_bias.verbosity >= 0.20 and v["energy"] >= 0.55:
            detail = "rich"
        elif cube_bias.verbosity <= -0.20:
            detail = "brief"

    return LeftBrainPolicy(
        min_score=min_score,
        limit=limit,
        prefer_kinds=prefer,
        detail=detail,
        posture=posture,
        stress=stress,
        curiosity=curiosity,
        fatigue=fatigue,
    )
