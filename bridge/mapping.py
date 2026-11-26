"""Affect → typesetter parameters.

The right brain produces a 7-dimensional affect state and a cube bias
(depth / verbosity). The left brain accepts a ParagraphRequest with a
``style_deg``, eight operator weights and a set of on/off flags.

This module is the whole mapping, and it is deliberately a pure function:
same input → same output, no imports from either brain, no randomness.
That makes it testable and keeps the "off" path byte-identical.

Design rules
------------
1. Everything is expressed as a *deviation from the persona's baseline*, so
   the same mapping works for 费曼 (high dopamine, low cortisol) and 老庄
   (low arousal, high valence) without per-persona tuning.
2. Corrections are monotonic: raising curiosity can never lower ``cross``.
3. All outputs are clamped to the left brain's own field bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

DIMENSIONS = ("dopamine", "cortisol", "valence", "arousal",
              "affinity", "defensiveness", "energy")

OPERATORS = ("micro", "cross", "boundary", "reversal",
             "anti_intuition", "jargon_strip", "least_action", "cargo_cult")

# Defaults copied from the left brain's models so the two can be compared
# without importing pydantic. tests/test_bridge_mapping.py asserts they agree.
STYLE_DEG_BOUNDS = (0.0, 120.0)
OPERATOR_BOUNDS = (0.0, 2.0)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _norm_positive(value: float, baseline: float) -> float:
    """Map a [0,1] dimension to [0,1] relative to its baseline.

    baseline 0.2 and value 0.6 → 0.5 (half of the remaining headroom).
    """
    headroom = 1.0 - baseline
    if headroom <= 1e-9:
        return 1.0 if value >= baseline else 0.0
    return _clamp((value - baseline) / headroom, 0.0, 1.0)


def _norm_below(value: float, baseline: float) -> float:
    """How far a [0,1] dimension sits *below* its baseline, in [0,1].

    At or above baseline → 0 (not tired, not lacking). At zero → 1.
    """
    if value >= baseline:
        return 0.0
    return 0.0 if baseline <= 1e-9 else _clamp((baseline - value) / baseline, 0.0, 1.0)


def _norm_symmetric(value: float, baseline: float) -> float:
    """Same idea for valence, which lives in [-1, 1]."""
    if value >= baseline:
        span = 1.0 - baseline
        return 0.0 if span <= 1e-9 else _clamp((value - baseline) / span, 0.0, 1.0)
    span = baseline - (-1.0)
    return 0.0 if span <= 1e-9 else -_clamp((baseline - value) / span, 0.0, 1.0)


@dataclass(frozen=True)
class Affects:
    """Normalised, persona-relative affects. All in [0,1] except valence."""

    curiosity: float
    stress: float
    engagement: float
    warmth: float
    guard: float
    fatigue: float
    valence: float

    @classmethod
    def from_state(cls, state: Mapping[str, float],
                   baselines: Mapping[str, float]) -> "Affects":
        def val(dim: str) -> float:
            return float(state.get(dim, baselines.get(dim, 0.0)))

        def base(dim: str) -> float:
            return float(baselines.get(dim, 0.0))

        return cls(
            curiosity=_norm_positive(val("dopamine"), base("dopamine")),
            stress=_norm_positive(val("cortisol"), base("cortisol")),
            engagement=_norm_positive(val("arousal"), base("arousal")),
            warmth=_norm_positive(val("affinity"), base("affinity")),
            guard=_norm_positive(val("defensiveness"), base("defensiveness")),
            fatigue=_norm_below(val("energy"), base("energy")),
            valence=_norm_symmetric(val("valence"), base("valence")),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "curiosity": self.curiosity,
            "stress": self.stress,
            "engagement": self.engagement,
            "warmth": self.warmth,
            "guard": self.guard,
            "fatigue": self.fatigue,
            "valence": self.valence,
        }


@dataclass(frozen=True)
class BridgeParams:
    style_deg: float
    operators: Dict[str, float]
    flags: Dict[str, bool] = field(default_factory=dict)
    # Diagnostics, never used by the typesetter itself.
    affects: Dict[str, float] = field(default_factory=dict)
    cube_depth: float = 0.0
    cube_verbosity: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "style_deg": round(self.style_deg, 3),
            "operators": {k: round(v, 3) for k, v in self.operators.items()},
            "flags": dict(self.flags),
            "affects": {k: round(v, 3) for k, v in self.affects.items()},
            "cube_depth": round(self.cube_depth, 3),
            "cube_verbosity": round(self.cube_verbosity, 3),
        }


def map_affect(state: Mapping[str, float],
               baselines: Mapping[str, float],
               cube_depth: float = 0.0,
               cube_verbosity: float = 0.0,
               jargon_aversion: bool = False) -> BridgeParams:
    """Turn affect + cube bias into typesetter parameters.

    ``cube_depth`` and ``cube_verbosity`` are the right brain's CubeBias
    fields; they are in [-1, 1] by construction (tanh-gained), so they act as
    a signed nudge on top of the affect mapping.
    """
    a = Affects.from_state(state, baselines)
    depth = _clamp(float(cube_depth), -1.0, 1.0)
    verbosity = _clamp(float(cube_verbosity), -1.0, 1.0)

    # --- operator weights (left brain bounds: 0..2) ---
    cross = 0.40 + 1.40 * a.curiosity + 0.25 * max(depth, 0.0)
    anti_intuition = 0.30 + 1.20 * a.curiosity
    boundary = 0.30 + 1.00 * a.engagement - 0.40 * a.fatigue
    micro = 0.50 + 1.20 * a.stress + 0.60 * (1.0 - a.engagement)
    reversal = 0.20 + 0.80 * a.engagement
    jargon_strip = 0.30 + 1.50 * a.stress + (0.50 if jargon_aversion else 0.0)
    least_action = 0.20 + 0.80 * a.fatigue
    cargo_cult = 0.20 + 0.90 * a.guard

    raw = {
        "micro": micro,
        "cross": cross,
        "boundary": boundary,
        "reversal": reversal,
        "anti_intuition": anti_intuition,
        "jargon_strip": jargon_strip,
        "least_action": least_action,
        "cargo_cult": cargo_cult,
    }
    operators = {k: _clamp(v, *OPERATOR_BOUNDS) for k, v in raw.items()}

    # --- style: playful when curious and warm, plain when stressed ---
    style_raw = (
        55.0
        + 22.0 * a.curiosity
        + 10.0 * a.warmth
        + 8.0 * max(a.valence, 0.0)
        - 18.0 * a.stress
        - 10.0 * a.fatigue
        + 8.0 * verbosity
    )
    style_deg = _clamp(style_raw, *STYLE_DEG_BOUNDS)

    flags = {
        # A stressed mind narrows to what was actually asked.
        "topic_lock": a.stress >= 0.55,
        # Analogy is only worth paying for when the mind is curious and the
        # cube has actually been moved by this conversation.
        "use_isomorphism": a.curiosity >= 0.45 and depth > 0.0,
        # Build paragraphs step by step only when engaged.
        "conditioned_chain": a.engagement >= 0.50 and a.fatigue < 0.60,
        # Repair a tense arc when the exchange has been rough.
        "dissonance_resolution": a.stress >= 0.60 or a.guard >= 0.50,
    }

    return BridgeParams(
        style_deg=style_deg,
        operators=operators,
        flags=flags,
        affects=a.to_dict(),
        cube_depth=depth,
        cube_verbosity=verbosity,
    )
