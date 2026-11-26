"""Second-order breakthrough dynamics.

The affect state is a leaky integrator. This module adds the one thing a
first-order integrator cannot do: a phase-locked loop around a design
resonance, where a strong appraisal does not merely push the state once but
re-enters it as a short sequence of cube turns (the "rumination" phase).

The constant 2.25 is a *design* parameter. It is not the matrix-multiplication
exponent: whether omega = 2.25 is an open problem in algebraic complexity, and
no proof of it exists. The name is a metaphor borrowed from the rumour; the
code treats it only as a threshold on this mind's activation energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from .appraisal import Activation
from .cubies import CUBIES_BY_NAME

OMEGA_RESONANCE = 2.25          # breakthrough energy (design constant)
FIRST_ORDER_INERTIA = 1.5       # sqrt(2.25): power in the first-order term
DAMPING_FACTOR = 2.0 / 3.0      # energy kept by one rumination cycle
HALF_LIFE_SECONDS = 45.0        # wall-clock half-life of residual energy
HARMONY_THRESHOLD = 2.0 / 9.0   # normalised distance still counted as in tune
TORQUE_DEG_PER_UNIT = 40.0      # visual rotation per unit of energy
SPRING_BACK_DEG = 20.0          # how far the needle falls back after a burst

# A verdant/harmonic appraisal sits between 1.75 and 2.75 energy: the burst is
# strong enough to re-enter the loop, but not so strong that it saturates the
# state. Kept explicit so the policy layer can use it.
HARMONY_LOW = OMEGA_RESONANCE * (1.0 - HARMONY_THRESHOLD)
HARMONY_HIGH = OMEGA_RESONANCE * (1.0 + HARMONY_THRESHOLD)


@dataclass
class Resonance:
    energy: float              # carried + this turn's fresh activation
    carried: float             # residual from the previous turn(s)
    fresh: float               # this turn's appraisal energy
    depth: int                 # 2..4, number of phases in the loop
    breakthrough: bool
    in_harmony: bool           # observation only: steady-state telemetry,
                               # no policy consumer (breakthrough/depth drive
                               # the cube). Kept so the dashboard can show how
                               # often the loop sits inside its design band.
    torque_deg: float
    spring_back_deg: float
    damping: float
    retained: float            # energy left for the next turn

    def to_dict(self) -> dict:
        return {
            "energy": round(self.energy, 4),
            "carried": round(self.carried, 4),
            "fresh": round(self.fresh, 4),
            "depth": self.depth,
            "breakthrough": self.breakthrough,
            "in_harmony": self.in_harmony,
            "torque_deg": round(self.torque_deg, 2),
            "spring_back_deg": round(self.spring_back_deg, 2),
            "damping": round(self.damping, 4),
            "retained": round(self.retained, 4),
        }


def activation_energy(activations: Iterable[Activation], bonus: float = 0.0) -> float:
    """Sum of appraisal strengths; a correction can inject its own jolt."""
    return max(0.0, sum(max(0.0, a.strength) for a in activations) + bonus)


def residual_energy(energy: float, seconds: float) -> float:
    """Energy still circulating after ``seconds`` of idle decay."""
    if energy <= 0.0 or seconds <= 0.0:
        return max(0.0, energy)
    return energy * math.exp(-math.log(2.0) * seconds / HALF_LIFE_SECONDS)


def carried_energy(energy: float, last_ts: float | None, now: float) -> float:
    """Residual from an earlier turn, decayed by wall-clock time.

    The first-order affect state forgets exponentially; the second-order loop
    keeps a little of the burst circulating for a few half-lives. A missing
    timestamp means there is nothing to carry.
    """
    if last_ts is None:
        return 0.0
    return residual_energy(energy, max(0.0, now - last_ts))


def damp(energy: float) -> float:
    """Energy kept after one rumination cycle, minus the spring-back.

    DAMPING_FACTOR says two thirds of the burst re-enters; SPRING_BACK_DEG is
    the visual needle falling back 20 degrees, expressed in energy units so
    the two numbers live on the same scale.
    """
    return max(0.0, energy * DAMPING_FACTOR - SPRING_BACK_DEG / TORQUE_DEG_PER_UNIT)


def assess(activations: Iterable[Activation], bonus: float = 0.0,
           carried: float = 0.0) -> Resonance:
    fresh = activation_energy(activations, bonus)
    energy = max(0.0, carried) + fresh
    depth = min(4, max(2, int(energy / FIRST_ORDER_INERTIA) + 1))
    breakthrough = energy >= OMEGA_RESONANCE
    return Resonance(
        energy=energy,
        carried=max(0.0, carried),
        fresh=fresh,
        depth=depth,
        breakthrough=breakthrough,
        in_harmony=HARMONY_LOW <= energy <= HARMONY_HIGH,
        torque_deg=min(180.0, energy * TORQUE_DEG_PER_UNIT),
        spring_back_deg=SPRING_BACK_DEG,
        damping=DAMPING_FACTOR,
        retained=damp(energy) if breakthrough else energy,
    )


_ADJACENT = {"U": "R", "R": "F", "F": "D", "D": "L", "L": "B", "B": "U"}


def rumination_moves(face: str, depth: int) -> List[str]:
    """A deterministic short commutator: the loop re-enters, then closes.

    ``depth`` 2 -> F U; 3 -> F U F'; 4 -> F U F' U'. Using an adjacent face
    avoids immediately undoing the quarter turn that triggered the burst.
    """
    face = face.upper()
    if face not in _ADJACENT:
        raise ValueError(f"unknown face: {face}")
    adjacent = _ADJACENT[face]
    sequence = [face, adjacent, f"{face}'", f"{adjacent}'"]
    return sequence[: max(0, min(4, depth))]


def dominant_face(activations: Iterable[Activation]) -> str | None:
    acts = [a for a in activations if a.strength > 0 and a.name in CUBIES_BY_NAME]
    if not acts:
        return None
    strongest = max(acts, key=lambda a: a.strength)
    return CUBIES_BY_NAME[strongest.name].face
