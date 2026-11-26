"""Bounded affect state with leaky-integrator decay toward a baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from .config import BOUNDS, DIMENSIONS, PersonaConfig


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class AffectState:
    values: Dict[str, float] = field(default_factory=dict)
    last_ts: float = 0.0

    @classmethod
    def from_persona(cls, persona: PersonaConfig, now: float) -> "AffectState":
        return cls(
            values={d: persona.baseline(d) for d in DIMENSIONS},
            last_ts=now,
        )

    def get(self, dim: str) -> float:
        return self.values[dim]

    def set(self, dim: str, value: float) -> None:
        low, high = BOUNDS[dim]
        self.values[dim] = _clamp(float(value), low, high)

    def advance(self, now: float, persona: PersonaConfig,
                baselines: Dict[str, float] | None = None) -> float:
        """Decay every dimension toward its baseline. Returns elapsed seconds.

        ``baselines`` overrides the persona's static targets when the cube
        feedback layer reports a chronic offset. The offset itself is capped
        there, so this method only needs to stay inside the value bounds.
        """
        dt = max(0.0, now - self.last_ts)
        if dt == 0:
            return 0.0
        for dim in DIMENSIONS:
            base = (baselines or {}).get(dim, persona.baseline(dim))
            tau = max(1e-6, persona.tau(dim))
            self.values[dim] = base + (self.values[dim] - base) * math.exp(-dt / tau)
        self.last_ts = now
        return dt

    def apply(self, deltas: Dict[str, float]) -> None:
        for dim, delta in deltas.items():
            if dim not in self.values:
                continue
            low, high = BOUNDS[dim]
            self.values[dim] = _clamp(self.values[dim] + delta, low, high)

    def snapshot(self) -> Dict[str, float]:
        return {dim: round(self.values[dim], 6) for dim in DIMENSIONS}

    def restore(self, data: Dict[str, float], last_ts: float | None = None) -> None:
        for dim in DIMENSIONS:
            if dim in data:
                self.set(dim, float(data[dim]))
        if last_ts is not None:
            self.last_ts = float(last_ts)
