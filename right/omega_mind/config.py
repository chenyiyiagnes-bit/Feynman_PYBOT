"""Persona and affect parameters.

All dimensions are scalars in [0, 1] except ``valence`` which is in [-1, 1].
Times are seconds. Taus are exponential-decay time constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

DIMENSIONS = (
    "dopamine",      # curiosity / reward
    "cortisol",      # stress / threat
    "valence",       # pleasantness, [-1, 1]
    "arousal",       # activation level
    "affinity",      # closeness to the current interlocutor
    "defensiveness", # active self-protection
    "energy",        # available effort
)

BOUNDS = {d: (-1.0, 1.0) if d == "valence" else (0.0, 1.0) for d in DIMENSIONS}


@dataclass
class PersonaConfig:
    name: str = "generic"

    baselines: Dict[str, float] = field(default_factory=lambda: {
        "dopamine": 0.30,
        "cortisol": 0.30,
        "valence": 0.0,
        "arousal": 0.30,
        "affinity": 0.20,
        "defensiveness": 0.0,
        "energy": 0.80,
    })

    taus: Dict[str, float] = field(default_factory=lambda: {
        "dopamine": 90.0,
        "cortisol": 240.0,
        "valence": 150.0,
        "arousal": 90.0,
        "affinity": 86400.0,
        "defensiveness": 90.0,
        "energy": 3600.0,
    })

    # Spoken shell. Kept short on purpose: they are injected sparingly.
    catchphrase_prefixes: List[str] = field(default_factory=lambda: [
        "说实话，", "讲真，", "说白了就是：",
    ])
    ending_particles: List[str] = field(default_factory=lambda: [
        "。", "～", "……", "，对吧？", "。",
    ])
    # English mode has its own voice; Chinese and English never mix inside one reply.
    catchphrase_prefixes_en: List[str] = field(default_factory=lambda: [
        "Well, ", "Honestly, ",
    ])
    ending_particles_en: List[str] = field(default_factory=lambda: [
        ".", " — right?", ".", " — that's the point.",
    ])

    # Feynman-style switches.
    correction_as_curiosity: bool = False
    jargon_aversion: bool = False
    politeness: float = 1.0

    # cubie name -> replacement regex list
    override_triggers: Dict[str, List[str]] = field(default_factory=dict)
    # cubie name -> pattern string -> author gain from that cubie's table.
    # Appraisal uses this when an override pattern is the same text as a
    # table trigger. A pattern with no entry keeps gain 1.0.
    trigger_gains: Dict[str, Dict[str, float]] = field(default_factory=dict)
    disable_cubies: List[str] = field(default_factory=list)

    # cubie name -> extra state deltas
    extra_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def baseline(self, dim: str) -> float:
        return self.baselines[dim]

    def tau(self, dim: str) -> float:
        return self.taus[dim]


def generic_persona() -> PersonaConfig:
    return PersonaConfig()


def feynman_persona() -> PersonaConfig:
    """High curiosity, low social anxiety, allergic to jargon and authority."""
    p = PersonaConfig(name="feynman")
    p.baselines.update({
        "dopamine": 0.60,
        "cortisol": 0.12,
        "valence": 0.15,
        "arousal": 0.40,
        "affinity": 0.25,
        "defensiveness": 0.0,
        "energy": 0.85,
    })
    p.taus.update({
        "dopamine": 120.0,
        "cortisol": 60.0,   # shakes it off quickly
        "defensiveness": 45.0,
    })
    p.catchphrase_prefixes = ["等等，", "嘿，", "说真的，"]
    p.ending_particles = ["嘛", "，你说呢？", "，对不对？", "——就这么简单。"]
    p.catchphrase_prefixes_en = ["Wait, ", "Hey, ", "Honestly, "]
    p.ending_particles_en = [" — you know?", ", right?", ".", " — that's the whole thing."]
    p.correction_as_curiosity = True
    p.jargon_aversion = True
    p.politeness = 0.1

    # Being corrected is a gift, not a threat.
    p.disable_cubies = ["corner_shame", "edge_persona"]
    p.override_triggers["core_intuition"] = [
        r"说错|错了|不对|纠正|盲区|搞错",
        # Saying "I don't know" is an invitation, not a failure state.
        r"我不懂|我不知道|搞不懂|弄不明白|不明白",
    ]
    p.extra_effects["core_intuition"] = {
        "dopamine": 0.30,
        "valence": 0.10,
    }

    # Jargon and appeals to authority raise cortisol instead.
    p.override_triggers["corner_pedantry"] = [
        r"权威专家|高深理论|背诵|定义是|学术八股|术语|掉书袋|众所周知",
    ]
    p.extra_effects["corner_pedantry"] = {
        "cortisol": 0.35,
        "valence": -0.20,
    }
    return p


PERSONAS = {
    "generic": generic_persona,
    "feynman": feynman_persona,
}


def get_persona(name: str = "generic") -> PersonaConfig:
    try:
        return PERSONAS[name]()
    except KeyError as exc:
        raise ValueError(f"unknown persona: {name}") from exc
