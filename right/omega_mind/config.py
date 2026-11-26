"""Persona and affect parameters.

All dimensions are scalars in [0, 1] except ``valence`` which is in [-1, 1].
Times are seconds. Taus are exponential-decay time constants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Union

#: Where packages live relative to this module (right/omega_mind/config.py).
PACKAGES_ROOT = Path(__file__).resolve().parents[2] / "personas"

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

    # Which rule profile inside the package's language the mind reads with
    # ("default" / "classical"). The persona package declares it; the CLI and
    # app.runtime both take it from here, so boundaries agree everywhere.
    language_profile: str = "default"
    # Opt-in for the derived term graph; the package decides, not the machine.
    recall_expansion: bool = False

    # cubie name -> replacement regex list
    override_triggers: Dict[str, List[str]] = field(default_factory=dict)
    disable_cubies: List[str] = field(default_factory=list)

    # cubie name -> extra state deltas
    extra_effects: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def baseline(self, dim: str) -> float:
        return self.baselines[dim]

    def tau(self, dim: str) -> float:
        return self.taus[dim]


def generic_persona() -> PersonaConfig:
    return PersonaConfig()


def persona_from_data(data: Mapping) -> PersonaConfig:
    """Map a persona.json (or anything shaped like one) onto a config.

    This is the only conversion from data to affect; ``app.runtime`` and the
    CLI both go through it, so a package's parameters are read one way.
    """
    cfg = PersonaConfig(name=str(data.get("id") or "generic"))
    cfg.baselines.update({k: float(v) for k, v in (data.get("baselines") or {}).items()})
    cfg.taus.update({k: float(v) for k, v in (data.get("taus") or {}).items()})
    voice = data.get("voice") or {}
    for attr, key in (
        ("catchphrase_prefixes", "prefixes_zh"),
        ("ending_particles", "endings_zh"),
        ("catchphrase_prefixes_en", "prefixes_en"),
        ("ending_particles_en", "endings_en"),
    ):
        values = voice.get(key)
        if values:
            setattr(cfg, attr, list(values))
    switches = data.get("switches") or {}
    cfg.correction_as_curiosity = bool(switches.get("correction_as_curiosity", False))
    cfg.jargon_aversion = bool(switches.get("jargon_aversion", False))
    try:
        cfg.politeness = float(switches.get("politeness", 1.0))
    except (TypeError, ValueError):
        cfg.politeness = 1.0
    cfg.language_profile = str(data.get("language_profile") or "default")
    cfg.recall_expansion = bool(data.get("recall_expansion") or False)
    cfg.disable_cubies = list(data.get("disable_cubies") or [])
    cfg.override_triggers = {k: list(v)
                             for k, v in (data.get("override_triggers") or {}).items()}
    cfg.extra_effects = {k: dict(v)
                         for k, v in (data.get("extra_effects") or {}).items()}
    return cfg


def load_package_persona(target: Union[str, Path],
                         root: Union[str, Path] = PACKAGES_ROOT) -> PersonaConfig:
    """The affect config of one persona package, read from its persona.json.

    ``target`` may be a package id, a package directory, or the JSON file.
    Nothing about any persona is compiled in: a package that is not on disk
    is an error, not a silent default.
    """
    path = Path(target)
    if path.is_file():
        manifest = path
    elif (path / "persona.json").is_file():
        manifest = path / "persona.json"
    else:
        manifest = Path(root) / str(target) / "persona.json"
    if not manifest.is_file():
        raise ValueError(f"no persona package named {str(target)!r} "
                         f"(looked for {manifest})")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return persona_from_data(data)


PERSONAS = {
    "generic": generic_persona,
}

#: Names saved states used before the package split. They still mean the
#: same package; the alias only keeps an old save readable, it is not a
#: second copy of the persona's parameters.
LEGACY_PERSONA_IDS = {"feynman": "feynman-zh"}


def get_persona(name: Union[str, Path] = "generic",
                root: Union[str, Path] = PACKAGES_ROOT) -> PersonaConfig:
    """``"generic"``, or any persona package id under ``personas/``."""
    name = LEGACY_PERSONA_IDS.get(str(name), name)
    if str(name) == "generic":
        return generic_persona()
    return load_package_persona(name, root=root)
