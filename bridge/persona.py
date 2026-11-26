"""Load a persona.json without importing either brain.

The released format is data, not code: this loader is the only thing the
bridge needs. It validates enough to fail loudly on a malformed persona and
returns a plain dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

DIMENSIONS = ("dopamine", "cortisol", "valence", "arousal",
              "affinity", "defensiveness", "energy")
ROLE_COUNT = {"core": 6, "edge": 12, "corner": 8}


class PersonaError(ValueError):
    pass


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    language: str
    # Which rule profile inside the language the mind reads with ("default"
    # unless a package says otherwise: the 老庄 package reads classical).
    language_profile: str
    baselines: Dict[str, float]
    taus: Dict[str, float]
    voice: Dict[str, List[str]]
    switches: Dict[str, Any]
    disable_cubies: List[str]
    override_triggers: Dict[str, List[str]]
    extra_effects: Dict[str, Dict[str, float]]
    cubies: List[Dict[str, Any]]
    knowledge: Dict[str, Any]
    root: Path

    def baseline(self, dim: str) -> float:
        return self.baselines[dim]

    def kb_dirs(self) -> List[Path]:
        return [(self.root / d).resolve() for d in self.knowledge.get("kb_dirs", [])]

    def corpus_path(self) -> Optional[Path]:
        raw = self.knowledge.get("corpus")
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def load(path: str | Path) -> Persona:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PersonaError(f"persona file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PersonaError(f"{path}: invalid JSON: {exc}") from exc

    if data.get("schema") != 1:
        raise PersonaError(f"{path}: schema must be 1, got {data.get('schema')!r}")

    baselines = data.get("baselines") or {}
    taus = data.get("taus") or {}
    missing = [d for d in DIMENSIONS if d not in baselines or d not in taus]
    if missing:
        raise PersonaError(f"{path}: missing dimensions: {', '.join(missing)}")

    for dim in DIMENSIONS:
        tau = taus[dim]
        if not isinstance(tau, (int, float)) or tau <= 0:
            raise PersonaError(f"{path}: tau[{dim}] must be positive")

    cubies = data.get("cubies") or []
    if len(cubies) != 26:
        raise PersonaError(f"{path}: expected 26 cubies, got {len(cubies)}")
    counts = dict.fromkeys(ROLE_COUNT, 0)
    names = set()
    for cubie in cubies:
        name = cubie.get("name")
        role = cubie.get("role")
        if not name or name in names:
            raise PersonaError(f"{path}: cubie name missing or duplicated: {name!r}")
        names.add(name)
        if role not in ROLE_COUNT:
            raise PersonaError(f"{path}: {name}: bad role {role!r}")
        counts[role] += 1
    for role, want in ROLE_COUNT.items():
        if counts[role] != want:
            raise PersonaError(f"{path}: role {role} expected {want}, got {counts[role]}")

    language = str(data.get("language") or "zh")
    if language not in ("zh", "en"):
        raise PersonaError(f"{path}: language must be 'zh' or 'en', got {language!r}")

    voice = data.get("voice") or {}
    # One package speaks one language: only that language's shell is required.
    # The other pair may be absent (or empty) and is simply never used.
    for key in (f"prefixes_{language}", f"endings_{language}"):
        if not isinstance(voice.get(key), list):
            raise PersonaError(
                f"{path}: voice.{key} must be a list ({language} package)")

    knowledge = data.get("knowledge") or {}
    if not isinstance(knowledge.get("kb_dirs", []), list):
        raise PersonaError(f"{path}: knowledge.kb_dirs must be a list")

    return Persona(
        id=str(data.get("id") or data.get("name") or path.parent.name),
        name=str(data.get("name") or ""),
        language=str(data.get("language") or "zh"),
        language_profile=str(data.get("language_profile") or "default"),
        baselines={d: float(baselines[d]) for d in DIMENSIONS},
        taus={d: float(taus[d]) for d in DIMENSIONS},
        voice={k: list(v) for k, v in voice.items() if isinstance(v, list)},
        switches=dict(data.get("switches") or {}),
        disable_cubies=list(data.get("disable_cubies") or []),
        override_triggers={k: list(v) for k, v in (data.get("override_triggers") or {}).items()},
        extra_effects={k: dict(v) for k, v in (data.get("extra_effects") or {}).items()},
        cubies=list(cubies),
        knowledge=dict(knowledge),
        root=path.parent,
    )


def discover(root: str | Path = "personas") -> Dict[str, Path]:
    """Map persona id → path for every ``<id>/persona.json`` under root."""
    base = Path(root)
    if not base.is_dir():
        return {}
    return {
        child.parent.name: child
        for child in sorted(base.glob("*/persona.json"))
    }
