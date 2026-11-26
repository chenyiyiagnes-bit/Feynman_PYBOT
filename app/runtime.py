"""Assemble a runnable PYBOT from data: persona.json + right brain + cards.

This is the "one-click character" entry point. A persona directory holds
everything the engine needs:

```
personas/<id>/
  persona.json      baselines, taus, voice, switches, cubie triggers
  knowledge/*.jsonl knowledge cards the left brain may answer from
```

``build()`` turns that directory into a ``MindEngine``. Nothing in here is
persona-specific: adding a character means adding a directory, not code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind.config import PersonaConfig  # noqa: E402
from omega_mind.engine import MindEngine  # noqa: E402
from omega_mind.feynman_kb import FeynmanKB  # noqa: E402
from omega_mind.pybot_bridge import PyBotBridge  # noqa: E402

from bridge import LeftResponder, Persona, PersonaError, load  # noqa: E402

DEFAULT_PERSONAS = ROOT / "personas"
STATE_DIR_ENV = "PYBOT_STATE_DIR"


def to_config(persona: Persona) -> PersonaConfig:
    """Map the data-only persona onto the right brain's PersonaConfig."""
    cfg = PersonaConfig(name=persona.id or "generic")
    cfg.baselines.update({k: float(v) for k, v in persona.baselines.items()})
    cfg.taus.update({k: float(v) for k, v in persona.taus.items()})
    voice = persona.voice
    for attr, key in (
        ("catchphrase_prefixes", "prefixes_zh"),
        ("ending_particles", "endings_zh"),
        ("catchphrase_prefixes_en", "prefixes_en"),
        ("ending_particles_en", "endings_en"),
    ):
        values = voice.get(key)
        if values:
            setattr(cfg, attr, list(values))
    switches = persona.switches
    cfg.correction_as_curiosity = bool(switches.get("correction_as_curiosity", False))
    cfg.jargon_aversion = bool(switches.get("jargon_aversion", False))
    try:
        cfg.politeness = float(switches.get("politeness", 1.0))
    except (TypeError, ValueError):
        cfg.politeness = 1.0
    cfg.disable_cubies = list(persona.disable_cubies)
    cfg.override_triggers = {k: list(v) for k, v in persona.override_triggers.items()}
    cfg.extra_effects = {k: dict(v) for k, v in persona.extra_effects.items()}
    return cfg


def knowledge_dir(persona: Persona) -> Optional[Path]:
    """First existing kb_dir that actually holds cards, else None."""
    for path in persona.kb_dirs():
        if path.is_dir() and any(path.glob("*.jsonl")):
            return path
    return None


def resolve_persona(target: Union[str, Path],
                    personas_root: Union[str, Path] = DEFAULT_PERSONAS) -> Path:
    """Accept an id ("feynman"), a directory, or the persona.json itself."""
    path = Path(target).expanduser()
    if path.is_file():
        return path
    if path.is_dir() and (path / "persona.json").is_file():
        return path / "persona.json"
    candidate = Path(personas_root) / str(target) / "persona.json"
    if candidate.is_file():
        return candidate
    raise PersonaError(f"no persona named {target!r} under {personas_root}")


def default_state_path(persona_id: str) -> Path:
    base = Path(os.getenv(STATE_DIR_ENV) or (Path.home() / ".feynman_pybot"))
    return base / f"{persona_id}.mind_state.json"


def build(target: Union[str, Path] = "feynman",
          language: Optional[str] = None,
          enable_bridge: Optional[bool] = None,
          state_path: Optional[Union[str, Path]] = None,
          personas_root: Union[str, Path] = DEFAULT_PERSONAS,
          clock=None) -> MindEngine:
    """Build a mind for one persona. ``enable_bridge=None`` follows the env flag."""
    persona = load(resolve_persona(target, personas_root))
    cfg = to_config(persona)
    kb_dir = knowledge_dir(persona)
    kb = (FeynmanKB(knowledge_dir=str(kb_dir)) if kb_dir is not None
          else FeynmanKB.disabled())
    responder = None
    if enable_bridge is None:
        enable_bridge = os.getenv("OMEGA_LEFT_BRIDGE") == "1"
    if enable_bridge:
        responder = LeftResponder(enabled=True)
    kwargs = dict(
        persona=cfg,
        bridge=PyBotBridge.disabled(),
        kb=kb,
        left_responder=responder,
        language=language or persona.language or "zh",
    )
    if clock is not None:
        kwargs["clock"] = clock
    if state_path is not None:
        kwargs["store_path"] = str(state_path)
    return MindEngine(**kwargs)
