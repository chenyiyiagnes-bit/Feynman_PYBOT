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

import dataclasses
import os
import sys
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind.config import (  # noqa: E402
    LEGACY_PERSONA_IDS,
    PersonaConfig,
    persona_from_data,
)
from omega_mind.engine import MindEngine  # noqa: E402
from omega_mind.feynman_kb import FeynmanKB  # noqa: E402
from omega_mind.paths import (  # noqa: E402
    STATE_DIR_ENV,
    default_state_path,
    learned_dir_for,
    state_base,
)
from omega_mind.pybot_bridge import PyBotBridge  # noqa: E402

from bridge import LeftResponder, Persona, PersonaError, load  # noqa: E402

from . import style_audit  # noqa: E402

DEFAULT_PERSONAS = ROOT / "personas"

#: The state helpers are defined once in ``omega_mind.paths`` so the CLI can
#: share them without importing this module (and its left-brain dependency).
__all__ = [
    "DEFAULT_PERSONAS", "STATE_DIR_ENV", "build", "default_state_path",
    "learned_dir_for", "resolve_persona", "state_base", "to_config",
]


def to_config(persona: Persona) -> PersonaConfig:
    """Map the data-only persona onto the right brain's PersonaConfig."""
    return persona_from_data(dataclasses.asdict(persona))


def knowledge_dir(persona: Persona) -> Optional[Path]:
    """First existing kb_dir that actually holds cards, else None."""
    for path in persona.kb_dirs():
        if path.is_dir() and any(path.glob("*.jsonl")):
            return path
    return None


def resolve_persona(target: Union[str, Path],
                    personas_root: Union[str, Path] = DEFAULT_PERSONAS) -> Path:
    """Accept an id ("feynman-zh"), a directory, or the persona.json itself.

    Legacy ids ("feynman") resolve to the package they used to mean — the
    same alias the right brain's ``get_persona`` applies, so a name does not
    mean two different things on two sides. ``"generic"`` is not a package:
    it is the right brain's card-less persona, and it is refused here with a
    message that says where it does belong.
    """
    name = LEGACY_PERSONA_IDS.get(str(target), target)
    if str(name) == "generic":
        raise PersonaError(
            "'generic' is not a persona package; it has no knowledge cards. "
            "Use MindEngine(persona=PersonaConfig()) directly, or name a "
            "package under personas/.")
    path = Path(name).expanduser()
    if path.is_file():
        return path
    if path.is_dir() and (path / "persona.json").is_file():
        return path / "persona.json"
    candidate = Path(personas_root) / str(name) / "persona.json"
    if candidate.is_file():
        return candidate
    raise PersonaError(f"no persona named {target!r} under {personas_root}")


def assoc_path_for(persona: Persona) -> Path:
    """Association edges are judgement about *this* persona's cards."""
    return (persona.root / "data" / "assoc" / "assoc_edges.jsonl").resolve()


def build(target: Union[str, Path],
          language: Optional[str] = None,
          enable_bridge: Optional[bool] = None,
          state_path: Optional[Union[str, Path]] = None,
          personas_root: Union[str, Path] = DEFAULT_PERSONAS,
          clock=None) -> MindEngine:
    """Build a mind for one persona.

    The package is never defaulted: the skeleton names no persona, so every
    caller says which package it wants. ``enable_bridge=None`` follows the
    env flag.
    """
    persona = load(resolve_persona(target, personas_root))
    package_language = persona.language or "zh"
    if language is not None and language != package_language:
        raise PersonaError(
            f"persona {persona.id!r} is a {package_language!r} package; "
            f"it cannot answer in {language!r}. Build the other package.")
    cfg = to_config(persona)
    kb_dir = knowledge_dir(persona)
    lexicon_dir = persona.root / "data" / "lexicon"
    kb = (FeynmanKB(knowledge_dir=str(kb_dir),
                    learned_dir=str(learned_dir_for(persona.id)),
                    assoc_path=str(assoc_path_for(persona)),
                    language=package_language,
                    profile=persona.language_profile,
                    lexicon_dir=(str(lexicon_dir) if persona.recall_expansion else None),
                    source_prefix=persona.id)
          if kb_dir is not None else FeynmanKB.disabled())
    responder = None
    if enable_bridge is None:
        enable_bridge = os.getenv("OMEGA_LEFT_BRIDGE") == "1"
    if enable_bridge:
        responder = LeftResponder(enabled=True,
                                  lexicon_dir=persona.root / "data" / "lexicon")
    kwargs = dict(
        persona=cfg,
        bridge=PyBotBridge.disabled(),
        kb=kb,
        language=package_language,
    )
    if clock is not None:
        kwargs["clock"] = clock
    state_file = Path(state_path).expanduser() if state_path is not None else None
    if state_file is not None and state_file.is_file():
        # Resume: MindEngine does not auto-load in __init__, and the loaded
        # engine must keep the same kb/persona it was built with here.
        engine = MindEngine.load(str(state_file), **kwargs)
        engine.store_path = state_file
        engine.auto_save = True
    else:
        engine = MindEngine(store_path=str(state_file) if state_file else None,
                            **kwargs)
    # load() predates the bridge, so attach the responder after the fact.
    engine.left_responder = responder
    if style_audit.enabled():
        # Log-only: measures each reply against the package's own band and
        # writes a JSONL line under the state directory. Never blocks.
        style_audit.install(engine, persona.id, package_language, kb)
    return engine
