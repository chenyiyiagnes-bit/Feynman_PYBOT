"""Where a mind's mutable state lives.

One rule for every entry point: what a mind learns is *derived* data, so it
lives under the state directory, never inside the package (the repo is
public and ``*.jsonl`` under ``personas/*/data`` must stay untracked).
``app.runtime``, the CLI and ``FeynmanKB`` all resolve the same path here,
which is what makes teaching in one entry point visible in the other.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Overrides ``~/.pybot``; used by tests and by deployments that keep state
#: outside the home directory.
STATE_DIR_ENV = "PYBOT_STATE_DIR"


def state_base() -> Path:
    """Where every mind's state lives: ``$PYBOT_STATE_DIR`` or ``~/.pybot``.

    Before the rename the state sat in ``~/.feynman_pybot``. If the new
    directory does not exist yet and the old one does, keep using the old
    one — an upgrade must not look like a wipe. Nothing is moved or copied.
    """
    override = os.getenv(STATE_DIR_ENV)
    if override:
        return Path(override)
    home = Path.home()
    current = home / ".pybot"
    legacy = home / ".feynman_pybot"
    if not current.exists() and legacy.exists():
        return legacy
    return current


def learned_dir_for(persona_id: str) -> Path:
    """Learned overlay of one persona: ``<state>/<id>/kb_learned``.

    Two personas never share what one of them was taught.
    """
    return state_base() / str(persona_id) / "kb_learned"


def default_state_path(persona_id: str) -> Path:
    """Saved affect/relations/tiers of one persona."""
    return state_base() / f"{persona_id}.mind_state.json"
