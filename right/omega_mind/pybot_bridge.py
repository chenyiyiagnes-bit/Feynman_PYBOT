"""Adapter for the real huggingface/pybot runtime.

Verified interface (2026-11):
    from bot.runtime import turn
    result = turn(text, memory)   # dict with at least "reply"

There is no PyBot class and no ``propose`` hook: modules expose
``observe`` / ``respond`` / ``after``. A subprocess fallback is deliberately
NOT used, because ``python -m pybot -p`` starts a fresh process every call and
silently loses multi-turn memory.

If PyBot is not installed, ``available()`` is False and the engine falls back
to its built-in knowledge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Where a local checkout of huggingface/pybot may live, in priority order.
# ``OMEGA_PYBOT_PATH`` wins; the rest are this project's own vendor dir.
_CANDIDATE_PATHS = [
    Path(__file__).resolve().parents[1] / "vendor" / "pybot",
]


def _local_checkout() -> Optional[Path]:
    env = os.environ.get("OMEGA_PYBOT_PATH")
    candidates = [Path(env)] if env else []
    candidates += _CANDIDATE_PATHS
    for path in candidates:
        if (path / "bot" / "runtime.py").is_file():
            return path
    return None


class PyBotBridge:
    # PyBot runs every module's regex over the raw text; past a few thousand
    # characters some patterns backtrack badly, so long inputs are truncated.
    MAX_INPUT = 2000

    def __init__(self, runtime: Any = None):
        self._runtime = runtime
        self._loaded = runtime is not None
        self.memory: Dict[str, Any] = {}
        self.last_module: Optional[str] = None
        self.mode = "injected" if runtime is not None else "unloaded"

    @classmethod
    def disabled(cls) -> "PyBotBridge":
        """A bridge that never imports PyBot. Used by tests for determinism."""
        bridge = cls(runtime=None)
        bridge._loaded = True
        bridge.mode = "disabled"
        return bridge

    def _load(self) -> bool:
        if not self._loaded:
            self._loaded = True
            return self._try_import()
        return self._runtime is not None

    def _try_import(self) -> bool:
        try:
            from bot import runtime  # type: ignore
        except Exception:
            path = _local_checkout()
            if path is None:
                self.mode = "unavailable"
                return False
            sys.path.insert(0, str(path))
            try:
                from bot import runtime  # type: ignore
            except Exception:
                self.mode = "unavailable"
                return False
        self._runtime = runtime
        self.mode = "bot.runtime"
        return True

    def available(self) -> bool:
        return self._load()

    def reply(self, text: str) -> Tuple[Optional[str], str]:
        """Return (base_reply, source). base_reply is None when unavailable."""
        if not self.available():
            return None, "unavailable"
        runtime = self._runtime
        if len(text) > self.MAX_INPUT:
            text = text[: self.MAX_INPUT]
        for name in ("turn", "reply"):
            fn = getattr(runtime, name, None)
            if callable(fn):
                try:
                    result = fn(text, self.memory)
                except Exception as exc:  # never take the mind down with it
                    return None, f"pybot_error:{type(exc).__name__}"
                if isinstance(result, dict):
                    self.last_module = result.get("module")
                    reply = result.get("reply") or result.get("text")
                    if reply:
                        return str(reply), f"pybot:{name}"
                elif isinstance(result, str) and result:
                    return result, f"pybot:{name}"
                return None, f"pybot:{name}:empty"
        return None, "pybot:no_entry"

    def reset(self) -> None:
        self.memory.clear()
        self.last_module = None
