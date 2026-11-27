"""Log-only style audit at the reply exit.

``bridge.style_gate.check()`` measures one reply against the band of the
package it came from. This module puts that measurement on the reply exit of
an engine built by ``app.runtime.build()``:

    PYBOT_STYLE_GATE=1 python3 ...

One JSON line per turn lands in ``<state>/style_audit/<persona>.jsonl``
(``PYBOT_STYLE_AUDIT_DIR`` overrides). The reply is never changed and never
blocked: any failure inside the audit is swallowed, because a measurement
must not cost the user an answer.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "right")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bridge import style_gate as gate  # noqa: E402
from omega_mind.paths import state_base  # noqa: E402

#: ``1`` turns the audit on; anything else leaves the engine untouched.
FLAG_ENV = "PYBOT_STYLE_GATE"
#: Where the JSONL logs go; default ``<state>/style_audit``.
DIR_ENV = "PYBOT_STYLE_AUDIT_DIR"

#: How much of the reply to keep for diagnosis. The log lives in the state
#: directory (private), but a short head is enough to find a sample again.
HEAD_CHARS = 160


def enabled() -> bool:
    return os.getenv(FLAG_ENV) == "1"


def audit_dir() -> Path:
    override = os.getenv(DIR_ENV)
    return Path(override).expanduser() if override else state_base() / "style_audit"


def band_for_kb(kb, language: str) -> Optional[dict]:
    """Package band from the KB's composed card answers, or None."""
    docs = getattr(kb, "docs", None) or []
    if not docs:
        return None
    return gate.band_for_docs(language, docs)["overall"]


def install(engine, persona_id: str, language: str, kb,
            directory: Optional[Path] = None) -> Optional[Path]:
    """Wrap ``engine.turn`` so every reply is measured and logged.

    Returns the log path, or None when there is nothing to measure
    (no cards). The wrapped ``turn`` returns exactly what the original did.
    """
    band = band_for_kb(kb, language)
    if band is None:
        return None
    directory = Path(directory) if directory is not None else audit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{persona_id}.jsonl"
    original = engine.turn

    @functools.wraps(original)
    def turn(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            reply = getattr(result, "reply", "") or ""
            source = getattr(result, "source", "") or ""
            # The aside is scaffolding appended after a single newline; the
            # band was built from card answers, so measure the body alone.
            body, appendix = gate.split_appendix(reply)
            verdict = gate.check(body or reply, band)
            # An abstention is the persona's own escape hatch, not its voice:
            # "I'm not confident about that one" is short by design, so the
            # length band does not apply. Language and cliché still do, and
            # the record says why the band was skipped.
            exempt = source.startswith("fallback:")
            if exempt:
                verdict["checks"].pop("mean_in_band", None)
                verdict["checks"].pop("no_runaway_sentence", None)
                verdict["ok"] = all(verdict["checks"].values())
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "persona": persona_id,
                "language": language,
                "turn": getattr(engine, "turn_index", None),
                "ok": verdict["ok"],
                "exempt": "fallback" if exempt else "",
                "failed": [k for k, v in verdict["checks"].items() if not v],
                "metrics": verdict["metrics"],
                "source": source,
                "chars": len(reply),
                "appendix_chars": len(appendix),
                "head": reply[:HEAD_CHARS],
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return result

    engine.turn = turn
    return path
