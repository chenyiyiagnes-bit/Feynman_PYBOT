"""Export a PersonaConfig + the 26 cubies into a single persona.json.

This is the bridge between the code-first persona definitions and the
data-first format the released repo uses (``personas/<id>/persona.json``).

Usage:
    python3 tools/export_persona.py feynman personas/feynman/persona.json
    python3 tools/export_persona.py laozhuang personas/laozhuang/persona.json

The right brain is imported from ``QBOT_RIGHT_ROOT`` (default: ``right/``),
so this runs both before and after the module is vendored into the repo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _right_root() -> Path:
    env = os.environ.get("QBOT_RIGHT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return ROOT / "right"


def build(persona_id: str) -> dict:
    sys.path.insert(0, str(_right_root()))
    from omega_mind.config import get_persona
    from omega_mind.cubies import CUBIES

    p = get_persona(persona_id)
    return {
        "schema": 1,
        "id": p.name,
        "name": p.name,
        "description": "",
        "language": "zh",
        "baselines": dict(p.baselines),
        "taus": dict(p.taus),
        "voice": {
            "prefixes_zh": list(p.catchphrase_prefixes),
            "endings_zh": list(p.ending_particles),
            "prefixes_en": list(p.catchphrase_prefixes_en),
            "endings_en": list(p.ending_particles_en),
        },
        "switches": {
            "correction_as_curiosity": p.correction_as_curiosity,
            "jargon_aversion": p.jargon_aversion,
            "politeness": p.politeness,
        },
        "disable_cubies": list(p.disable_cubies),
        "override_triggers": {k: list(v) for k, v in p.override_triggers.items()},
        "extra_effects": {k: dict(v) for k, v in p.extra_effects.items()},
        "cubies": [
            {
                "name": c.name,
                "role": c.role,
                "face": c.face,
                "valence": c.valence,
                "triggers": [[pat, gain] for pat, gain in c.triggers],
                "effects": dict(c.effects),
                "note": c.note,
            }
            for c in CUBIES
        ],
        "knowledge": {
            "kb_dirs": ["knowledge"],
            "corpus": None,
            "corpus_note": "Bring your own corpus. Nothing copyrighted ships with this repo.",
        },
        "license": {"persona": "CC-BY-4.0", "code": "MIT"},
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    persona_id, out = argv[1], Path(argv[2])
    if not out.is_absolute():
        out = ROOT / out
    data = build(persona_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"wrote {out} ({len(data['cubies'])} cubies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
