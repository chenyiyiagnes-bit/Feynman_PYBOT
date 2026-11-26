"""Preflight check: can this repo run right now?

    python3 tools/doctor.py
    python3 tools/doctor.py --persona laozhuang

Prints PASS/FAIL/WARN for the things that actually break a first run:
Python version, persona files, cards, the typesetter bridge (optional),
and the test count. Exit code 0 when nothing failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime import build, knowledge_dir, resolve_persona  # noqa: E402
from bridge import discover, load  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    want = args[args.index("--persona") + 1] if "--persona" in args else "feynman-zh"
    rows = []

    def add(status, label, detail=""):
        rows.append((status, label, detail))

    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        add(PASS, "Python", f"{major}.{minor}")
    else:
        add(FAIL, "Python", f"{major}.{minor} — need 3.10+")

    for rid, path in discover(ROOT / "personas").items():
        try:
            persona = load(path)
            kdir = knowledge_dir(persona)
            cards = len(list(kdir.glob("*.jsonl"))) if kdir else 0
            add(PASS, f"persona {rid}", f"{len(persona.cubies)} cubies, {cards} card files")
            if cards == 0:
                add(WARN, f"cards {rid}", "no knowledge cards — only persona voice replies")
            corpus = persona.corpus_path()
            if corpus is not None and not corpus.exists():
                add(WARN, f"corpus {rid}", f"missing: {corpus}")
        except Exception as exc:  # noqa: BLE001
            add(FAIL, f"persona {rid}", str(exc))

    try:
        path = resolve_persona(want)
        engine = build(want, enable_bridge=False)
        add(PASS, "build", f"{path.parent.name}: {engine.kb.mode}")
        if engine.kb.mode.startswith("feynman_kb") and not engine.kb.docs:
            add(WARN, "cards", "KB loaded but empty")
    except Exception as exc:  # noqa: BLE001
        add(FAIL, "build", f"{type(exc).__name__}: {exc}")

    try:
        from bridge.responder import LeftResponder
        r = LeftResponder()
        if not r.enabled:
            add(WARN, "left bridge", "off (OMEGA_LEFT_BRIDGE=1 to enable)")
        elif r.available():
            add(PASS, "left bridge", f"vault {r.vault_path.name}")
        else:
            add(WARN, "left bridge", r.last_error or "unavailable")
    except Exception as exc:  # noqa: BLE001
        add(WARN, "left bridge", f"{type(exc).__name__}: {exc}")

    width = max(len(label) for _, label, _ in rows)
    for status, label, detail in rows:
        print(f"{status:4}  {label:<{width}}  {detail}")
    failed = sum(1 for status, _, _ in rows if status == FAIL)
    print(f"\n{len(rows) - failed}/{len(rows)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
