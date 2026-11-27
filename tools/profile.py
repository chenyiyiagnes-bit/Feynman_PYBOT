"""Show what a mind remembers about each person, read-only.

    python3 tools/profile.py                       # every persona with a save
    python3 tools/profile.py --persona feynman-zh
    python3 tools/profile.py --persona feynman-zh --user 小明
    python3 tools/profile.py --json                # JSONL, one fact per line

The per-user profile already exists in the state file (``memory.py``:
``semantic`` keyed by ``kind:user``, with the quote and turn it came from,
plus retractions in ``withdrawn``). What was missing is a way for the person
it describes to read it: Hermes' ``USER.md`` is a file you can open.

This tool only reads the save. It never edits, never infers, and never
writes: a fact is changed by saying so in conversation (``我不再住在…``), so
the quote that justifies it stays attached.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "right"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from omega_mind import paths  # noqa: E402

KINDS = ("name", "likes", "dislikes", "lives_in", "work")


def personas_with_state() -> list:
    base = paths.state_base()
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob("*.mind_state.json")):
        out.append(path.name[: -len(".mind_state.json")])
    return out


def facts(entry: dict, persona: str) -> list:
    """One persona's semantic plus withdrawn facts, as plain rows."""
    tiers = entry.get("tiers") or {}
    rows = []
    for key, value in sorted((tiers.get("semantic") or {}).items()):
        kind, _, user = key.partition(":")
        rows.append({"persona": persona, "user": user, "kind": kind,
                     "value": value.get("value", ""),
                     "raw": value.get("raw", ""),
                     "quote": value.get("quote", ""),
                     "turn": value.get("source_turn"),
                     "state": "current"})
    for key, value in sorted((tiers.get("withdrawn") or {}).items()):
        kind, _, user = key.partition(":")
        rows.append({"persona": persona, "user": user, "kind": kind,
                     "value": "", "raw": value.get("raw", ""),
                     "quote": value.get("quote", ""),
                     "turn": value.get("source_turn"),
                     "state": "withdrawn"})
    return rows


def collect(only: str = "", user: str = "") -> list:
    rows = []
    for persona in personas_with_state():
        if only and persona != only:
            continue
        path = paths.default_state_path(persona)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in facts(entry, persona):
            if user and row["user"] != user:
                continue
            rows.append(row)
    order = {kind: i for i, kind in enumerate(KINDS)}
    rows.sort(key=lambda r: (r["persona"], r["user"],
                             order.get(r["kind"], len(order))))
    return rows


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persona", default="", help="only this persona")
    parser.add_argument("--user", default="", help="only this interlocutor")
    parser.add_argument("--json", action="store_true",
                        help="one fact per line, JSONL")
    args = parser.parse_args(argv)

    rows = collect(args.persona, args.user)
    if not rows:
        print("nothing remembered yet", file=sys.stderr)
        return 0
    if args.json:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))
        return 0
    for row in rows:
        note = "" if row["state"] == "current" else "  (withdrawn)"
        print(f"{row['persona']:14} {row['user']:10} {row['kind']:9} "
              f"{row['raw']}  —turn {row['turn']}{note}")
        if row["quote"]:
            print(f"{'':25} “{row['quote']}”")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
