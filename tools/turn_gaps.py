"""Export what the cards could not answer, as a mining queue.

    python3 tools/turn_gaps.py                     # every package, table
    python3 tools/turn_gaps.py --json gaps.jsonl   # one question per line
    python3 tools/turn_gaps.py --package feynman-zh --promote

Every turn is judged when it ends (``omega_mind/reviewer.py``) and failures
are appended to ``<state>/<id>/kb_learned/turn_gaps.jsonl``. This tool reads
those trails across packages and emits one queue the offline mining pass can
consume. It does not answer the questions and does not touch cards.

``--promote`` queues questions seen at least ``REPEAT_TO_QUEUE`` times as
``gap`` candidates in each package's review journal. A candidate is still not
a decision: a human approves or revokes it (``agent/review``), so repetition
only earns a hearing.

Reading the trail proves nothing about why a question missed; pair each row
with the nearest card id and score it records. Whether the nearest card was
close on purpose is a judgement for the mining pass, not for this counter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "right"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from omega_mind import paths, reviewer  # noqa: E402
from omega_mind.learner import LearnedStore  # noqa: E402


def packages(base: Path) -> List[str]:
    directory = base / "personas"
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def collect(only: str = "") -> List[dict]:
    rows: List[dict] = []
    for name in packages(ROOT):
        if only and name != only:
            continue
        store = LearnedStore(paths.learned_dir_for(name))
        for item in reviewer.summary(store):
            rows.append({"package": name, **item})
    rows.sort(key=lambda r: (r["package"], -r["count"]))
    return rows


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", default="", help="only this package")
    parser.add_argument("--json", default="", help="write JSONL here (default: stdout table)")
    parser.add_argument("--promote", action="store_true",
                        help="queue repeated questions as review candidates")
    args = parser.parse_args(argv)

    rows = collect(args.package)
    if not rows:
        print("no failed turns recorded", file=sys.stderr)
        return 0

    if args.promote:
        for name in sorted({row["package"] for row in rows}):
            if args.package and name != args.package:
                continue
            store = LearnedStore(paths.learned_dir_for(name))
            queued = reviewer.queue_repeats(store)
            print(f"{name}: queued {len(queued)} candidate(s)", file=sys.stderr)

    payload = [{key: row[key] for key in
                ("package", "text", "language", "count", "reasons",
                 "near", "first_turn", "last_ts")}
               for row in rows]
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            for row in payload:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(payload)} question(s) to {args.json}", file=sys.stderr)
        return 0

    width = max(len(row["text"]) for row in rows)
    width = min(max(width, 8), 40)
    for row in payload:
        reasons = ",".join(row["reasons"])
        near = row["near"] or "-"
        print(f"{row['package']:14} {row['count']:3}  {reasons:22} "
              f"{row['text'][:width]:{width}}  near={near}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
