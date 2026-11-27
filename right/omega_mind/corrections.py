"""Corrections: "that cut was wrong / that answer was wrong", with provenance.

The point of writing a correction down is not the apology — it is being able
to count what keeps going wrong and to replay the same question later against
a new build. Every record says what was produced, what should have been
produced, which card or rule was involved, and which lexicon version was
active when it happened, so a fix can be checked rather than believed.

The log is one JSON object per line under the state directory (never in git):
    {"kind": "segmentation", "query": "...", "produced": [...],
     "expected": [...], "source": "词表", "lexicon": "0a1b...", "ts": "..."}

This is a manual utility: nothing in the pipeline calls ``record()`` on its
own. When a cut or an answer is wrong, someone records it by hand (or from a
script they write); ``summary()``/``unresolved()``/``replay()`` then work on
whatever has accumulated. Hooking it to an automatic producer is deliberately
left open — an automatic "wrong answer" signal would need a source of truth
this repo does not have.
"""

from __future__ import annotations

import datetime
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

KINDS = ("segmentation", "answer")
ENV = "PYBOT_STATE_DIR"
FILE = "corrections.jsonl"
ROOT = Path(__file__).resolve().parents[2]


def state_dir(path: Optional[str | Path] = None) -> Path:
    if path:
        return Path(path).expanduser()
    env = os.getenv(ENV)
    return Path(env).expanduser() if env else ROOT / "data" / "state"


def log_path(path: Optional[str | Path] = None) -> Path:
    return state_dir(path) / FILE


def record(kind: str, query: str, produced, expected,
           source: str = "", note: str = "", lexicon: str = "",
           path: Optional[str | Path] = None,
           now: Optional[str] = None) -> Dict:
    """Append one correction. ``produced``/``expected`` are strings or lists."""
    if kind not in KINDS:
        raise ValueError(f"unknown correction kind: {kind!r} (use one of {KINDS})")
    row = {
        "kind": kind,
        "query": str(query),
        "produced": list(produced) if isinstance(produced, (list, tuple)) else str(produced),
        "expected": list(expected) if isinstance(expected, (list, tuple)) else str(expected),
        "source": str(source),
        "note": str(note),
        "lexicon": str(lexicon),
        "ts": now or datetime.datetime.now(datetime.timezone.utc)
              .replace(microsecond=0).isoformat(),
    }
    target = log_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def replay(path: Optional[str | Path] = None) -> List[Dict]:
    """Every recorded correction, oldest first."""
    target = log_path(path)
    if not target.is_file():
        return []
    return [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summary(path: Optional[str | Path] = None) -> Dict:
    rows = replay(path)
    by_kind = Counter(row["kind"] for row in rows)
    by_source = Counter(row["source"] for row in rows if row["source"])
    by_query = Counter(row["query"] for row in rows)
    return {
        "total": len(rows),
        "by_kind": dict(sorted(by_kind.items())),
        "by_source": dict(sorted(by_source.items())),
        # The question that keeps being got wrong is the one worth fixing.
        "repeat_offenders": [[query, count] for query, count in
                             sorted(by_query.items(), key=lambda kv: (-kv[1], kv[0]))
                             if count > 1][:10],
        "path": str(log_path(path)),
    }


def unresolved(rows: Iterable[Dict], kind: Optional[str] = None,
               lexicon: Optional[str] = None) -> List[Dict]:
    """Records made under a different build — the ones nobody has checked since."""
    return [row for row in rows
            if (kind is None or row["kind"] == kind)
            and (lexicon is None or row["lexicon"] != lexicon)]


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", help="state directory (default $PYBOT_STATE_DIR)")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="record a correction")
    add.add_argument("kind", choices=KINDS)
    add.add_argument("query")
    add.add_argument("--produced", required=True)
    add.add_argument("--expected", required=True)
    add.add_argument("--source", default="")
    add.add_argument("--note", default="")
    add.add_argument("--lexicon", default="")
    sub.add_parser("summary", help="counts by kind and source")
    args = parser.parse_args(argv)

    if args.command == "add":
        row = record(args.kind, args.query,
                     args.produced.split("|") if "|" in args.produced else args.produced,
                     args.expected.split("|") if "|" in args.expected else args.expected,
                     source=args.source, note=args.note, lexicon=args.lexicon,
                     path=args.state)
        print(f"recorded {row['kind']}: {row['query']} -> {log_path(args.state)}")
        return 0
    print(json.dumps(summary(args.state), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
