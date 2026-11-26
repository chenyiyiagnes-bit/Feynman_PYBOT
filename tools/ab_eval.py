"""Blind pairwise comparison of two personas' answers.

    python3 tools/ab_eval.py make --a feynman-zh --b feynman-zh-v2 \\
        --questions q.jsonl --pairs pairs.jsonl --key key.json --seed 7
    python3 tools/ab_eval.py score --pairs pairs.jsonl --key key.json \\
        --votes votes.jsonl

``make`` asks both packages the same questions and writes each pair as
``left``/``right`` with the order shuffled per pair. The answer file never
names a package: whoever judges reads only the two replies. ``key.json``
maps each pair back to its packages and is written separately, so the judge
is not given it.

Questions are one JSON object per line: ``{"id": "q1", "question": "..."}``.
Votes are ``{"id": "q1", "pick": "left" | "right" | "tie"}``.

What this measures: a reader's preference between two answers on the same
questions. What it does not measure: factual correctness (a confident wrong
answer can win), and with few questions a win rate is noise — record the
count, not just the winner.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "right"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from omega_mind import feynman_kb  # noqa: E402

NO_ANSWER = "(no answer)"


def _read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _answer(package_dir: Path, question: str) -> str:
    kb = feynman_kb.FeynmanKB(knowledge_dir=str(package_dir / "knowledge"))
    text, _source = kb.answer(question)
    return text if text else NO_ANSWER


def make(root: Path, a: str, b: str, questions: Path, pairs: Path,
         key: Path, seed: int) -> int:
    root = Path(root)
    missing = [name for name in (a, b)
               if not (root / name / "persona.json").is_file()]
    if missing:
        print(f"package not found under {root}: {', '.join(missing)}")
        return 2
    rng = random.Random(seed)
    pair_rows: List[dict] = []
    key_data: Dict[str, object] = {"_packages": {"a": a, "b": b}}
    for row in _read_jsonl(questions):
        qid = str(row["id"])
        question = str(row["question"])
        answer_a = _answer(root / a, question)
        answer_b = _answer(root / b, question)
        a_left = rng.random() < 0.5
        left, right = (answer_a, answer_b) if a_left else (answer_b, answer_a)
        pair_rows.append({"id": qid, "question": question,
                          "left": left, "right": right})
        key_data[qid] = {"a": "left" if a_left else "right",
                         "b": "right" if a_left else "left"}
    pairs.write_text("\n".join(json.dumps(row, ensure_ascii=False)
                               for row in pair_rows) + "\n", encoding="utf-8")
    key.write_text(json.dumps(key_data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"{len(pair_rows)} pair(s) written to {pairs}; key kept in {key}")
    return 0


def score(pairs: List[dict], key: dict, votes: List[dict]) -> dict:
    packages = key.get("_packages", {"a": "a", "b": "b"})
    wins: Dict[str, int] = {packages["a"]: 0, packages["b"]: 0}
    ties = missing = 0
    known = {str(row["id"]) for row in pairs}
    for vote in votes:
        qid = str(vote.get("id"))
        if qid not in known or qid not in key:
            missing += 1
            continue
        pick = vote.get("pick")
        if pick == "tie":
            ties += 1
        elif pick in ("left", "right"):
            side = pick
            winner = ("a" if key[qid]["a"] == side else "b")
            wins[packages[winner]] += 1
        else:
            missing += 1
    return {"wins": wins, "ties": ties, "invalid": missing,
            "votes": len(votes)}


def render(report: dict) -> str:
    lines = []
    for name, count in report["wins"].items():
        lines.append(f"{name:<20} {count:>4}")
    lines.append(f"{'ties':<20} {report['ties']:>4}")
    if report["invalid"]:
        lines.append(f"{'invalid votes':<20} {report['invalid']:>4}")
    decided = sum(report["wins"].values())
    if decided:
        for name, count in report["wins"].items():
            lines.append(f"{name:<20} {count / decided:>6.0%} of decided")
    lines.append(f"({decided} decided, {report['votes']} votes — a few "
                 f"questions are noise, not a verdict)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    maker = sub.add_parser("make", help="answer questions from two packages")
    maker.add_argument("--root", default=str(feynman_kb.PACKAGES_ROOT))
    maker.add_argument("--a", required=True, help="scenario A package id")
    maker.add_argument("--b", required=True, help="scenario B package id")
    maker.add_argument("--questions", required=True)
    maker.add_argument("--pairs", required=True)
    maker.add_argument("--key", required=True)
    maker.add_argument("--seed", type=int, default=0,
                       help="same seed, same left/right order")

    scorer = sub.add_parser("score", help="count wins from blind votes")
    scorer.add_argument("--pairs", required=True)
    scorer.add_argument("--key", required=True)
    scorer.add_argument("--votes", required=True)

    args = parser.parse_args(argv)
    if args.command == "make":
        return make(Path(args.root), args.a, args.b, Path(args.questions),
                    Path(args.pairs), Path(args.key), args.seed)
    if not Path(args.votes).is_file():
        print(f"votes file not found: {args.votes}")
        return 2
    report = score(pairs=_read_jsonl(Path(args.pairs)),
                   key=json.loads(Path(args.key).read_text(encoding="utf-8")),
                   votes=_read_jsonl(Path(args.votes)))
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
