"""Replay frozen dialogue scenarios with isolated state and report behavioral metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.runtime import build
from tools.ab_eval import POSTURES, fingerprint, transcript

DEFAULT_SET = ROOT / "tests" / "data" / "dialogue_holdout_v1.json"


def diagnostics(entry: dict, result: dict) -> dict:
    data = result["retrieval"]
    prefix, _, cid = result["source"].partition(":")
    expected = set(entry.get("cards", ()))
    stages = {name: bool(expected.intersection(ids))
              for name, ids in data.get("candidates", {}).items()} if expected else {}
    outcomes = {}
    if expected:
        outcomes["selection"] = cid in expected
    if "source" in entry:
        outcomes["source"] = result["source"].startswith(entry["source"])
    if entry.get("abstain"):
        outcomes["abstention"] = prefix in ("fallback", "clarification")
    if "contains" in entry or "excludes" in entry:
        normalized = result["reply"].casefold()
        outcomes["reply"] = (
            all(word.casefold() in normalized for word in entry.get("contains", []))
            and all(word.casefold() not in normalized for word in entry.get("excludes", [])))
    if "action" in entry:
        outcomes["action"] = data.get("dialogue_act") == entry["action"]
    return {"checks": outcomes, "recall": stages}


def run(cases_file: Path, personas: Path, mode: str, posture: str,
        confirmation_packages: dict | None = None) -> dict:
    inputs = cases_file.read_bytes()
    suite = json.loads(inputs)
    if confirmation_packages is not None:
        suite = {"cases": [
            {"id": f"confirm-{i:02}", "package": confirmation_packages[case["language"]],
             "turns": [{"text": case["query"], "cards": case["acceptable"]}]}
            for i, case in enumerate(suite["cases"], 1)]}
    results = []
    metadata = {}
    tags = defaultdict(lambda: [0, 0])
    times = []
    with tempfile.TemporaryDirectory(prefix="pybot-dialogue-eval-") as scratch:
        for index, case in enumerate(suite["cases"]):
            package = case["package"]
            env = {"PYBOT_STATE_DIR": str(Path(scratch) / str(index)),
                   "OMEGA_RETRIEVAL": mode, "OMEGA_STRUCTURED_EVIDENCE": "1",
                   "OMEGA_POSTURE": posture, "OMEGA_REVIEW_LEARNING": "1"}
            with patch.dict("os.environ", env):
                engine = build(package, personas_root=personas, enable_bridge=True,
                               clock=lambda: 1000.0)
                outputs = transcript(engine, case["turns"])
            if package not in metadata:
                metadata[package] = fingerprint(personas / package)
            for turn, (entry, output) in enumerate(zip(case["turns"], outputs), 1):
                times.append(output["ms"])
                diag = diagnostics(entry if isinstance(entry, dict) else {}, output)
                for group in diag.values():
                    for name, ok in group.items():
                        tags[name][0] += int(ok)
                        tags[name][1] += 1
                results.append({"id": case["id"], "turn": turn, **diag, **output})
    times.sort()
    return {"set": hashlib.sha256(inputs).hexdigest(),
            "code": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                   check=True, capture_output=True, text=True).stdout.strip(),
            "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                         check=True, capture_output=True, text=True).stdout.strip()),
            "python": sys.version.split()[0], "mode": mode, "posture": posture,
            "packages": metadata, "metrics": dict(tags),
            "p95_ms": round(times[max(0, math.ceil(.95 * len(times)) - 1)], 3) if times else None,
            "turns": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--questions", type=Path, default=DEFAULT_SET)
    inputs.add_argument("--confirm-set", type=Path)
    parser.add_argument("--confirmation-packages", nargs=2, metavar=("ZH", "EN"))
    parser.add_argument("--personas", type=Path, default=ROOT / "personas")
    parser.add_argument("--mode", choices=("legacy", "shadow", "rrf"), default="shadow")
    parser.add_argument("--posture", choices=POSTURES, default="affect_cube")
    parser.add_argument("--full", type=Path)
    parser.add_argument("--strict", action="store_true", help="fail on scenario mismatches")
    args = parser.parse_args(argv)
    if bool(args.confirm_set) != bool(args.confirmation_packages):
        parser.error("--confirm-set and --confirmation-packages must be used together")
    packages = dict(zip(("zh", "en"), args.confirmation_packages)) if args.confirm_set else None
    report = run(args.confirm_set or args.questions, args.personas, args.mode, args.posture,
                 confirmation_packages=packages)
    summary = {key: value for key, value in report.items() if key != "turns"}
    summary["failed"] = [(r["id"], r["turn"], k) for r in report["turns"]
                         for k, ok in r["checks"].items() if not ok]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.full:
        args.full.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return int(args.strict and bool(summary["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
