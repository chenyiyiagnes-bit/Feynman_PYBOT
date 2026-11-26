"""Account for every card row: loaded, or dropped for a stated reason.

    python3 tools/card_coverage.py
    python3 tools/card_coverage.py --package feynman-zh --json out.json
    python3 tools/card_coverage.py --write-ledger data/card_ledger.json
    python3 tools/card_coverage.py --check-ledger data/card_ledger.json

The loader drops a row when it has no body, when an English package meets a
row that is not English, or when the line is not JSON. Each drop is a card
the mind will never answer from, and until now it happened silently. This
tool re-reads every row with the same rules the loader uses and names the
reason for each drop, so a broken card shows up as a number, not as a query
that mysteriously misses.

What this proves: every row either became a doc or was dropped for a reason
the tool can name, and (with a ledger) that no card changed behind the
ledger's back.

What this does NOT prove: that a loaded card answers the question it should
(that is ``tools/eval_segmentation.py`` and the golden set), that its text is
faithful to a source, or that its ``support`` quotes are real (that is
``tools/card_support.py``). Coverage is accounting, not quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "right"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from omega_mind import feynman_kb  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_sha(row: dict) -> str:
    return _sha(json.dumps(row, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")))


def survey_package(package: Path) -> dict:
    """Every row of one package, with the loader's decision and the reason."""
    knowledge = package / "knowledge"
    kb = feynman_kb.FeynmanKB(knowledge_dir=str(knowledge))
    language = kb.language
    rows = loaded = 0
    dropped: List[dict] = []
    cards: List[dict] = []
    for file in sorted(knowledge.glob("*.jsonl")):
        if file.name in feynman_kb.SKIP_FILES:
            continue
        for lineno, line in enumerate(
                file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                dropped.append({"file": file.name, "line": lineno,
                                "id": None, "reason": "bad-json",
                                "detail": str(error)})
                continue
            card_id = str(row.get("id") or f"{file.stem}:{lineno}")
            block = _block_id(package.name, file.stem, card_id)
            kind = str(row.get("type") or feynman_kb.STEM_KIND.get(
                file.stem, file.stem)).replace(" ", "_")
            reason = _drop_reason(kb, file.stem, row, kind)
            if reason is not None:
                dropped.append({"file": file.name, "line": lineno,
                                "id": card_id, "reason": reason})
                continue
            loaded += 1
            cards.append({"block_id": block, "id": card_id, "kind": kind,
                          "text_sha256": _row_sha(row),
                          "has_support": bool(row.get("support"))})
    in_kb = len([doc for doc in kb.docs if not doc.id.startswith("learned-")])
    return {
        "package": package.name,
        "language": language,
        "knowledge": str(knowledge.relative_to(ROOT))
        if knowledge.is_relative_to(ROOT) else str(knowledge),
        "rows": rows,
        "loaded": loaded,
        "in_kb": in_kb,
        "unaccounted": in_kb - loaded,
        "dropped": dropped,
        "cards": cards,
    }


def _block_id(package: str, stem: str, card_id: str) -> str:
    return f"{package}:{stem}:{card_id}"


def _drop_reason(kb, stem: str, row: dict, kind: str) -> Optional[str]:
    """The reason the loader would drop ``row``, or None if it keeps it."""
    if kb.language == "en" and not kb._already_english(row, kind):
        return "not-english"
    if kb._row_to_doc(stem, row, language=kb.language) is None:
        return "no-answer-text"
    return None


def survey(root: Optional[Path] = None,
           package: Optional[str] = None) -> dict:
    root = Path(root) if root is not None else feynman_kb.PACKAGES_ROOT
    report = {"root": str(root), "packages": {}}
    for manifest in sorted(root.glob("*/persona.json")):
        name = manifest.parent.name
        if package and name != package:
            continue
        if not (manifest.parent / "knowledge").is_dir():
            continue
        report["packages"][name] = survey_package(manifest.parent)
    return report


def render(report: dict) -> str:
    lines = []
    for name, row in report["packages"].items():
        status = (PASS if row["unaccounted"] == 0 else FAIL)
        lines.append(f"{status}  {name:<16} {row['language']}  "
                     f"rows {row['rows']:>4}  loaded {row['loaded']:>4}  "
                     f"dropped {len(row['dropped']):>3}  "
                     f"with-support {sum(1 for c in row['cards'] if c['has_support'])}")
        for drop in row["dropped"][:20]:
            lines.append(f"      drop {drop['reason']:<15} "
                         f"{drop['file']}:{drop['line']} {drop['id'] or ''}")
        if len(row["dropped"]) > 20:
            lines.append(f"      ... {len(row['dropped']) - 20} more drops")
        if row["unaccounted"]:
            lines.append(f"      UNACCOUNTED: KB holds {row['in_kb']} cards "
                         f"but {row['loaded']} rows were accounted for")
    return "\n".join(lines)


def write_ledger(report: dict, path: Path) -> None:
    ledger = {"packages": {}}
    for name, row in report["packages"].items():
        ledger["packages"][name] = {
            "language": row["language"],
            "cards": [{key: card[key] for key in
                       ("block_id", "id", "kind", "text_sha256")}
                      for card in row["cards"]],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def check_ledger(report: dict, path: Path) -> List[str]:
    """Cards whose identity changed since the ledger was written."""
    if not path.is_file():
        return [f"ledger not found: {path}"]
    ledger = json.loads(path.read_text(encoding="utf-8")).get("packages", {})
    problems: List[str] = []
    for name, row in report["packages"].items():
        recorded = {card["block_id"]: card["text_sha256"]
                    for card in ledger.get(name, {}).get("cards", [])}
        current = {card["block_id"]: card["text_sha256"]
                   for card in row["cards"]}
        for block in sorted(current.keys() - recorded.keys()):
            problems.append(f"{name}: added since ledger: {block}")
        for block in sorted(recorded.keys() - current.keys()):
            problems.append(f"{name}: in ledger but not loaded: {block}")
        for block in sorted(current.keys() & recorded.keys()):
            if current[block] != recorded[block]:
                problems.append(f"{name}: changed since ledger: {block}")
    for name in sorted(set(ledger) - set(report["packages"])):
        problems.append(f"{name}: package in ledger but not in this checkout")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="personas root (default: the repo's)")
    parser.add_argument("--package", help="only this package id")
    parser.add_argument("--json", metavar="PATH", help="write the full report")
    parser.add_argument("--write-ledger", metavar="PATH",
                        help="write card identities + hashes")
    parser.add_argument("--check-ledger", metavar="PATH",
                        help="fail when a card changed or vanished")
    args = parser.parse_args(argv)

    report = survey(root=args.root, package=args.package)
    print(render(report))
    if not report["packages"]:
        print("no packages with knowledge/ were found")
        return 1
    problems = []
    if any(row["unaccounted"] for row in report["packages"].values()):
        problems.append("unaccounted rows: the loader and the survey disagree")
    if any(drop["reason"] == "bad-json"
           for row in report["packages"].values() for drop in row["dropped"]):
        problems.append("a knowledge file has a line that is not JSON")
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    if args.write_ledger:
        write_ledger(report, Path(args.write_ledger))
        print(f"ledger written: {args.write_ledger}")
    if args.check_ledger:
        ledger_problems = check_ledger(report, Path(args.check_ledger))
        if ledger_problems:
            problems.extend(ledger_problems)
            for problem in ledger_problems:
                print(f"  ledger: {problem}")
        else:
            print(f"ledger matches: {args.check_ledger}")
    for problem in problems:
        print(f"FAIL: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
