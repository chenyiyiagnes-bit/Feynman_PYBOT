"""Check where a card's claim comes from: the ``support`` block or ``source``.

A card records provenance in one of two shapes. A *cite* names the source
without quoting it — the only honest option for a rewrite:

    "support": [{"source": "FLP I-1-1"}]

A *quote* also carries the span, hashed, so the text can be re-checked:

    "support": [
      {"source": "FLP I-1-1",
       "quote": "先看它是什么样、怎么动。",
       "source_file": "reference/flp-i.txt",
       "quote_sha256": "<sha256 of the quote, utf-8>"}
    ]

The Chinese packages put the same claim in a top-level ``source`` string;
that counts as a one-entry cite, not as a missing block. The three shapes
are read by ``omega_mind.provenance``, the one place that knows them; the
canonical form for new cards is ``support``.

    python3 tools/card_support.py                 # check every package
    python3 tools/card_support.py --strict        # also fail on cards without support
    python3 tools/card_support.py --list-missing  # the cards still to annotate
    python3 tools/card_support.py --json out.json

What a passing check proves: the block is well formed, the recorded hash
matches the quote, and — when ``source_file`` is on this machine — the quote
appears in that file verbatim (whitespace ignored). A cite whose file is
absent is reported as *unverified*, never as valid.

What it does NOT prove: that the quote actually supports the card's wording
(a quote can be real and irrelevant), that the card is complete or faithful
to its source, or that an unverified cite is wrong. It checks provenance
shape, not meaning. Source files are local only and never enter the repo.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _extra in (ROOT, ROOT / "right"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from omega_mind import feynman_kb, provenance  # noqa: E402


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    """Whitespace-insensitive form: line wraps and spacing never matter."""
    return "".join(text.split())


def _resolve_source(path_text: str, bases: List[Path]) -> Optional[Path]:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in bases:
        path = base / candidate
        if path.is_file():
            return path
    return None


def _check_entry(entry: object, bases: List[Path]) -> Dict[str, str]:
    """('valid'|'unverified'|'cite'|'invalid', problem) for one entry."""
    if not isinstance(entry, dict):
        return {"status": "invalid", "problem": "support entry is not an object"}
    source = entry.get("source")
    quote = entry.get("quote")
    if not isinstance(source, str) or not source.strip():
        return {"status": "invalid", "problem": "missing 'source'"}
    if quote is None:
        return {"status": "cite", "problem": ""}
    if not isinstance(quote, str) or not quote.strip():
        return {"status": "invalid", "problem": "missing 'quote'"}
    recorded = entry.get("quote_sha256")
    if recorded is not None and recorded != _sha(quote):
        return {"status": "invalid",
                "problem": "quote_sha256 does not match the quote"}
    source_file = entry.get("source_file")
    if source_file is None:
        return {"status": "unverified",
                "problem": "no source_file; quote cannot be checked"}
    if not isinstance(source_file, str) or not source_file.strip():
        return {"status": "invalid", "problem": "'source_file' must be a path"}
    path = _resolve_source(source_file, bases)
    if path is None:
        return {"status": "unverified",
                "problem": f"source file not on this machine: {source_file}"}
    body = _norm(path.read_text(encoding="utf-8", errors="replace"))
    if _norm(quote) not in body:
        return {"status": "invalid",
                "problem": f"quote not found in {source_file}"}
    return {"status": "valid", "problem": ""}


def check(root: Optional[Path] = None,
          package: Optional[str] = None) -> dict:
    root = Path(root) if root is not None else feynman_kb.PACKAGES_ROOT
    bases = [root] + [ROOT]
    report = {"root": str(root), "packages": {}}
    for manifest in sorted(root.glob("*/persona.json")):
        name = manifest.parent.name
        language = feynman_kb.package_language(manifest.parent)
        if package and name != package:
            continue
        knowledge = manifest.parent / "knowledge"
        if not knowledge.is_dir():
            continue
        package_row = {"cards": 0, "with_support": 0, "valid": 0,
                       "unverified": 0, "cited": 0, "missing": 0,
                       "invalid": [], "missing_ids": [], "reviewed": 0,
                       "translation_markers": 0}
        for file in sorted(knowledge.glob("*.jsonl")):
            if file.name in feynman_kb.SKIP_FILES:
                continue
            for lineno, line in enumerate(
                    file.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                card_id = str(row.get("id") or f"{file.stem}:{lineno}")
                block = f"{name}:{file.stem}:{card_id}"
                package_row["cards"] += 1
                if row.get("translation_of") or row.get("source_language") not in (
                        None, "", language):
                    package_row["translation_markers"] += 1
                support = provenance.entries(row)
                if not support:
                    if provenance.declared(row):
                        package_row["with_support"] += 1
                        package_row["invalid"].append(
                            {"card": block,
                             "problem": "'support' must be a non-empty list"})
                    else:
                        package_row["missing"] += 1
                        package_row["missing_ids"].append(block)
                    continue
                if provenance.declared(row):
                    package_row["with_support"] += 1
                statuses = [_check_entry(entry, bases) for entry in support]
                bad = [s for s in statuses if s["status"] == "invalid"]
                if bad:
                    package_row["invalid"].append(
                        {"card": block, "problem": bad[0]["problem"]})
                elif any(s["status"] == "unverified" for s in statuses):
                    package_row["unverified"] += 1
                elif all(s["status"] == "cite" for s in statuses):
                    package_row["cited"] += 1
                else:
                    package_row["valid"] += 1
                    if all(entry.get("reviewed_by") and entry.get("quote_sha256")
                           for entry in support):
                        package_row["reviewed"] += 1
        report["packages"][name] = package_row
    return report


def render(report: dict, list_missing: bool = False) -> str:
    lines = []
    for name, row in report["packages"].items():
        lines.append(
            f"{name:<16} cards {row['cards']:>4}  supported {row['with_support']:>4}"
            f"  valid {row['valid']:>4}  unverified {row['unverified']:>4}"
            f"  cited {row['cited']:>4}"
            f"  missing {row['missing']:>4}  invalid {len(row['invalid'])}"
            f"  reviewed {row['reviewed']}  translation markers {row['translation_markers']}")
        for problem in row["invalid"][:20]:
            lines.append(f"      INVALID {problem['card']}: {problem['problem']}")
        if list_missing:
            for block in row["missing_ids"]:
                lines.append(f"      missing  {block}")
    return "\n".join(lines)



def pilot(package: Path, questions: Path, limit: int) -> list[dict]:
    """Prioritize human review with a fixed query list, without editing cards."""
    from omega_mind.dialogue import card_revision
    from omega_mind.golden_queries import for_package

    kb = feynman_kb.FeynmanKB(knowledge_dir=str(package / "knowledge"),
                              source_prefix=package.name)
    suite = json.loads(questions.read_text(encoding="utf-8"))
    counts = Counter(cid for case in suite["cases"] if case.get("language") == kb.language
                     for cid in case["acceptable"] if cid in kb.by_id)
    golden, _ = for_package(package.name, package.parent)
    counts.update(cid for _, language, acceptable in golden if language == kb.language
                  for cid in acceptable if cid in kb.by_id)
    physics = {"physics_guardrail", "physics_card", "lecture_card", "physics_edge", "teachable_unit"}
    ordered = sorted((doc for doc in kb.docs if doc.id in counts and doc.kind in physics
                      and not provenance.cited(doc.row)),
                     key=lambda doc: (-counts[doc.id], doc.id))
    return [{"id": doc.id, "title": doc.title, "revision": card_revision(doc),
             "query_count": counts[doc.id], "current_source": doc.row.get("source"),
             "status": "needs_source_and_review"}
            for doc in ordered[:limit]]

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", help="personas root (default: the repo's)")
    parser.add_argument("--package", help="only this package id")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when a card has no support block")
    parser.add_argument("--list-missing", action="store_true",
                        help="print every card still without support")
    parser.add_argument("--json", metavar="PATH", help="write the full report")
    parser.add_argument("--pilot", type=int, metavar="N",
                        help="emit up to N unannotated cards prioritized for review")
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "tests" / "data" / "confirm_set_v1.json")
    args = parser.parse_args(argv)

    if args.pilot is not None:
        if not args.package or args.pilot <= 0:
            parser.error("--pilot requires a positive count and --package")
        root = Path(args.root) if args.root else feynman_kb.PACKAGES_ROOT
        if not (root / args.package / "persona.json").is_file():
            parser.error("package not found")
        body = json.dumps(pilot(root / args.package, args.queries, args.pilot),
                          ensure_ascii=False, indent=2) + "\n"
        if args.json:
            Path(args.json).write_text(body, encoding="utf-8")
        else:
            print(body, end="")
        return 0
    report = check(root=args.root, package=args.package)
    print(render(report, list_missing=args.list_missing))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    invalid = sum(len(row["invalid"]) for row in report["packages"].values())
    missing = sum(row["missing"] for row in report["packages"].values())
    if invalid:
        print(f"FAIL: {invalid} card(s) carry an invalid support block")
        return 1
    if args.strict and missing:
        print(f"FAIL: {missing} card(s) have no support block")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
