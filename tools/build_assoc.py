"""Rebuild association edges from the cards themselves.

A fresh clone has cards and no edges: ``data/assoc/`` is never committed. This
tool mines candidate edges — two cards that share several of the package's
own terms — and writes them as ``origin="cooccur"``, the quarantine origin the
right brain never speaks aloud. Human-confirmed edges are the only ones with a
mechanism, and this tool never invents one.

The mine is deterministic: same cards, same edges, byte for byte.

Usage:
    python3 tools/build_assoc.py personas/feynman-zh
    python3 tools/build_assoc.py personas/feynman-zh --out /tmp/assoc
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "right", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from omega_mind import assoc, lang  # noqa: E402

import build_lexicon  # noqa: E402

TOOL_VERSION = 1
MIN_SHARED = 3        # two cards sharing two words is a coincidence
MAX_PER_CARD = 8      # a hub that links to everything carries no signal
EVIDENCE = 5          # shared terms kept in the row, for review


def card_terms(core, cards: Sequence[Dict]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for card in cards:
        tokens = {tok for tok in core.segment(build_lexicon.card_text(card))
                  if build_lexicon.keep(core, tok)}
        out[str(card["id"])] = tokens
    return out


def mine(core, cards: Sequence[Dict]) -> List[Dict]:
    terms = card_terms(core, cards)
    by_term: Dict[str, List[str]] = defaultdict(list)
    for card_id in sorted(terms):
        for term in terms[card_id]:
            by_term[term].append(card_id)

    shared: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for term, holders in by_term.items():
        if len(holders) < 2:
            continue
        # A term in most cards says "same genre", not "same subject". Only
        # judge that once there are enough cards for "most" to mean anything,
        # the same guard the lexicon uses.
        if len(cards) >= build_lexicon.MIN_CARDS_FOR_BREADTH \
                and len(holders) / len(cards) > 0.6:
            continue
        for index, left in enumerate(holders):
            for right in holders[index + 1:]:
                shared[(left, right)][term] += 1

    rows = []
    for (left, right), counts in shared.items():
        if sum(counts.values()) < MIN_SHARED:
            continue
        a, b = (left, right) if left < right else (right, left)
        evidence = sorted(counts, key=lambda term: (-counts[term], term))[:EVIDENCE]
        rows.append({"a": a, "b": b, "kind": "same_domain", "origin": "cooccur",
                     "shared": sum(counts.values()), "terms": evidence})

    # Cap each card's fan-out; ties keep the tighter pairing.
    rows.sort(key=lambda row: (-row["shared"], row["a"], row["b"]))
    kept: List[Dict] = []
    degree: Counter = Counter()
    for row in rows:
        if degree[row["a"]] >= MAX_PER_CARD or degree[row["b"]] >= MAX_PER_CARD:
            continue
        degree[row["a"]] += 1
        degree[row["b"]] += 1
        kept.append(row)
    kept.sort(key=lambda row: (row["a"], row["b"]))
    return kept


def build(persona_dir: Path, out_dir: Path | None = None,
          built: str | None = None) -> Dict:
    persona_dir = Path(persona_dir).resolve()
    meta_path = persona_dir / "persona.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    core = lang.load(meta.get("language") or "zh",
                     meta.get("language_profile") or "default")
    cards = build_lexicon.load_cards(persona_dir / "knowledge")
    if not cards:
        raise SystemExit(f"{persona_dir / 'knowledge'} has no cards")

    rows = mine(core, cards)
    for row in rows:
        # Never write an edge the graph itself would drop.
        edge = assoc.Edge(a=row["a"], b=row["b"], kind=row["kind"],
                          origin=row["origin"])
        trouble = assoc.problems(edge)
        if trouble:
            raise SystemExit(f"mined edge {row['a']}->{row['b']}: {trouble}")

    out_dir = Path(out_dir) if out_dir else persona_dir / "data" / "assoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "assoc_edges.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                              for row in rows), encoding="utf-8")
    manifest = {
        "tool": "tools/build_assoc.py",
        "tool_version": TOOL_VERSION,
        "built": built or datetime.date.today().isoformat(),
        "persona": persona_dir.name,
        "cards": len(cards),
        "edges": len(rows),
        "min_shared": MIN_SHARED,
        "max_per_card": MAX_PER_CARD,
        "source_digest": build_lexicon.digest(persona_dir / "knowledge"),
        "note": ("cooccur edges only: candidates for review, never spoken; "
                 "derived from knowledge/*.jsonl, never committed"),
    }
    (out_dir / "assoc_edges.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("persona", help="package directory, e.g. personas/feynman-zh")
    parser.add_argument("--out", help="output directory (default <persona>/data/assoc)")
    parser.add_argument("--date", help="manifest build date (default today)")
    args = parser.parse_args(argv)
    manifest = build(Path(args.persona), out_dir=Path(args.out) if args.out else None,
                     built=args.date)
    print(f"{manifest['persona']}: {manifest['cards']} cards -> "
          f"{manifest['edges']} candidate edges (cooccur)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
