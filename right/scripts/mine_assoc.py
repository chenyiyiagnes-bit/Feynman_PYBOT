"""Mine candidate association edges from card overlap — offline only.

The runtime graph (``omega_mind/assoc.py``) says *why* two cards belong
together, and every entry there is a human judgement. This script proposes
pairs by shared significant tokens (title/tag/alias vocabulary), using the
same PPMI spirit as ``mine_bridge.py``:

    score = sum over shared tokens of log(N / df)

Output rows are quarantined candidates: ``origin="cooccur"``,
``kind="same_domain"``, no mechanism. They may be inspected and promoted by
hand, never spoken from the graph as-is.

Usage:
    python3 scripts/mine_assoc.py [out.json]
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_mind.feynman_kb import FeynmanKB

OUT_DEFAULT = ROOT / "scripts" / "data" / "assoc_candidates.json"

# A candidate needs two shared subject tokens; one is usually a scene word.
MIN_SHARED = 2
# Tokens in more than a quarter of the cards carry no subject signal.
COMMON_SHARE = 0.25
# Same floor the scorer uses for "this token names the card's subject".
SIGNIFICANT = 2.5
TOP_PAIRS = 200


# Same concept filed twice (C_RELAX vs U0_RELAX): sharing a stem is a
# duplicate to be merged, not an association to be added.
_SUFFIX = re.compile(r"[-_]")


def same_variant(a: str, b: str) -> bool:
    return _SUFFIX.split(a)[-1].lower() == _SUFFIX.split(b)[-1].lower()


def candidates(docs: Iterable) -> List[Dict[str, object]]:
    """Candidate edges from card-level significant-token overlap."""
    items = [(doc.id, {tok for tok, weight in (doc.weights or {}).items()
                       if weight >= SIGNIFICANT})
             for doc in docs]
    n = max(1, len(items))
    df: Dict[str, int] = {}
    for _doc_id, toks in items:
        for tok in toks:
            df[tok] = df.get(tok, 0) + 1
    ceiling = max(2.0, COMMON_SHARE * n)
    rare = {tok for tok, count in df.items() if 2 <= count <= ceiling}
    significant = [(doc_id, toks & rare) for doc_id, toks in items]
    found: List[Dict[str, object]] = []
    for i, (a, ta) in enumerate(significant):
        for b, tb in significant[i + 1:]:
            if same_variant(a, b):
                continue
            shared = ta & tb
            if len(shared) < MIN_SHARED:
                continue
            score = sum(math.log(n / df[tok]) for tok in shared)
            found.append({
                "a": a,
                "b": b,
                "kind": "same_domain",
                "origin": "cooccur",
                "mechanism": "",
                "shared": sorted(shared),
                "score": round(score, 3),
            })
    found.sort(key=lambda row: row["score"], reverse=True)
    return found[:TOP_PAIRS]


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    kb = FeynmanKB()
    if not kb.docs:
        raise SystemExit("corpus not found; set OMEGA_KB_PATH or check DEFAULT_DIRS")
    found = candidates(kb.docs)
    payload = {
        "generated": date.today().isoformat(),
        "corpus_docs": len(kb.docs),
        "note": ("候选，不是结论。origin=cooccur 的边不得直接说出口；"
                 "人工补上 mechanism 与 difference 并把 origin 改成 human "
                 "后，才写入 omega_mind/data/assoc/assoc_edges.jsonl。"),
        "pairs": found,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"mined {len(found)} candidate pairs over {len(kb.docs)} docs "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
