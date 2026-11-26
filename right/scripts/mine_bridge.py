"""Mine candidate bridge pairs from the corpus — offline, for hand review.

The ``bridge`` table in the Chinese rule file (``lang/zh/*.json``) maps
everyday words to the vocabulary the cards actually use. It is small because every entry is a human judgement.
This script proposes more of the same kind of entry by scoring token pairs
with PPMI over document co-occurrence:

    PPMI(x, y) = max(0, log( P(x, y) / (P(x) * P(y)) ))

It only writes a JSON file. Nothing here changes retrieval, and the output is
a candidate list, not a lexicon: an everyday word that co-occurs with a
corpus term is a lead, not evidence that the substitution is right.

Usage:
    python3 scripts/mine_bridge.py [out.json]
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_mind.feynman_kb import FeynmanKB, _CJK
from omega_mind.lang import load as load_lang

OUT_DEFAULT = ROOT / "scripts" / "data" / "bridge_candidates.json"

# A pair needs real support; one document in common is a coincidence.
MIN_COOC = 3
TOP_PER_SEED = 8

_ASCII = re.compile(r"[a-z0-9]{2,}")


def tokens(text: str) -> set:
    out = set(_ASCII.findall(text.lower()))
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.add(run)
        else:
            out.update(run[i:i + 2] for i in range(len(run) - 1))
    return {tok for tok in out if tok not in stop_bigrams}


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DEFAULT
    zh = load_lang("zh")
    stop_bigrams = zh._stop_bigrams
    bridge = zh._bridge
    kb = FeynmanKB(language="zh")
    if not kb.docs:
        raise SystemExit("corpus not found; set OMEGA_KB_PATH or check DEFAULT_DIRS")

    doc_tokens: List[set] = []
    for doc in kb.docs:
        row = doc.row or {}
        blob = " ".join(str(row.get(k, "")) for k in
                        ("title", "text", "tags", "aliases"))
        doc_tokens.append(tokens(blob) | set(doc.weights))
    total = len(doc_tokens)
    df: Counter = Counter()
    for toks in doc_tokens:
        df.update(toks)

    def cooc(seed: str, term: str) -> int:
        return sum(1 for toks in doc_tokens if seed in toks and term in toks)

    report: Dict[str, List[Dict[str, float]]] = {}
    for seed, _mapped in bridge:
        if df[seed] < MIN_COOC:
            continue
        scored: List[Tuple[float, str, int]] = []
        candidates = [t for t, n in df.items()
                      if n >= MIN_COOC and t != seed and len(t) >= 2]
        for term in candidates:
            joint = cooc(seed, term)
            if joint < MIN_COOC:
                continue
            p_xy = joint / total
            p_x = df[seed] / total
            p_y = df[term] / total
            score = math.log(p_xy / (p_x * p_y))
            if score > 0:
                scored.append((score, term, joint))
        scored.sort(reverse=True)
        report[seed] = [
            {"term": term, "ppmi": round(score, 3), "cooc": joint}
            for score, term, joint in scored[:TOP_PER_SEED]
        ]

    payload = {
        "generated": date.today().isoformat(),
        "corpus_docs": total,
        "note": "候选，不是词典。每条须人工核对后才可写入 lang/zh 的 bridge 表。",
        "seeds": report,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"mined {sum(len(v) for v in report.values())} candidates "
          f"for {len(report)} seeds over {total} docs -> {out_path}")


if __name__ == "__main__":
    main()
