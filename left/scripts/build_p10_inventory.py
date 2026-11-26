"""P10 后果库存：P9 库存 + 语料挖出的实体专属从句（已过质量门禁）。

不改 P9 产物；新库存单独落盘，随后由 build_vault_p10 重建 vault 与拓扑。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_p10_inventory.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import plugins.feynman_vault as fv
from plugins.entity_clause_miner import filter_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--p9", default="data/p9_consequence_inventory.json")
    ap.add_argument("--candidates", default="data/entity_consequence_candidates.json")
    ap.add_argument("--out", default="data/p10_consequence_inventory.json")
    ap.add_argument("--report", default="data/p10_consequence_inventory_report.json")
    args = ap.parse_args()

    p9 = json.loads(Path(args.p9).read_text(encoding="utf-8"))
    tier = {it["word"]: it["tier"] for it in p9}
    cand = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    rows = filter_candidates(cand["rows"])

    added, rejected = [], []
    for r in rows:
        w = r["clause"]
        if w in tier:
            rejected.append({"word": w, "why": "duplicate"})
            continue
        if not fv._p8_consequence_ok(w):
            rejected.append({"word": w, "why": "p8_gate"})
            continue
        tier[w] = 2
        added.append({"word": w, "entity": r["entity"],
                      "count": r["count"], "sections": r["sections"]})

    words = list(tier)
    pattern = re.compile("|".join(
        sorted((re.escape(w) for w in words), key=len, reverse=True)))
    sub_map = {w: [x for x in words if x != w and x in w] for w in words}

    counts: Counter = Counter()
    sources: dict[str, list[str]] = {}
    examples: dict[str, str] = {}
    for src, section, text in fv.iter_sections(Path(args.corpus)):
        tag = f"{src}｜{section[:24]}"
        for sent in fv.sentences(text):
            present: set[str] = set()
            for m in pattern.finditer(sent):
                w = m.group(0)
                present.add(w)
                present.update(sub_map.get(w, ()))
            for w in present:
                counts[w] += 1
                examples.setdefault(w, sent[:80])
                lst = sources.setdefault(w, [])
                if tag not in lst and len(lst) < 3:
                    lst.append(tag)

    missing = [w for w in words if counts[w] == 0]
    bad_specific = [w for w in words
                    if tier[w] == 2 and not fv._p8_consequence_ok(w)]

    max_count = max(counts.values()) if counts else 1
    items = []
    for w in words:
        if counts[w] == 0:
            continue
        sample = examples.get(w) or w
        style, phase = fv._style_of(sample)
        amp = round(0.94 + 0.06 * math.log1p(counts[w]) / math.log1p(max_count), 3)
        items.append({
            "word": w,
            "slot": "consequence",
            "count": counts[w],
            "amp": amp,
            "domain": "physics",
            "style": style,
            "phase_deg": phase,
            "sources": sources.get(w, []),
            "example": fv._clean_consequence(fv._trim_tail(sample))[:60],
            "tier": tier[w],
        })
    items.sort(key=lambda it: (-it["count"], it["word"]))

    report = {
        "p9_words": len(p9),
        "candidates_reviewed": len(rows),
        "added": len(added),
        "rejected": len(rejected),
        "rejected_examples": rejected[:20],
        "inventory_words": len(items),
        "generic": sum(1 for it in items if it["tier"] == 1),
        "specific": sum(1 for it in items if it["tier"] == 2),
        "missing_in_corpus": missing,
        "specific_gate_failed": bad_specific,
        "added_detail": added,
    }
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("p9_words", "candidates_reviewed", "added", "rejected",
                       "inventory_words", "generic", "specific")},
                     ensure_ascii=False, indent=1))
    if missing or bad_specific:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
