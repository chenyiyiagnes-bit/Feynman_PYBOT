"""实体专用后果缺口普查。

对库里每个实体，看它的检索池（E→C 边指向的后果）里有没有"专用从句"
（tier=2 的具体后果），还是只有通用原子（温度升高/振幅减小这类）。
输出缺口清单，供语料挖掘闭环定向补库。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/census_entity_consequences.py \
        --vault data/feynman_style_vault_p9.json --out data/entity_consequence_gaps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plugins.feynman_typesetter import Vault, _retrieval_candidate_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="data/feynman_style_vault_p9.json")
    ap.add_argument("--out", default="data/entity_consequence_gaps.json")
    args = ap.parse_args()

    vault = Vault(Path(args.vault))
    entities = [g["word"] for g in vault.by_slot["entity"]]
    cons = vault.by_slot["consequence"]
    tier = {g["word"]: g.get("tier", 1) for g in cons}

    report = []
    for e in entities:
        wired = [g for g in cons if vault.cooc_count(e, g["word"]) > 0]
        gated = [g for g in wired
                 if _retrieval_candidate_ok(g["word"], e, e)[0]]
        specific = [g["word"] for g in gated if tier.get(g["word"], 1) == 2]
        generic = [g["word"] for g in gated if tier.get(g["word"], 1) == 1]
        if specific:
            status = "has_specific"
        elif wired:
            status = "generic_only"
        else:
            status = "no_edges"
        report.append({
            "entity": e,
            "status": status,
            "wired": len(wired),
            "specific": specific,
            "generic_count": len(generic),
            "generic_sample": generic[:6],
        })

    order = {"no_edges": 0, "generic_only": 1, "has_specific": 2}
    report.sort(key=lambda r: (order[r["status"]], r["entity"]))
    summary = {k: sum(1 for r in report if r["status"] == k) for k in order}
    out = {"vault": args.vault, "entities": len(report), "summary": summary,
           "items": report}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for r in report:
        if r["status"] != "has_specific":
            print(f"  [{r['status']:12s}] {r['entity']}  wired={r['wired']:2d} "
                  f"generic={r['generic_count']:2d} {r['generic_sample']}")


if __name__ == "__main__":
    main()
