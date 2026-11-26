"""P14 机械效应评估：specific_pref 开关对前 48 个查询的实际影响。

不用人工评分，只看客观量：出句是否变化、首选句的后果槽是否为 tier=2 专属从句、
变化里有多少是真换成了专属从句而不是拓扑重排噪声。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/eval_p14_specific.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plugins.topology_builder as tb
from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.topology_builder import TopologyManifoldEngine
from scripts.eval_fusion import PRESET, QUERIES  # noqa: F401  (PRESET 供对照保留)

VAULT = Path("data/feynman_style_vault_p10.json")
TOPO = Path("data/topology_graph_p10.npz")
OUT = Path("data/eval_p14_specific.json")

BASE = {"topic_lock": True, "retrieve_consequence": True, "retrieval_gates": True,
        "use_topology": True}
GRID = (1.45, 0.55, 0.60)


def _run(vault, engine, q, specific: bool):
    tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN = GRID
    req = PrintRequest(user_query=q, nonce=0, top_k=10,
                       specific_pref=specific, **BASE)
    return run_feynman_print(vault, req, ext_hash="", topology=engine)


def main() -> None:
    vault = Vault(VAULT)
    engine = TopologyManifoldEngine.load(TOPO)
    tier2 = {g["word"]: g.get("tier") for g in vault.by_slot["consequence"]}

    rows = []
    for q in QUERIES[:48]:
        a = _run(vault, engine, q, False)
        b = _run(vault, engine, q, True)
        ta = a.get("printed_collapse") or ""
        tb_ = b.get("printed_collapse") or ""
        cons_a = [p["glyphs"]["consequence"] for p in a.get("paths", [])]
        cons_b = [p["glyphs"]["consequence"] for p in b.get("paths", [])]
        tier_a = [tier2.get(c) for c in cons_a]
        tier_b = [tier2.get(c) for c in cons_b]
        rows.append({
            "query": q,
            "changed": ta != tb_,
            "top_before": ta, "top_after": tb_,
            "top_is_specific_before": bool(tier_a and tier_a[0] == 2),
            "top_is_specific_after": bool(tier_b and tier_b[0] == 2),
            "n_specific_before": sum(1 for t in tier_a if t == 2),
            "n_specific_after": sum(1 for t in tier_b if t == 2),
            "new_specific": sorted(set(c for c, t in zip(cons_b, tier_b) if t == 2)
                                   - set(c for c, t in zip(cons_a, tier_a) if t == 2)),
        })

    changed = [r for r in rows if r["changed"]]
    summary = {
        "queries": len(rows),
        "changed": len(changed),
        "top_specific_before": sum(r["top_is_specific_before"] for r in rows),
        "top_specific_after": sum(r["top_is_specific_after"] for r in rows),
        "queries_gaining_specific_in_paths": sum(
            1 for r in rows if r["n_specific_after"] > r["n_specific_before"]),
        "queries_losing_specific_in_paths": sum(
            1 for r in rows if r["n_specific_after"] < r["n_specific_before"]),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in changed:
        print(f"\n[{r['query']}]\n  前：{r['top_before']}\n  后：{r['top_after']}")


if __name__ == "__main__":
    main()
