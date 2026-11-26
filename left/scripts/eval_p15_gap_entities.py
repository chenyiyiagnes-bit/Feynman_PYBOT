"""P15 空档实体检索评估：p10 vs p15 在 12 个补库实体上的前 3 条后果。

不看模型输出，只看检索约束本身：
- 这些实体在 p10 里有多少 E→C 边（0 条即退回相位池、检索失效）；
- p15 补边后边数、前 3 条是否换成了新补的专属从句；
- 新从句必须逐条来自 p15 库存的新增集合（可溯源）。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/eval_p15_gap_entities.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from plugins.feynman_typesetter import Vault, _retrieve_consequences

P10 = Path("data/feynman_style_vault_p10.json")
P15 = Path("data/feynman_style_vault_p15.json")
P15_INV = Path("data/p15_consequence_inventory.json")
P10_INV = Path("data/p10_consequence_inventory.json")
OUT = Path("data/eval_p15_gap_entities.json")

ENTITIES = [
    "坐标系", "量子力学", "直线", "常数", "平面", "强度", "颜色",
    "函数", "速率", "长度", "电路", "线圈", "曲线",
]


def _top3(vault: Vault, entity: str, deg: float) -> list[str]:
    pool = vault.by_slot["consequence"]
    got = _retrieve_consequences(vault, entity, pool, deg, top_k=3,
                                 specific_pref=True)
    return [g["word"] for g in got]


def _wired(vault: Vault, entity: str) -> dict[str, int]:
    out = {}
    for g in vault.by_slot["consequence"]:
        n = vault.cooc_count(entity, g["word"])
        if n > 0:
            out[g["word"]] = n
    return out


def main() -> None:
    p10 = Vault(P10)
    p15 = Vault(P15)
    p10_words = {it["word"] for it in
                 json.loads(P10_INV.read_text(encoding="utf-8"))}
    new_words = {it["word"] for it in
                 json.loads(P15_INV.read_text(encoding="utf-8"))} - p10_words
    tier = {g["word"]: g.get("tier", 1) for g in p15.by_slot["consequence"]}
    phase = {g["word"]: g.get("phase_deg", 45.0) for g in p15.by_slot["entity"]}

    rows = []
    for e in ENTITIES:
        deg = float(phase.get(e, 45.0))
        wa, wb = _wired(p10, e), _wired(p15, e)
        ta, tb_ = _top3(p10, e, deg), _top3(p15, e, deg)
        rows.append({
            "entity": e,
            "wired_p10": len(wa),
            "wired_p15": len(wb),
            "gap_edges_added": sorted(set(wb) - set(wa)),
            "top3_p10": ta,
            "top3_p15": tb_,
            "top3_new": [w for w in tb_ if w in new_words],
            "top3_specific": sum(1 for w in tb_ if tier.get(w) == 2),
            "changed": ta != tb_,
            "fallback_p10": not wa,
        })

    summary = {
        "entities": len(rows),
        "p10_no_edge": sum(r["fallback_p10"] for r in rows),
        "p15_no_edge": sum(r["wired_p15"] == 0 for r in rows),
        "entities_with_new_top3_clause": sum(1 for r in rows if r["top3_new"]),
        "total_new_top3_slots": sum(len(r["top3_new"]) for r in rows),
        "entities_changed": sum(r["changed"] for r in rows),
        "wired_edges_total_p10": sum(r["wired_p10"] for r in rows),
        "wired_edges_total_p15": sum(r["wired_p15"] for r in rows),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"\n[{r['entity']}] 边 {r['wired_p10']}→{r['wired_p15']}"
              f"{'（p10 无约束，退回相位池）' if r['fallback_p10'] else ''}")
        print(f"  前：{r['top3_p10']}")
        print(f"  后：{r['top3_p15']}")


if __name__ == "__main__":
    main()
