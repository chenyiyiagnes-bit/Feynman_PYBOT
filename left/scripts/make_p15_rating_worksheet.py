"""P15 决策盲评表：只比两档——p10 检索 vs p15+专属从句优先。

旧表（rating_worksheet_p14_blind.jsonl）用的是 p9 字库和 12 个配置，
大量查询在两档下产出同一句、去重后无法区分，且不含 P15 的新句，
评完也回答不了"要不要把 P15 + specific_pref 设为默认"。

本表只保留能回答该问题的两个配置，查询覆盖 13 个补库实体（每个 3 种问法）
加 16 条日常对照查询；行内不含配置，映射写单独文件，评分时不可见。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/make_p15_rating_worksheet.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import plugins.topology_builder as tb
from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.topology_builder import TopologyManifoldEngine
from scripts.eval_fusion import QUERIES, QUERY_SET_VERSION

GRID = (1.45, 0.55, 0.60)
SHUFFLE_SEED = 20261125
WORKSHEET_VERSION = "p15-decision-v1"

GAP_ENTITIES = [
    "坐标系", "量子力学", "直线", "常数", "平面", "强度", "颜色",
    "函数", "速率", "长度", "电路", "线圈", "曲线",
]
TEMPLATES = ["{e}是什么", "{e}为什么会变化", "{e}有什么特点"]

BASE_REQ = dict(topic_lock=True, retrieve_consequence=True,
                retrieval_gates=True, use_topology=True)

CONFIGS = {
    "p10_lock_retrieval": {
        "vault": "data/feynman_style_vault_p10.json",
        "topology": "data/topology_graph_p10.npz",
        "overrides": {},
    },
    "p15_lock_retrieval_specific": {
        "vault": "data/feynman_style_vault_p15.json",
        "topology": "data/topology_graph_p15.npz",
        "overrides": {"specific_pref": True},
    },
}


def _queries() -> list[str]:
    qs = [t.format(e=e) for e in GAP_ENTITIES for t in TEMPLATES]
    qs += QUERIES[:16]
    return qs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rating_worksheet_p15_blind.jsonl")
    ap.add_argument("--map", default="data/rating_worksheet_p15_blind.map.json")
    ap.add_argument("--meta", default="data/rating_worksheet_p15_blind.meta.json")
    args = ap.parse_args()

    defaults = (tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN)
    tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN = GRID
    pool: dict[tuple[str, str], list[str]] = {}
    try:
        for cfg, spec in CONFIGS.items():
            vault = Vault(Path(spec["vault"]))
            engine = TopologyManifoldEngine.load(Path(spec["topology"]))
            for q in _queries():
                req = PrintRequest(user_query=q, nonce=0, top_k=10,
                                   **BASE_REQ, **spec["overrides"])
                res = run_feynman_print(vault, req, ext_hash="", topology=engine)
                text = res.get("printed_collapse") or res.get("top_resonant_line")
                if not text or res.get("greeting"):
                    continue
                pool.setdefault((q, text), []).append(cfg)
    finally:
        tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN = defaults

    rows = list(pool)
    random.Random(SHUFFLE_SEED).shuffle(rows)

    cmap = {}
    with Path(args.out).open("w", encoding="utf-8") as fh:
        for i, (q, text) in enumerate(rows, 1):
            rid = "r" + hashlib.sha256(f"{q}|{text}".encode()).hexdigest()[:10]
            cmap[rid] = pool[(q, text)]
            fh.write(json.dumps({"id": rid, "query": q, "text": text,
                                 "rating": None}, ensure_ascii=False) + "\n")
    Path(args.map).write_text(json.dumps(cmap, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    both = sum(1 for c in cmap.values() if len(c) == 2)
    only10 = sum(1 for c in cmap.values() if c == ["p10_lock_retrieval"])
    only15 = sum(1 for c in cmap.values() if c == ["p15_lock_retrieval_specific"])
    meta = {
        "worksheet_version": WORKSHEET_VERSION,
        "query_set_version": QUERY_SET_VERSION,
        "queries": len(_queries()),
        "rows": len(rows),
        "rows_both_configs": both,
        "rows_only_p10": only10,
        "rows_only_p15": only15,
        "shuffle_seed": SHUFFLE_SEED,
        "scale": {"1": "很差", "2": "差", "3": "一般", "4": "好", "5": "很好"},
    }
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                               encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
