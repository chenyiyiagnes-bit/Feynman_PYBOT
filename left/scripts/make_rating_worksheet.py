"""生成出句评分工作表：固定抽样、配置溯源、去重、确定性乱序。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/make_rating_worksheet.py \
        --out data/rating_worksheet_p8b.jsonl --limit 24

评分者在每行 "rating" 处填 1–5（或 -1 表示无法判断），不改其他字段。
工作表版本写进 meta，旧表不可与新报告混用。

配置集刻意选得窄：baseline、线性预设、Boltzmann、拓扑默认、拓扑三档网格
（simplex_boost/hole_factor/geo_gain）。同一 (query, text) 只出现一次，
configs 字段记录所有产生过这句话的配置，评一次分即给所有对应配置记账。
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
from scripts.eval_fusion import PRESET, QUERIES, QUERY_SET_VERSION

VAULT_PATH = Path("data/feynman_style_vault_p9.json")
TOPOLOGY_PATH = Path("data/topology_graph_p9.npz")
WORKSHEET_VERSION = "p12-rating-v1"
SHUFFLE_SEED = 20261125

# 配置 id -> (请求参数, 拓扑网格参数或 None)
CONFIG_SET = {
    "baseline": ({}, None),
    "linear_preset": ({"operators": PRESET}, None),
    "boltzmann_t1": ({"operators": PRESET, "fusion": "boltzmann", "fusion_tau": 1.0}, None),
    "topology_default": ({"use_topology": True}, (1.45, 0.55, 0.60)),
    "conditioned_topology": ({"conditioned_chain": True, "use_topology": True}, (1.45, 0.55, 0.60)),
    # P10：解码期检索约束（实体 E→C 边选后果）
    "retrieval_topology": ({"retrieve_consequence": True, "use_topology": True}, (1.45, 0.55, 0.60)),
    # P10b：检索池再过滤主语错配/方向矛盾/残句
    "retrieval_gates": ({"retrieve_consequence": True, "retrieval_gates": True,
                         "use_topology": True}, (1.45, 0.55, 0.60)),
    # P12：问句实体锁定
    "topic_lock": ({"topic_lock": True, "use_topology": True}, (1.45, 0.55, 0.60)),
    # P12+检索+门禁：实体锁到问句，后果按实体 E→C 边取
    "lock_retrieval": ({"topic_lock": True, "retrieve_consequence": True,
                        "retrieval_gates": True,
                        "use_topology": True}, (1.45, 0.55, 0.60)),
    # P4/P5：信息量偏好（空话重复降权、引入新指称加权）
    "lock_retrieval_informative": ({"topic_lock": True, "retrieve_consequence": True,
                                    "retrieval_gates": True, "informative_pref": True,
                                    "use_topology": True}, (1.45, 0.55, 0.60)),
    # P14：专属从句（tier=2）优先于通用原子
    "lock_retrieval_specific": ({"topic_lock": True, "retrieve_consequence": True,
                                  "retrieval_gates": True, "specific_pref": True,
                                  "use_topology": True}, (1.45, 0.55, 0.60)),
}

# 配置子集：全量或只比较 P4/P5 前后
CONFIG_SUBSETS = {
    "full": list(CONFIG_SET),
    "p13": ["baseline", "retrieval_gates", "lock_retrieval", "lock_retrieval_informative"],
    "p14": ["lock_retrieval", "lock_retrieval_specific"],
}


def _patch(grid) -> None:
    if grid is None:
        return
    tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN = grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rating_worksheet_p8b.jsonl")
    ap.add_argument("--meta", default="data/rating_worksheet_p8b.meta.json")
    ap.add_argument("--limit", type=int, default=24, help="参与抽样的查询数")
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    ap.add_argument("--vault", default=str(VAULT_PATH))
    ap.add_argument("--blind", action="store_true",
                    help="盲评模式：行内不含 configs，配置溯源写入 --map 文件，评分时不可见")
    ap.add_argument("--map", default=None, help="盲评模式的 configs 映射表路径")
    ap.add_argument("--subset", default="full", choices=sorted(CONFIG_SUBSETS),
                    help="配置子集：full = 全部配置；p13 = 只比 P4/P5 前后")
    args = ap.parse_args()
    if args.blind and not args.map:
        ap.error("--blind 必须同时给 --map")

    vault = Vault(Path(args.vault))
    queries = QUERIES[: args.limit]
    topo_path = Path(args.topology)

    # (query, text) -> configs；保持首次出现顺序，最后统一乱序
    pool: dict[tuple[str, str], list[str]] = {}
    defaults = (tb.SIMPLEX_BOOST, tb.HOLE_FACTOR, tb.GEO_GAIN)
    try:
        for cfg in CONFIG_SUBSETS[args.subset]:
            overrides, grid = CONFIG_SET[cfg]
            _patch(grid)
            engine = TopologyManifoldEngine.load(topo_path)
            for q in queries:
                req = PrintRequest(user_query=q, nonce=0, top_k=10, **overrides)
                topo = engine if (req.use_topology or grid) else None
                res = run_feynman_print(vault, req, ext_hash="", topology=topo)
                text = res.get("printed_collapse") or res.get("top_resonant_line")
                if not text or res.get("greeting"):
                    continue
                key = (q, text)
                pool.setdefault(key, []).append(cfg)
    finally:
        _patch(defaults)

    rows = [(q, t, cfgs) for (q, t), cfgs in pool.items()]
    random.Random(SHUFFLE_SEED).shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config_map = {}
    with out.open("w", encoding="utf-8") as fh:
        for i, (q, text, cfgs) in enumerate(rows, 1):
            rid = "r" + hashlib.sha256(f"{q}|{text}".encode("utf-8")).hexdigest()[:10]
            row = {"id": rid, "query": q, "text": text}
            if args.blind:
                config_map[rid] = cfgs
            else:
                row["configs"] = cfgs
            row["rating"] = None
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.blind:
        Path(args.map).write_text(json.dumps(config_map, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    meta = {
        "worksheet_version": f"{WORKSHEET_VERSION}/{args.subset}",
        "query_set_version": QUERY_SET_VERSION,
        "queries": len(queries),
        "rows": len(rows),
        "shuffle_seed": SHUFFLE_SEED,
        "config_grids": {k: v[1] for k, v in CONFIG_SET.items()},
        "scale": {"1": "很差", "2": "差", "3": "一般", "4": "好", "5": "很好", "-1": "无法判断"},
    }
    Path(args.meta).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"工作表已写入 {out}（{len(rows)} 行，去重后）")


if __name__ == "__main__":
    main()
