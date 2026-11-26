"""生成产品默认档的黄金夹具：P15 字库 + 四检索开关 + 拓扑层。

固定问法、风格 60°、nonce 0；只记录首句与前三句，用于回归时发现
"没人打算改、但结果变了"的排版漂移。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/make_p15_default_golden.py [--force]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["QBOT_FEYNMAN_VAULT"] = "data/feynman_style_vault_p15.json"
os.environ["QBOT_FEYNMAN_TOPOLOGY"] = "data/topology_graph_p15.npz"

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.topology_builder import get_engine

WORKSHEET = Path("data/rating_worksheet_p15_blind.jsonl")
OUT = Path("tests/fixtures_feynman_p15_default.json")

QUERIES = [
    "动量为什么守恒", "光速为什么不变", "测量为什么会改变结果", "曲线为什么会变化",
    "常数是什么", "电路有什么特点", "波粒二象性怎么理解", "温度升高以后会怎样",
    "函数为什么会变化", "坐标系是什么", "直线为什么会变化", "强度有什么特点",
    "熵为什么会增加", "能量守恒是什么意思", "长度是什么", "量子是什么东西",
    "磁场能看见吗", "颜色为什么会变化", "线圈是什么", "量子力学有什么特点",
    "相对论为什么会改变时间", "速率是什么", "平面有什么特点", "电子为什么会发光",
    "引力为什么这么弱",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖已有夹具")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} 已存在；确认要重建请加 --force")

    vault = Vault(Path(os.environ["QBOT_FEYNMAN_VAULT"]))
    engine = get_engine(force=True)
    assert engine is not None, "P15 拓扑图加载失败"

    fixture = {}
    for q in QUERIES:
        req = PrintRequest(style_deg=60.0, nonce=0, user_query=q,
                           topic_lock=True, retrieve_consequence=True,
                           retrieval_gates=True, specific_pref=True)
        out = run_feynman_print(vault, req, topology=engine)
        paths = out.get("paths") or []
        fixture[q] = {
            "line": out.get("printed_collapse"),
            "weight": round(float(paths[0]["weight"]), 6) if paths else None,
            "top3": [p.get("line") for p in paths[:3]],
        }
    payload = {
        "vault": os.environ["QBOT_FEYNMAN_VAULT"],
        "topology": os.environ["QBOT_FEYNMAN_TOPOLOGY"],
        "topology_hash": engine.topo_hash,
        "style_deg": 60.0,
        "nonce": 0,
        "config": ["topic_lock", "retrieve_consequence", "retrieval_gates",
                   "specific_pref", "topology"],
        "cases": fixture,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {len(fixture)} 条黄金用例 → {OUT}")


if __name__ == "__main__":
    main()
