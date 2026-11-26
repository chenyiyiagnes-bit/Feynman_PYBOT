"""P15 拓扑层消融：同一字库、同一问题，拓扑开/关只差一个乘数。

拓扑层（测地线能量 + 实心三角形判据）一直是产品默认开启，但没有证据说明
它在 P15 字库上改变了什么。本脚本只测可数的事实：
- 首句是否变化、空白率（坍缩失败或空句）；
- 前 3 条路径重合度（拓扑是否只是打乱顺序）；
- 单次调用延迟增量。

不替代人工评分：若首句变化率极低，说明它没在做事，可以默认关；
变化率高再补 20 题盲评判断方向。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ablate_topology_p15.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ["QBOT_FEYNMAN_VAULT"] = "data/feynman_style_vault_p15.json"
os.environ["QBOT_FEYNMAN_TOPOLOGY"] = "data/topology_graph_p15.npz"

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.topology_builder import get_engine

WORKSHEET = Path("data/rating_worksheet_p15_blind.jsonl")
OUT = Path("data/eval_topology_ablation_p15.json")
BLANK = "【此处没有可测的东西】"


def _questions() -> list[str]:
    seen = []
    for line in WORKSHEET.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)["query"]
        if q not in seen:
            seen.append(q)
    return seen


def _run(vault, q, topology):
    req = PrintRequest(style_deg=60.0, nonce=0, user_query=q,
                       topic_lock=True, retrieve_consequence=True,
                       retrieval_gates=True, specific_pref=True)
    t0 = time.perf_counter()
    out = run_feynman_print(vault, req, topology=topology)
    ms = (time.perf_counter() - t0) * 1000.0
    line = out.get("printed_collapse") or ""
    top3 = [p.get("line") or "" for p in (out.get("paths") or [])[:3]]
    return {"line": line, "blank": (not line) or line == BLANK,
            "top3": top3, "ms": ms}


def main() -> None:
    vault = Vault(Path(os.environ["QBOT_FEYNMAN_VAULT"]))
    engine = get_engine(force=True)
    assert engine is not None, "P15 拓扑图加载失败"
    qs = _questions()

    rows, same, blanks_off, blanks_on, ms_off, ms_on, jac = [], 0, 0, 0, [], [], []
    for q in qs:
        off = _run(vault, q, None)
        on = _run(vault, q, engine)
        same += off["line"] == on["line"]
        blanks_off += off["blank"]
        blanks_on += on["blank"]
        ms_off.append(off["ms"])
        ms_on.append(on["ms"])
        a, b = set(x for x in off["top3"] if x), set(x for x in on["top3"] if x)
        jac.append(len(a & b) / len(a | b) if (a | b) else 1.0)
        rows.append({"query": q, "off": off["line"], "on": on["line"],
                     "off_blank": off["blank"], "on_blank": on["blank"],
                     "top3_jaccard": round(jac[-1], 3)})

    n = len(qs)
    summary = {
        "vault": os.environ["QBOT_FEYNMAN_VAULT"],
        "topology_hash": engine.topo_hash,
        "questions": n,
        "identical_top_line": same,
        "identical_top_line_rate": round(same / n, 3),
        "blank_rate_off": round(blanks_off / n, 3),
        "blank_rate_on": round(blanks_on / n, 3),
        "top3_jaccard_mean": round(sum(jac) / n, 3),
        "ms_off_mean": round(sum(ms_off) / n, 1),
        "ms_on_mean": round(sum(ms_on) / n, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ 明细 → {OUT}")


if __name__ == "__main__":
    main()
