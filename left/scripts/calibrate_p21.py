"""P21 相容乘子标定：在开发集问句上量 score 分布与池内分位乘子的行为。

只读 P15 字库与 P21 基底，不写任何产物之外的运行时代码。标定结果写进
`data/p21_calibration.json` / `.md`，供人工核对后再改 `plugins/isomorphism_p21.py`
里的 COMPAT_CENTER / COMPAT_SLOPE。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/calibrate_p21.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.isomorphism_p21 import load_engine

VAULT = Path("data/feynman_style_vault_p15.json")
OUT_JSON = Path("data/p21_calibration.json")
OUT_MD = Path("data/p21_calibration.md")

OOV_QUERIES = [
    "股票为什么会下跌", "猫为什么喜欢纸箱", "咖啡为什么会变凉", "合同为什么要有条款",
    "台风为什么会转向", "公司为什么会长大", "算法为什么会收敛", "音乐为什么好听",
    "城市为什么变热", "货币为什么会贬值", "钥匙为什么能开锁", "沙子为什么烫脚",
    "影子为什么跟着人走", "面包为什么会发霉", "电梯为什么失重", "镜子为什么会反光",
]


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main() -> None:
    vault = Vault(VAULT)
    engine = load_engine()
    if engine is None:
        raise SystemExit("缺少 data/isomorphism_p21.json，先跑 build_isomorphism_p21.py")

    queries = list(OOV_QUERIES)
    try:
        from scripts.ab_tension_blind import build_queries
        queries += [q for q in build_queries(vault) if q not in queries]
    except Exception:
        pass

    rows = []
    flipped = 0
    entity_hit = {"off": 0, "on": 0}
    blank = {"off": 0, "on": 0}
    total = 0
    for i, q in enumerate(queries):
        for nonce in (0, 1, 2):
            base = dict(user_query=q, style_deg=60.0, nonce=100 + i * 7 + nonce)
            off = run_feynman_print(vault, PrintRequest(**base))
            on = run_feynman_print(vault, PrintRequest(**base, use_isomorphism=True,
                                                       top_k=50))
            total += 1
            off_line = off.get("printed_collapse")
            on_line = on.get("printed_collapse")
            if off_line != on_line:
                flipped += 1
            for key, out in (("off", off), ("on", on)):
                if not out.get("collapsed"):
                    blank[key] += 1
            ents = on.get("user_entities") or []
            if ents:
                if ents[0] in (off_line or ""):
                    entity_hit["off"] += 1
                if ents[0] in (on_line or ""):
                    entity_hit["on"] += 1
            for p in on.get("paths") or []:
                info = p.get("p21")
                if not info or info.get("score") is None:
                    continue
                rows.append({
                    "query": q,
                    "nonce": nonce,
                    "line": p["line"],
                    "score": info["score"],
                    "multiplier": info["multiplier"],
                    "ea": info.get("entity_action"),
                    "ac": info.get("action_consequence"),
                    "em": info.get("entity_mechanism"),
                })

    scores = [r["score"] for r in rows]
    mults = [r["multiplier"] for r in rows]
    summary = {
        "queries": len(queries),
        "paths_scored": len(rows),
        "distinct_collapses": flipped,
        "blank_off": blank["off"],
        "blank_on": blank["on"],
        "user_entity_in_line": entity_hit,
        "user_entity_line_total": total,
        "scored_paths_per_query": round(len(rows) / max(1, total), 2),
        "score": {
            "min": round(min(scores), 4) if scores else None,
            "p10": round(percentile(scores, 0.10), 4) if scores else None,
            "p25": round(percentile(scores, 0.25), 4) if scores else None,
            "median": round(statistics.median(scores), 4) if scores else None,
            "p75": round(percentile(scores, 0.75), 4) if scores else None,
            "p90": round(percentile(scores, 0.90), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
        },
        "multiplier": {
            "median": round(statistics.median(mults), 4) if mults else None,
            "share_above_1": round(sum(1 for m in mults if m > 1.001) / len(mults), 4) if mults else None,
            "share_below_1": round(sum(1 for m in mults if m < 0.999) / len(mults), 4) if mults else None,
        },
        "engine_meta": {
            "k": engine.meta.get("k"),
            "words": engine.meta.get("words"),
            "corpus_sha256": (engine.meta.get("corpus_sha256") or "")[:16],
        },
    }
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    s = summary["score"]
    lines = [
        "# P21 相容乘子标定（开发集）", "",
        f"问句 {summary['queries']} 条 × 3 个 nonce；有 p21 评分的路径 {summary['paths_scored']} 条；",
        f"开关改变打印结果的次数 {flipped}/{len(queries) * 3}；"
        f"用户实体入句 {entity_hit['off']}（关）→ {entity_hit['on']}（开）。", "",
        "| 分位 | score |", "|---|---|",
        f"| p10 | {s['p10']} |", f"| p25 | {s['p25']} |", f"| 中位 | {s['median']} |",
        f"| p75 | {s['p75']} |", f"| p90 | {s['p90']} |",
        f"| min / max | {s['min']} / {s['max']} |", "",
        f"乘子（池内分位）中位 {summary['multiplier']['median']}，"
        f"高于 1 的占 {summary['multiplier']['share_above_1']}，"
        f"低于 1 的占 {summary['multiplier']['share_below_1']}。", "",
        "判读：乘子按同一请求内 score 的分位施加（1 + 1.2·(pct−0.5)，夹到 [0.6, 1.6]），",
        "不再依赖绝对尺度，也不需要手挑中心；单条路径 pct=0.5、乘子 1.0。",
        "score 的绝对分位只用于观察区分度：p25–p75 跨度小说明 12 维余弦分辨力有限，",
        "排序信息仍然可用，但别把绝对差当质量差。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
