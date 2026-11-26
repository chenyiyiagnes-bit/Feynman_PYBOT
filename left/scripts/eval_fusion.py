"""P0 评测台：固定查询集 × 融合/拓扑/类比/条件级联配置，与全基线逐项对比。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/eval_fusion.py [--out data/eval_fusion_p8b.json]

设计要点：
- 查询集固定写死（QUERY_SET_VERSION），改动必须升版本，旧报告不可混用；
- 每个查询同一 seed（nonce 固定），配置之间只差被测参数；
- 指标全部相对 baseline（linear + 全零算子 + 无拓扑 + 无级联）计算：
  top-1 翻转率、top-5 Jaccard、最高概率、拓扑覆盖率/三角形率、
  乘子分布（原始/归一后）、条件乘子分布、双重惩罚占比。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from plugins.feynman_typesetter import (
    OperatorWeights,
    PrintRequest,
    Vault,
    _operator_multiplier,
    _operator_scores,
    run_feynman_print,
)
from plugins.topology_builder import TopologyManifoldEngine

VAULT_PATH = Path("data/feynman_style_vault_p8b.json")
TOPOLOGY_PATH = Path("data/topology_graph_p8c.npz")
QUERY_SET_VERSION = "p8b-48-v1"

# 固定查询集：日常物×物理、纯物理、过程/因果、测量/观测、抽象概念各占一部分。
QUERIES = [
    "电子为什么会发光", "光子有质量吗", "引力为什么这么弱", "熵为什么会增加",
    "能量守恒是什么意思", "动量为什么守恒", "温度升高以后会怎样", "磁场能看见吗",
    "量子是什么东西", "波粒二象性怎么理解", "相对论为什么会改变时间", "时空为什么会弯曲",
    "概率在物理里是什么", "测量为什么会改变结果", "真空里有什么", "光速为什么不变",
    "质量从哪里来", "电荷为什么会互相吸引", "原子里面有什么", "分子为什么会振动",
    "摩擦力为什么会生热", "声音为什么能在空气里传", "水为什么会结冰", "冰为什么会浮在水上",
    "天空为什么是蓝的", "彩虹是怎么来的", "太阳为什么会发光", "星星为什么会闪烁",
    "潮汐是怎么发生的", "指南针为什么指向北", "电流为什么会发热", "电池是怎么存电的",
    "磁铁为什么能吸铁", "镜子为什么会反光", "透镜为什么能聚焦", "棱镜为什么分光",
    "热量是怎么传过来的", "声音在真空里会怎样", "气球为什么能飞起来", "飞机为什么能飞",
    "陀螺为什么不会倒", "弹簧为什么会弹回来", "钟摆为什么越摆越慢", "无线电波是怎么传的",
    "核能是怎么释放的", "激光为什么那么亮", "半导体为什么能导电", "超导体为什么没有电阻",
]

# 预设算子权重：中等强度，让 Boltzmann 有东西可区分，又不至于全池贴地。
PRESET = OperatorWeights(
    micro=0.6, cross=0.8, boundary=0.3, reversal=0.4,
    anti_intuition=0.6, jargon_strip=0.4, least_action=0.5, cargo_cult=0.3,
)

CONFIGS = {
    "baseline": {},
    "linear_preset": {"operators": PRESET},
    "boltzmann_t0.5": {"operators": PRESET, "fusion": "boltzmann", "fusion_tau": 0.5},
    "boltzmann_t1": {"operators": PRESET, "fusion": "boltzmann", "fusion_tau": 1.0},
    "boltzmann_t2": {"operators": PRESET, "fusion": "boltzmann", "fusion_tau": 2.0},
    "topology": {"use_topology": True},
    "analogy": {"use_topology": True, "style_deg": 90.0},
    "conditioned": {"conditioned_chain": True},
    "conditioned_topology": {"conditioned_chain": True, "use_topology": True},
}


def _topk_lines(result: dict, k: int = 5) -> list[str]:
    return [p.get("rendered", p.get("line", "")) for p in result.get("paths", [])[:k]]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _mult_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "stdev": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def _path_metrics(result: dict) -> dict:
    paths = result.get("paths", [])
    # 归一后写回的是 multiplier（相对池中位数），原始值留在 multiplier_raw
    topo_norm = [float(p["topology"]["multiplier"]) for p in paths if p.get("topology")]
    topo_raw = [float(p["topology"].get("multiplier_raw", p["topology"]["multiplier"]))
                for p in paths if p.get("topology")]
    simplex = [p for p in paths if (p.get("topology") or {}).get("is_simplex")]
    covered = [p for p in paths if (p.get("topology") or {}).get("covered")]
    pair_ev = [p for p in paths if float((p.get("topology") or {}).get("pair_score", 0.0)) > 0]
    arbitrated = [p for p in paths
                  if (p.get("topology") or {}).get("double_penalty_arbitrated")]
    analogy = [p["topology"]["analogy_sim"] for p in paths
               if (p.get("topology") or {}).get("analogy_sim") is not None]
    cond = [float(p["conditioned_multiplier"]) for p in paths
            if p.get("conditioned_multiplier") is not None]
    # 双重惩罚：条件乘子与拓扑乘子同时 <1 的路径占比及其复合乘子
    both = [float(p["conditioned_multiplier"]) * float(p["topology"]["multiplier"])
            for p in paths
            if p.get("conditioned_multiplier") is not None and p.get("topology")]
    double = [v for v in both if v < 1.0]

    op_mult = []
    for p in paths:
        scores = p.get("operator_scores") or {}
        if scores:
            op_mult.append(_operator_multiplier(scores, PRESET, fusion="boltzmann", tau=1.0))

    return {
        "topology_coverage": round(len(covered) / len(paths), 4) if paths else 0.0,
        "pair_evidence_coverage": round(len(pair_ev) / len(paths), 4) if paths else 0.0,
        "simplex_rate": round(len(simplex) / len(paths), 4) if paths else 0.0,
        "simplex_hits": len(simplex),
        "arbitrated_rate": round(len(arbitrated) / len(paths), 4) if paths else 0.0,
        "topo_mult_raw": _mult_stats(topo_raw),
        "topo_mult_recorded": _mult_stats(topo_norm),
        "analogy_sim": _mult_stats(analogy),
        "conditioned_mult": _mult_stats(cond),
        "double_penalty_rate": round(len(double) / len(both), 4) if both else 0.0,
        "double_penalty_mult": _mult_stats(both),
        "boltzmann_mult": _mult_stats(op_mult),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/eval_fusion_p8b.json")
    ap.add_argument("--queries", type=int, default=len(QUERIES))
    ap.add_argument("--vault", default=str(VAULT_PATH))
    ap.add_argument("--topology", default=str(TOPOLOGY_PATH))
    args = ap.parse_args()

    vault = Vault(Path(args.vault))
    engine = TopologyManifoldEngine.load(Path(args.topology))
    queries = QUERIES[: args.queries]

    report: dict = {
        "query_set_version": QUERY_SET_VERSION,
        "queries": len(queries),
        "vault": args.vault,
        "topology": engine.stats() if engine else None,
        "configs": {},
    }
    baseline_lines: dict[str, list[str]] = {}
    baseline_top1: dict[str, str] = {}

    for name, overrides in CONFIGS.items():
        rows = {
            "top1_flips": 0, "jaccard": [], "top_probability": [],
            "path_metrics": [], "greeting": 0, "collapsed": 0,
        }
        for q in queries:
            req = PrintRequest(user_query=q, nonce=0, top_k=10, **overrides)
            topo = engine if req.use_topology or name == "analogy" else None
            res = run_feynman_print(vault, req, ext_hash="", topology=topo,
                                    condition_chain_hint=None) if False else run_feynman_print(
                vault, req, ext_hash="", topology=topo)
            if res.get("greeting"):
                rows["greeting"] += 1
                continue
            rows["collapsed"] += 1 if res.get("collapsed") else 0
            lines = _topk_lines(res)
            top1 = res.get("top_resonant_line", "")
            if name == "baseline":
                baseline_lines[q] = lines
                baseline_top1[q] = top1
            else:
                if top1 != baseline_top1.get(q, ""):
                    rows["top1_flips"] += 1
                rows["jaccard"].append(_jaccard(lines, baseline_lines.get(q, [])))
            if res.get("top_probability") is not None:
                rows["top_probability"].append(float(res["top_probability"]))
            rows["path_metrics"].append(_path_metrics(res))

        n = max(1, rows["collapsed"])
        agg = {k: rows[k] for k in ("greeting", "collapsed")}
        agg["top1_flip_rate"] = round(rows["top1_flips"] / n, 4)
        agg["top5_jaccard_mean"] = round(statistics.fmean(rows["jaccard"]), 4) \
            if rows["jaccard"] else 1.0
        agg["top_probability_mean"] = round(statistics.fmean(rows["top_probability"]), 6) \
            if rows["top_probability"] else None
        agg["metrics_mean"] = {}
        keys = sorted({k for m in rows["path_metrics"] for k in m})
        for k in keys:
            vals = [m[k] for m in rows["path_metrics"] if isinstance(m.get(k), (int, float))]
            if vals:
                agg["metrics_mean"][k] = round(statistics.fmean(vals), 4)
        # 分布类指标合并所有路径后再统计，避免每查询小样本噪声
        pools: dict[str, list[float]] = {}
        for m in rows["path_metrics"]:
            for k in ("topo_mult_raw", "topo_mult_recorded", "analogy_sim",
                      "conditioned_mult", "boltzmann_mult", "double_penalty_mult"):
                d = m.get(k) or {}
                for f in ("mean", "median", "min", "max"):
                    if f in d:
                        pools.setdefault(f"{k}.{f}", []).append(d[f])
        agg["distributions"] = {k: round(statistics.fmean(v), 4)
                                for k, v in sorted(pools.items())}
        report["configs"][name] = agg

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
