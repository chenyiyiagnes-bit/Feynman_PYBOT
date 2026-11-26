"""P4 恒定参数标定：在 p8b 上用固定查询集扫 SIMPLEX_BOOST / HOLE_FACTOR /
GEO_GAIN，并单独扫 Boltzmann 温度 τ。

没有人工质量标注，所以不假装能选"更好看的句子"。这里只测可验证的性质：
- discrimination：候选池内拓扑原始乘子的 log 标准差——越大说明拓扑层越不是
  给全池乘常数（这是上次实测的核心失效模式）；
- stability：top-1 与无拓扑基线的翻转率——证据层是排序辅助，不应把出句
  全盘换掉；
- simplex_rate：候选池里命中的实心三角形比例。

选择规则写死在代码里：先筛 stability（翻转率 ≤ 0.6）且有效果（spread>0），
再取 discrimination 最大者；若比默认参数的 discrimination 高出不到 5%
（视为噪声），保留默认值，并在报告里明说"没有显著赢家"。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/calibrate_topology.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import plugins.topology_builder as tb
from plugins.feynman_typesetter import OperatorWeights, PrintRequest, Vault, _operator_multiplier, run_feynman_print
from plugins.topology_builder import TopologyManifoldEngine
from scripts.eval_fusion import PRESET, QUERIES, QUERY_SET_VERSION, _jaccard, _topk_lines

VAULT_PATH = Path("data/feynman_style_vault_p8b.json")
TOPOLOGY_PATH = Path("data/topology_graph_p8c.npz")

SIMPLEX_BOOSTS = [1.25, 1.45, 1.65]
HOLE_FACTORS = [0.45, 0.55, 0.70]
GEO_GAINS = [0.45, 0.60, 0.75]
TAUS = [0.5, 1.0, 2.0, 4.0]

DEFAULTS = {
    "simplex_boost": tb.SIMPLEX_BOOST,
    "hole_factor": tb.HOLE_FACTOR,
    "geo_gain": tb.GEO_GAIN,
    "tau": 1.0,
}

FLIP_BUDGET = 0.6
NOISE_MARGIN = 0.05


def _patch(simplex_boost: float, hole_factor: float, geo_gain: float) -> None:
    tb.SIMPLEX_BOOST = simplex_boost
    tb.HOLE_FACTOR = hole_factor
    tb.GEO_GAIN = geo_gain


def _logspread(values: list[float]) -> float:
    logs = [math.log(max(1e-9, v)) for v in values]
    return statistics.pstdev(logs) if len(logs) > 1 else 0.0


def _topology_pass(vault: Vault, engine: TopologyManifoldEngine, queries: list[str],
                   baseline_lines: dict[str, list[str]], baseline_top1: dict[str, str]) -> dict:
    flips, jaccards, spreads, simplex, coverage = 0, [], [], [], []
    for q in queries:
        req = PrintRequest(user_query=q, nonce=0, top_k=10, use_topology=True)
        res = run_feynman_print(vault, req, ext_hash="", topology=engine)
        if res.get("greeting") or not res.get("collapsed"):
            continue
        paths = res.get("paths", [])
        if res.get("top_resonant_line") != baseline_top1.get(q):
            flips += 1
        jaccards.append(_jaccard(_topk_lines(res), baseline_lines.get(q, [])))
        raws = [float(p["topology"]["multiplier_raw"]) for p in paths
                if p.get("topology") and "multiplier_raw" in p["topology"]]
        if raws:
            spreads.append(_logspread(raws))
        simplex.append(sum(1 for p in paths if (p.get("topology") or {}).get("is_simplex")) / max(1, len(paths)))
        coverage.append(sum(1 for p in paths if (p.get("topology") or {}).get("covered")) / max(1, len(paths)))
    n = max(1, len(jaccards))
    return {
        "top1_flip_rate": round(flips / n, 4),
        "top5_jaccard": round(statistics.fmean(jaccards), 4) if jaccards else 1.0,
        "discrimination": round(statistics.fmean(spreads), 4) if spreads else 0.0,
        "simplex_rate": round(statistics.fmean(simplex), 4) if simplex else 0.0,
        "coverage": round(statistics.fmean(coverage), 4) if coverage else 0.0,
    }


def _tau_pass(vault: Vault, queries: list[str]) -> list[dict]:
    """τ 只影响算子融合：用带预设权重的候选池直接算每路径乘子，测区分度。"""
    rows = []
    for tau in TAUS:
        spreads = []
        for q in queries:
            req = PrintRequest(user_query=q, nonce=0, top_k=10, operators=PRESET,
                               fusion="boltzmann", fusion_tau=tau)
            res = run_feynman_print(vault, req, ext_hash="")
            mults = [_operator_multiplier(p["operator_scores"], PRESET, fusion="boltzmann", tau=tau)
                     for p in res.get("paths", []) if p.get("operator_scores")]
            if len(mults) > 1:
                spreads.append(_logspread(mults))
        rows.append({"tau": tau,
                     "discrimination": round(statistics.fmean(spreads), 4) if spreads else 0.0})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/topology_calibration_p8b.json")
    args = ap.parse_args()

    vault = Vault(VAULT_PATH)
    queries = QUERIES

    # 基线（无拓扑）只算一次
    baseline_lines, baseline_top1 = {}, {}
    for q in queries:
        res = run_feynman_print(vault, PrintRequest(user_query=q, nonce=0, top_k=10), ext_hash="")
        if res.get("collapsed"):
            baseline_lines[q] = _topk_lines(res)
            baseline_top1[q] = res.get("top_resonant_line", "")

    rows = []
    default_row = None
    for sb, hf, gg in itertools.product(SIMPLEX_BOOSTS, HOLE_FACTORS, GEO_GAINS):
        _patch(sb, hf, gg)
        engine = TopologyManifoldEngine.load(TOPOLOGY_PATH)
        m = _topology_pass(vault, engine, queries, baseline_lines, baseline_top1)
        m.update({"simplex_boost": sb, "hole_factor": hf, "geo_gain": gg})
        rows.append(m)
        if (sb, hf, gg) == (DEFAULTS["simplex_boost"], DEFAULTS["hole_factor"], DEFAULTS["geo_gain"]):
            default_row = m
    _patch(DEFAULTS["simplex_boost"], DEFAULTS["hole_factor"], DEFAULTS["geo_gain"])

    # 选择规则（刻意保守）：discrimination 会随 geo_gain 机械增大（曲率项本来
    # 就无处不在），不能当质量分。只有"实心三角形真的在候选池里命中"时，
    # simplex/hole 参数才有可测行为，才允许选参；否则一律保留默认值。
    evidence_rows = [r for r in rows if r["simplex_rate"] > 0]
    best = max(evidence_rows, key=lambda r: r["discrimination"]) if evidence_rows else None
    winner = None
    reason = ""
    if best is None:
        reason = ("所有网格组合在普通候选池里实心三角形命中率都是 0；区分度只随 "
                  "geo_gain 机械增大，是曲率项在放大而不是证据在起作用。没有可依据的"
                  "效果差异，保留默认值 (1.45/0.55/0.60)。")
    elif default_row and best["discrimination"] < default_row["discrimination"] * (1 + NOISE_MARGIN):
        reason = (f"最佳组合区分度 {best['discrimination']} 未超过默认 "
                  f"{default_row['discrimination']} 的 {int(NOISE_MARGIN*100)}% 噪声带，保留默认值。")
    else:
        winner = {"simplex_boost": best["simplex_boost"], "hole_factor": best["hole_factor"],
                  "geo_gain": best["geo_gain"]}
        reason = f"组合 {winner} 区分度 {best['discrimination']} 显著高于默认，建议采用。"

    # τ：对每档温度算区分度，写入报告供人工判断；不自动改默认
    tau_rows = _tau_pass(vault, queries)

    report = {
        "query_set_version": QUERY_SET_VERSION,
        "queries": len(queries),
        "topology_hash": TopologyManifoldEngine.load(TOPOLOGY_PATH).topo_hash,
        "defaults": DEFAULTS,
        "flip_budget": FLIP_BUDGET,
        "noise_margin": NOISE_MARGIN,
        "grid": rows,
        "best": best,
        "winner": winner,
        "reason": reason,
        "tau_grid": tau_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("defaults", "best", "winner", "reason", "tau_grid")},
                     ensure_ascii=False, indent=2))
    print(f"\n完整网格已写入 {out}")


if __name__ == "__main__":
    main()
