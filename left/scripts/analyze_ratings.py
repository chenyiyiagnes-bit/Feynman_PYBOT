"""分析评分工作表：每个配置的平均分、自助法 95% 置信区间、与 baseline 的对比。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/analyze_ratings.py \
        --worksheet data/rating_worksheet_p8b.jsonl --out data/rating_analysis_p8b.json

规则：
- rating 必须是 1–5 的整数；-1（无法判断）和空值不参与统计但计入未评数；
- 少于 MIN_RATED 条已评行时只输出描述性计数，拒绝给"哪个配置更好"的结论；
- 任一配置有效评分 < MIN_PER_CONFIG 条时，该配置的均值标注为低样本、不参与推荐；
- 推荐只在 Bootstrap 差值 95% CI 不含 0 且方向为正时给出，否则如实说"无显著差异"。
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

MIN_RATED = 60
MIN_PER_CONFIG = 15
BOOTSTRAP_N = 2000
SEED = 20261125


def _load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _boot_ci(values: list[float], rng: random.Random) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(BOOTSTRAP_N):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return (means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet", default="data/rating_worksheet_p8b.jsonl")
    ap.add_argument("--out", default="data/rating_analysis_p8b.json")
    ap.add_argument("--config-map", default=None,
                    help="盲评工作表：id -> configs 的映射文件，缺失该字段时用它补上")
    ap.add_argument("--reference", default="baseline",
                    help="对照配置名；旧表用 baseline，P15 决策表用 p10_lock_retrieval")
    args = ap.parse_args()

    rows = _load(Path(args.worksheet))
    if args.config_map:
        cmap = json.loads(Path(args.config_map).read_text(encoding="utf-8"))
        for r in rows:
            if "configs" not in r:
                r["configs"] = cmap.get(r.get("id"), [])
    rated, unrated, invalid = [], 0, 0
    for r in rows:
        v = r.get("rating")
        if v is None:
            unrated += 1
        elif isinstance(v, int) and (1 <= v <= 5 or v == -1):
            if v == -1:
                unrated += 1
            else:
                rated.append(r)
        else:
            invalid += 1

    by_cfg: dict[str, list[float]] = {}
    for r in rated:
        for cfg in r.get("configs") or []:
            by_cfg.setdefault(cfg, []).append(float(r["rating"]))

    rng = random.Random(SEED)
    stats = {}
    for cfg, vals in sorted(by_cfg.items()):
        lo, hi = _boot_ci(vals, rng)
        stats[cfg] = {
            "n": len(vals),
            "mean": round(statistics.fmean(vals), 3),
            "ci95": [round(lo, 3), round(hi, 3)],
            "low_sample": len(vals) < MIN_PER_CONFIG,
        }

    ref_name = args.reference
    # 按问句配对的差值：同一问句下两档各取平均分再相减，控制问句难度差异
    per_query: dict[str, dict[str, list[float]]] = {}
    for r in rated:
        for cfg in r.get("configs") or []:
            per_query.setdefault(r.get("query", ""), {}).setdefault(cfg, []).append(
                float(r["rating"]))

    recommendation = None
    reason = ""
    if len(rated) < MIN_RATED:
        reason = (f"已评 {len(rated)} 条，低于最低门槛 {MIN_RATED}；"
                  f"只输出描述统计，不给配置结论。")
    else:
        base = by_cfg.get(ref_name, [])
        if not base:
            reason = f"工作表里没有对照配置 {ref_name}，无法给出结论。"
        best, best_diff = None, None
        for cfg, vals in by_cfg.items():
            if cfg == ref_name or len(vals) < MIN_PER_CONFIG or len(base) < MIN_PER_CONFIG:
                continue
            # 独立样本差值；CI 用同一 bootstrap 流程估计
            diffs = []
            for _ in range(BOOTSTRAP_N):
                diffs.append(statistics.fmean(rng.choice(vals) for _ in vals)
                             - statistics.fmean(rng.choice(base) for _ in base))
            diffs.sort()
            lo, hi = diffs[int(0.025 * BOOTSTRAP_N)], diffs[int(0.975 * BOOTSTRAP_N)]
            stats[cfg][f"vs_{ref_name}_diff_ci95"] = [round(lo, 3), round(hi, 3)]

            # 配对差值（按问句），优先用于结论
            pdiffs = []
            for q, m in per_query.items():
                if cfg in m and ref_name in m:
                    pdiffs.append(statistics.fmean(m[cfg]) - statistics.fmean(m[ref_name]))
            if len(pdiffs) >= MIN_PER_CONFIG:
                boot = []
                for _ in range(BOOTSTRAP_N):
                    boot.append(statistics.fmean(rng.choice(pdiffs) for _ in pdiffs))
                boot.sort()
                plo, phi = boot[int(0.025 * BOOTSTRAP_N)], boot[int(0.975 * BOOTSTRAP_N)]
                stats[cfg]["paired"] = {
                    "queries": len(pdiffs),
                    "mean_diff": round(statistics.fmean(pdiffs), 3),
                    "ci95": [round(plo, 3), round(phi, 3)],
                }
                if plo > 0 and (best_diff is None or plo > best_diff):
                    best, best_diff = cfg, plo
        if best:
            recommendation = best
            reason = (f"{best} 相对 {ref_name} 的配对差值 95% CI 下限为 "
                      f"{best_diff:.3f}，显著为正。")
        else:
            reason = (f"没有任何配置相对 {ref_name} 的配对差值 95% CI "
                      f"完全落在正区间，视为无显著差异。")

    report = {
        "worksheet_rows": len(rows),
        "rated": len(rated),
        "unrated": unrated,
        "invalid": invalid,
        "min_rated": MIN_RATED,
        "min_per_config": MIN_PER_CONFIG,
        "reference": ref_name,
        "stats": stats,
        "recommendation": recommendation,
        "reason": reason,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
