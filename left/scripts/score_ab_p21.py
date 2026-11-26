"""P21 盲评揭盲与汇总：读评分表 × 钥匙，算两版均分、配对差与胜负。

评分由 Claude 模型给出（同 P16 的做法，不是人评），评的是"是否贴题、是否顺畅"，
不评物理正确性。脚本只汇总，不改评分。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/score_ab_p21.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

KEY = Path("data/ab_p21_key.json")
SCORES = Path("data/ab_p21_scores_model.jsonl")
OUT_JSON = Path("data/ab_p21_summary.json")
OUT_MD = Path("data/ab_p21_report.md")
OOV_PAIRS = 16  # 前 16 组是语料外实体问句


def main() -> None:
    key = json.loads(KEY.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in SCORES.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    base_scores, p21_scores, diffs = [], [], []
    wins = {"p21": 0, "baseline": 0, "tie": 0}
    groups = {"oov": {"base": [], "p21": []}, "vault": {"base": [], "p21": []}}
    for r in rows:
        k = key[r["pair"]]
        a_variant, b_variant = k["A"], k["B"]
        score = {a_variant: r["A"], b_variant: r["B"]}
        base, p21 = score["baseline"], score["p21"]
        base_scores.append(base)
        p21_scores.append(p21)
        diffs.append(p21 - base)
        if p21 > base:
            wins["p21"] += 1
        elif p21 < base:
            wins["baseline"] += 1
        else:
            wins["tie"] += 1
        idx = int(r["pair"].split("_")[1])
        grp = "oov" if idx <= OOV_PAIRS else "vault"
        groups[grp]["base"].append(base)
        groups[grp]["p21"].append(p21)

    def mean(xs):
        return round(statistics.fmean(xs), 3) if xs else None

    summary = {
        "pairs": len(rows),
        "judge": "claude-model",
        "baseline_mean": mean(base_scores),
        "p21_mean": mean(p21_scores),
        "paired_mean_diff": mean(diffs),
        "wins": wins,
        "groups": {
            g: {"baseline": mean(v["base"]), "p21": mean(v["p21"]),
                "n": len(v["base"])} for g, v in groups.items()
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    lines = ["# P21 段落盲评（模型评，非人评）", "",
             f"{summary['pairs']} 组；评分 1–5（是否贴题、是否顺畅，不评物理正确性）。", "",
             "| | 基线 | 开启 P21 |", "|---|---|---|",
             f"| 均分 | {summary['baseline_mean']} | {summary['p21_mean']} |",
             f"| 组数 | {summary['pairs']} | {summary['pairs']} |", "",
             f"配对平均差（P21 − 基线）：**{summary['paired_mean_diff']:+.3f}**", "",
             f"胜负：P21 {wins['p21']} 胜 / 基线 {wins['baseline']} 胜 / 平 {wins['tie']}", "",
             "| 问句组 | 基线 | 开启 P21 | 组数 |", "|---|---|---|---|"]
    for g, v in summary["groups"].items():
        label = "语料外实体" if g == "oov" else "语料内实体"
        lines.append(f"| {label} | {v['baseline']} | {v['p21']} | {v['n']} |")
    lines += ["", "判读：均分差小于 0.2 视为同水平；胜负集中在实体是否守住问句，"
              "与 P16 的实体漂移问题同源。机器评分只作旁证，结论以人评为准。", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
