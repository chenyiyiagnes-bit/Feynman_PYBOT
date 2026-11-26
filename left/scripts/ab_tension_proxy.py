"""P16 张力弧的机械代理评估：不替人打分，只量可自动核对的指标。

对 40 个同源问句各渲染"关闭/开启张力弧"两段，统计：
1. 完整率（三句是否凑齐）、起手式重复、连接词后叠起手式；
2. 问句谓语命中（动作槽是否落在用户问的那件事上）；
3. 张力弧是否单调下降、T1/T3 均值是否落在设计区间。

结论只用于排除"开启后明显更差"的回归；阅读偏好仍需人评，
脚本不会输出"哪版更好"，只给机械指标对照。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ab_tension_proxy.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from plugins.feynman_typesetter import Vault, extract_query_verbs
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
from scripts.ab_tension_blind import build_queries

VAULT_PATH = Path("data/feynman_style_vault_p15.json")
OUT_MD = Path("data/ab_tension_proxy.md")
OUT_JSON = Path("data/ab_tension_proxy.json")
STYLE_DEG = 60.0
BEAM = 6
HOOK_HEADS = ("我们看到", "在我看来", "你看", "注意")


def metrics(out: dict, query: str) -> dict:
    sentences = out.get("sentences") or []
    hooks = [(s.get("glyphs") or {}).get("hook") for s in sentences]
    actions = [(s.get("glyphs") or {}).get("action") for s in sentences]
    verbs = set(extract_query_verbs(query))
    tensions = [s.get("tension") for s in sentences]
    return {
        "complete": bool(out.get("complete")),
        "dup_hooks": max(0, len(hooks) - len(set(hooks))),
        "verb_hit": bool(verbs) and any(a in verbs for a in actions),
        "double_open": sum(
            1 for s in sentences
            if s.get("connector") and any((s.get("line") or "").startswith(h)
                                          for h in HOOK_HEADS)),
        "monotone": (all(tensions[i] >= tensions[i + 1] for i in range(len(tensions) - 1))
                     if len(tensions) == 3 and all(t is not None for t in tensions)
                     else None),
        "T1": (out.get("tension_arc") or {}).get("T1"),
        "T3": (out.get("tension_arc") or {}).get("T3"),
    }


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def main() -> None:
    vault = Vault(VAULT_PATH)
    queries = build_queries(vault)
    rows = {"off": [], "on": []}
    identical = 0
    for i, q in enumerate(queries, 1):
        base = dict(user_query=q, style_deg=STYLE_DEG, nonce=100 + i, beam_width=BEAM)
        off = weave_paragraph(vault, ParagraphRequest(**base, dissonance_resolution=False))
        on = weave_paragraph(vault, ParagraphRequest(**base, dissonance_resolution=True))
        identical += off["paragraph"] == on["paragraph"]
        rows["off"].append(metrics(off, q))
        rows["on"].append(metrics(on, q))

    summary = {}
    for key, items in rows.items():
        summary[key] = {
            "complete": sum(x["complete"] for x in items),
            "dup_hooks": sum(x["dup_hooks"] for x in items),
            "verb_hit": sum(1 for x in items if x["verb_hit"]),
            "double_open": sum(x["double_open"] for x in items),
            "monotone": sum(1 for x in items if x["monotone"]),
            "T1_mean": _mean([x["T1"] for x in items]),
            "T3_mean": _mean([x["T3"] for x in items]),
        }
    payload = {"queries": len(queries), "identical_paragraphs": identical,
               "summary": summary}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(queries)
    lines = [
        "# P16 张力弧机械代理评估（非人评）", "",
        f"问句 {n} 条，同 nonce 渲染关闭/开启两版；两版文本完全相同 {identical} 条。",
        "本表只查回归：完整率、重复起手式、双开头、问句谓语命中、弧是否单调。",
        "阅读偏好仍须人评，`data/ab_tension_pairs.md` 的评分未回收前不下胜负结论。", "",
        "| 指标 | 关闭 | 开启 |", "|---|---|---|",
        f"| 三句完整 | {summary['off']['complete']}/{n} | {summary['on']['complete']}/{n} |",
        f"| 重复起手式（段/次） | {summary['off']['dup_hooks']} | {summary['on']['dup_hooks']} |",
        f"| 连接词后叠起手式 | {summary['off']['double_open']} | {summary['on']['double_open']} |",
        f"| 问句谓语命中 | {summary['off']['verb_hit']}/{n} | {summary['on']['verb_hit']}/{n} |",
        f"| 张力弧单调下降 | {summary['off']['monotone']}/{n} | {summary['on']['monotone']}/{n} |",
        f"| T1 均值 | {summary['off']['T1_mean']} | {summary['on']['T1_mean']} |",
        f"| T3 均值 | {summary['off']['T3_mean']} | {summary['on']['T3_mean']} |",
        "",
        "判读规则（预先写死）：开启版若在完整率、双开头、谓语命中任一项低于关闭版，",
        "即视为回归、不得晋升；机械指标不构成阅读偏好证据。",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ {OUT_MD} / {OUT_JSON}：{n} 组，文本相同 {identical}")


if __name__ == "__main__":
    main()
