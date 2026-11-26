"""按语体相位动态束宽的消融：同一批问题，逐档 beam_width 跑段落。

只回答可数的问题：
- 每个语体档下，束宽 3/4/6/8/10 的留白率、延迟、跨问题多样性（不同句数）；
- 相对静态 6，各档首句变化率（束宽是否真在改结果，还是只改内部路径）；
- 相邻档位的三句集合重合度（收益是否递减）。

不判断"哪种更好看"——那一步只能在自动指标无损或更好时再进盲评。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ablate_adaptive_beam_p15.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_VAULT"] = "data/feynman_style_vault_p15.json"
os.environ["QBOT_FEYNMAN_TOPOLOGY"] = "data/topology_graph_p15.npz"

from plugins.feynman_typesetter import BLANK_GLYPH, Vault  # noqa: E402
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph  # noqa: E402
from plugins.topology_builder import get_engine  # noqa: E402

WORKSHEET = Path("data/rating_worksheet_p15_blind.jsonl")
OUT = Path("data/eval_adaptive_beam_p15.json")
STYLES = (15.0, 60.0, 105.0)
WIDTHS = (3, 4, 6, 8, 10)
BASELINE = 6


def _questions() -> list[str]:
    seen: list[str] = []
    for line in WORKSHEET.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)["query"]
        if q not in seen:
            seen.append(q)
    return seen


def _req(q: str, style: float, width: int) -> ParagraphRequest:
    return ParagraphRequest(
        user_query=q, style_deg=style, nonce=0, beam_width=width,
        topic_lock=True, retrieve_consequence=True, retrieval_gates=True,
        specific_pref=True, use_topology=True)


def _run(vault, topology, q, style, width):
    t0 = time.perf_counter()
    out = weave_paragraph(vault, _req(q, style, width), topology=topology)
    ms = (time.perf_counter() - t0) * 1000.0
    lines = [s.get("line") or "" for s in (out.get("sentences") or [])]
    blank = (not lines) or all(x == BLANK_GLYPH for x in lines)
    return {"lines": lines, "blank": blank, "ms": ms}


def _jaccard(a, b) -> float:
    sa, sb = set(x for x in a if x), set(x for x in b if x)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def main() -> None:
    vault = Vault(Path(os.environ["QBOT_FEYNMAN_VAULT"]))
    topology = get_engine(force=True)
    qs = _questions()

    rows = []
    per = {}
    for style in STYLES:
        for w in WIDTHS:
            per[(style, w)] = {"blank": 0, "ms": [], "lines": set(),
                               "tops": [], "sent_distinct": 0, "n": 0}
    changes = {(style, w): 0 for style in STYLES for w in WIDTHS}

    for q in qs:
        by_style = {}
        for style in STYLES:
            by_style[style] = {}
            for w in WIDTHS:
                r = _run(vault, topology, q, style, w)
                by_style[style][w] = r
                st = per[(style, w)]
                st["n"] += 1
                st["blank"] += r["blank"]
                st["ms"].append(r["ms"])
                st["lines"].update(x for x in r["lines"] if x)
                st["tops"].append(r["lines"][0] if r["lines"] else "")
                lines = [x for x in r["lines"] if x]
                st["sent_distinct"] += len(set(lines))
            base = by_style[style][BASELINE]["lines"][:1]
            for w in WIDTHS:
                if by_style[style][w]["lines"][:1] != base:
                    changes[(style, w)] += 1
        rows.append({str(style): {str(w): by_style[style][w]["lines"]
                                  for w in WIDTHS} for style in STYLES})

    summary = {}
    for style in STYLES:
        table = {}
        for w in WIDTHS:
            st = per[(style, w)]
            n = st["n"]
            table[str(w)] = {
                "blank_rate": round(st["blank"] / n, 3),
                "distinct_lines": len(st["lines"]),
                "distinct_top_lines": len(set(st["tops"])),
                "mean_distinct_sentences": round(st["sent_distinct"] / n, 2),
                "ms_mean": round(sum(st["ms"]) / n, 1),
                "top_line_change_vs_6": round(changes[(style, w)] / n, 3),
            }
        summary[str(style)] = table

    # 相邻档位重合度：收益是否递减（用每题的句子集合）
    adjacent = {}
    for style in STYLES:
        for a, b in zip(WIDTHS, WIDTHS[1:]):
            key = f"{style:g}°:{a}→{b}"
            adjacent[key] = round(
                sum(_jaccard(r[str(style)][str(a)], r[str(style)][str(b)])
                    for r in rows) / len(rows), 3)

    out = {"vault": os.environ["QBOT_FEYNMAN_VAULT"],
           "questions": len(qs), "widths": list(WIDTHS),
           "baseline_width": BASELINE, "by_style": summary,
           "adjacent_jaccard": adjacent, "rows": rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("questions", "widths", "by_style", "adjacent_jaccard")},
                     ensure_ascii=False, indent=2))
    print(f"✅ 明细 → {OUT}")


if __name__ == "__main__":
    main()
