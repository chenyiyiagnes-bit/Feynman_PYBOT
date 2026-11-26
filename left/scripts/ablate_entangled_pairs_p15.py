"""纠缠对消融：同一字库/问题/语体，只换 entangled_pairs 表，看输出变化。

两臂：
- old：字库自带的 80 对（分句候选名词，实际 0 对能命中 (实体, 后果)）；
- new：data/entangled_pairs_p15.json（E→C 共现边重算，80 对全部可命中）。

只测可数事实：首句变化率、三句集合重合度、留白率、跨问题多样性、延迟。
若变化率接近 0，说明这条加成仍然没在做事；变化明显才值得进盲评。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ablate_entangled_pairs_p15.py
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
PAIRS = Path("data/entangled_pairs_p15.json")
OUT = Path("data/eval_entangled_pairs_p15.json")
STYLES = (15.0, 60.0, 105.0)


def _questions(limit: int = 0) -> list[str]:
    seen: list[str] = []
    for line in WORKSHEET.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)["query"]
        if q not in seen:
            seen.append(q)
    return seen[:limit] if limit else seen


def _req(q: str, style: float) -> ParagraphRequest:
    return ParagraphRequest(
        user_query=q, style_deg=style, nonce=0, beam_width=6,
        topic_lock=True, retrieve_consequence=True, retrieval_gates=True,
        specific_pref=True, use_topology=True)


def _run(vault, ext, topology, q, style):
    t0 = time.perf_counter()
    out = weave_paragraph(vault, _req(q, style), ext_hash=ext,
                          topology=topology)
    ms = (time.perf_counter() - t0) * 1000.0
    lines = [s.get("line") or "" for s in (out.get("sentences") or [])]
    blank = (not lines) or all(x == BLANK_GLYPH for x in lines)
    return {"lines": lines, "blank": blank, "ms": ms}


def _jaccard(a, b) -> float:
    sa, sb = set(x for x in a if x), set(x for x in b if x)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def main() -> None:
    vault = Vault(Path(os.environ["QBOT_FEYNMAN_VAULT"]))
    old_pairs = list(vault.entangled)
    new_pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    topology = get_engine(force=True)
    ext, _ = ("", "")
    try:
        from plugins.domain_expander import load_domain_extensions_with_hash
        _, ext = load_domain_extensions_with_hash()
    except Exception:
        pass

    qs = _questions()
    rows = []
    stats = {k: {"same_top": 0, "blank": 0, "jacs": [], "ms": [],
                 "lines": set(), "changed_any": 0}
             for k in ("old", "new")}
    by_style = {s: {"same_top": 0, "n": 0} for s in STYLES}
    for q in qs:
        for s in STYLES:
            vault.entangled = old_pairs
            a = _run(vault, ext, topology, q, s)
            vault.entangled = new_pairs
            b = _run(vault, ext, topology, q, s)
            same = a["lines"][:1] == b["lines"][:1]
            by_style[s]["same_top"] += same
            by_style[s]["n"] += 1
            for key, r in (("old", a), ("new", b)):
                st = stats[key]
                st["blank"] += r["blank"]
                st["ms"].append(r["ms"])
                st["lines"].update(x for x in r["lines"] if x)
            stats["old"]["jacs"].append(_jaccard(a["lines"], b["lines"]))
            if a["lines"] != b["lines"]:
                stats["new"]["changed_any"] += 1
            rows.append({"query": q, "style": s,
                         "old": a["lines"], "new": b["lines"],
                         "identical_top": same})
    vault.entangled = old_pairs

    n = len(rows)
    summary = {
        "vault": os.environ["QBOT_FEYNMAN_VAULT"],
        "questions": len(qs), "runs": n,
        "styles": list(STYLES),
        "identical_top_line": sum(1 for r in rows if r["identical_top"]),
        "identical_top_rate": round(
            sum(1 for r in rows if r["identical_top"]) / n, 3),
        "identical_top_rate_by_style": {
            str(s): round(by_style[s]["same_top"] / by_style[s]["n"], 3)
            for s in STYLES},
        "sentence_set_jaccard_mean": round(
            sum(stats["old"]["jacs"]) / n, 3),
        "changed_outputs": stats["new"]["changed_any"],
        "blank_rate_old": round(stats["old"]["blank"] / n, 3),
        "blank_rate_new": round(stats["new"]["blank"] / n, 3),
        "distinct_lines_old": len(stats["old"]["lines"]),
        "distinct_lines_new": len(stats["new"]["lines"]),
        "ms_old_mean": round(sum(stats["old"]["ms"]) / n, 1),
        "ms_new_mean": round(sum(stats["new"]["ms"]) / n, 1),
        "usable_pairs_old": sum(
            1 for p in old_pairs
            if p.get("everyday") in {g["word"] for g in vault.by_slot["entity"]}
            and p.get("physics") in {g["word"] for g in vault.by_slot["consequence"]}),
        "usable_pairs_new": len(new_pairs),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✅ 明细 → {OUT}")


if __name__ == "__main__":
    main()
