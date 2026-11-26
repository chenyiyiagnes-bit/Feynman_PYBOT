"""生成 P9 后果槽库存：策展清单 + 语料实证溯源。

输入 data/p9_consequence_allowlist.json（generic / specific 两档），
在语料里逐句核对每个词是否原样出现，收集出现次数、来源节、首个例句，
按词数归一化 amp。任何清单词在语料里找不到、或 specific 词过不了
_p8_consequence_ok，都写进报告并以非零码退出——清单和语料对不上时
不允许悄悄产出库存。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_p9_inventory.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import plugins.feynman_vault as fv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allowlist", default="data/p9_consequence_allowlist.json")
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--out", default="data/p9_consequence_inventory.json")
    ap.add_argument("--report", default="data/p9_consequence_inventory_report.json")
    args = ap.parse_args()

    allow = json.loads(Path(args.allowlist).read_text(encoding="utf-8"))
    generic = list(dict.fromkeys(allow.get("generic", [])))
    specific = list(dict.fromkeys(allow.get("specific", [])))
    dup = [w for w in specific if w in set(generic)]
    tier = {w: 1 for w in generic}
    tier.update({w: 2 for w in specific if w not in tier})   # 同词两档时按通用档
    words = list(tier)

    pattern = re.compile("|".join(
        sorted((re.escape(w) for w in words), key=len, reverse=True)))
    # 交替式匹配是非重叠的：长词会吞掉嵌在里面的短词（"电子的密度增大"吞"密度增大"），
    # 所以每个命中再展开成它包含的清单词，按句计一次。
    sub_map = {w: [x for x in words if x != w and x in w] for w in words}

    counts: Counter = Counter()
    sources: dict[str, list[str]] = {}
    examples: dict[str, str] = {}
    for src, section, text in fv.iter_sections(Path(args.corpus)):
        tag = f"{src}｜{section[:24]}"
        for sent in fv.sentences(text):
            present: set[str] = set()
            for m in pattern.finditer(sent):
                w = m.group(0)
                present.add(w)
                present.update(sub_map.get(w, ()))
            for w in present:
                counts[w] += 1
                examples.setdefault(w, sent[:80])
                lst = sources.setdefault(w, [])
                if tag not in lst and len(lst) < 3:
                    lst.append(tag)

    missing = [w for w in words if counts[w] == 0]
    bad_specific = [w for w in words
                    if tier[w] == 2 and not fv._p8_consequence_ok(w)]

    max_count = max(counts.values()) if counts else 1
    items = []
    for w in words:
        if counts[w] == 0:
            continue
        sample = examples.get(w) or w
        style, phase = fv._style_of(sample)
        amp = round(0.94 + 0.06 * math.log1p(counts[w]) / math.log1p(max_count), 3)
        items.append({
            "word": w,
            "slot": "consequence",
            "count": counts[w],
            "amp": amp,
            "domain": "physics",
            "style": style,
            "phase_deg": phase,
            "sources": sources.get(w, []),
            "example": fv._clean_consequence(fv._trim_tail(sample))[:60],
            "tier": tier[w],
        })
    items.sort(key=lambda it: (-it["count"], it["word"]))

    report = {
        "allowlist_words": len(words),
        "inventory_words": len(items),
        "generic": sum(1 for it in items if it["tier"] == 1),
        "specific": sum(1 for it in items if it["tier"] == 2),
        "duplicates_generic_and_specific": dup,
        "missing_in_corpus": missing,
        "specific_gate_failed": bad_specific,
        "count_max": max_count,
    }
    Path(args.out).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if missing or bad_specific:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
