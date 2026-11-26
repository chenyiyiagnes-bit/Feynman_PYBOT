"""P4/P5 语料层：为每个后果从句收集左上下文证据。

扫描 p8 语料，对每个后果字粒（377 条）找到句子里的每次出现，截取左侧
同一分句内最多 ``--window`` 个字符，记录窗口里出现了哪些库内实体/动作。
产物用于解码期判断一个后果"是不是通用原子"：左上下文里什么都接的
（高多样性、低集中度）是空话；只跟某一实体/动作同处的才有信息量。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_left_context.py \
        --corpus data/p8_corpus_v2.jsonl \
        --vault data/feynman_style_vault_p9.json \
        --out data/left_context_p9.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

STOP = re.compile(r"[，。；：？！、\n]")

# 同一后果在一次句子里出现多次时，只记一次左上下文，避免长句刷分
def _left_window(text: str, pos: int, window: int) -> str:
    start = max(0, pos - window)
    frag = text[start:pos]
    # 截到最近的标点之后，保证只看同一分句
    m = None
    for m in STOP.finditer(frag):
        pass
    if m is not None:
        frag = frag[m.end():]
    return frag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--vault", default="data/feynman_style_vault_p9.json")
    ap.add_argument("--out", default="data/left_context_p9.json")
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()

    vault = json.loads(Path(args.vault).read_text(encoding="utf-8"))
    entities = [g["word"] for g in vault["entities"]]
    actions = [g["word"] for g in vault["actions"]]
    consequences = [g["word"] for g in vault["consequences"]]
    tier = {g["word"]: g.get("tier", 1) for g in vault["consequences"]}

    stats = {c: {"occ": 0, "left_entity": Counter(), "left_action": Counter()}
             for c in consequences}

    sentences = 0
    with Path(args.corpus).open(encoding="utf-8") as fh:
        for line in fh:
            sentences += 1
            text = json.loads(line).get("text", "")
            if not text:
                continue
            for c in consequences:
                pos = text.find(c)
                if pos < 0:
                    continue
                seen_frag = set()
                while pos >= 0:
                    frag = _left_window(text, pos, args.window)
                    if frag not in seen_frag:
                        seen_frag.add(frag)
                        st = stats[c]
                        st["occ"] += 1
                        for e in entities:
                            if e in frag:
                                st["left_entity"][e] += 1
                        for a in actions:
                            if a in frag:
                                st["left_action"][a] += 1
                    pos = text.find(c, pos + len(c))

    out = {"corpus": str(args.corpus), "sentences": sentences,
           "window": args.window, "consequences": {}}
    for c in consequences:
        st = stats[c]
        occ = st["occ"]
        le = dict(st["left_entity"])
        la = dict(st["left_action"])
        out["consequences"][c] = {
            "tier": tier[c],
            "occ": occ,
            "left_entity": dict(sorted(le.items(), key=lambda kv: -kv[1])[:8]),
            "left_action": dict(sorted(la.items(), key=lambda kv: -kv[1])[:8]),
            "entity_kinds": len(le),
            "action_kinds": len(la),
            "top_entity_share": round(max(le.values()) / occ, 4) if occ and le else 0.0,
            "entropy_entity": _entropy(le, occ),
        }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"句子 {sentences}；后果 {len(consequences)}；产物 {args.out}")


def _entropy(counter: dict, occ: int) -> float:
    """归一化左实体熵：1 = 谁都能接（通用），0 = 只跟一个实体同处。"""
    import math
    if not counter or occ <= 0:
        return 1.0
    total = sum(counter.values())
    h = -sum((v / total) * math.log(v / total) for v in counter.values())
    kinds = len(counter)
    if kinds <= 1:
        return 0.0
    return round(h / math.log(kinds), 4)


if __name__ == "__main__":
    main()
