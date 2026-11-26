"""P9 后果槽扩表回归：build_vault 的 slot_limits 只改指定槽配额，别的不动。

背景：p8b 词表的后果槽被 limit_per_slot=60 截断，实际总支撑只有 200 次
（每条中位数 3 次），组合空间九成以上落在语料流形外。扩表靠放宽配额，
不靠改门禁、不靠加语料。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.feynman_vault import build_vault

TMP = Path("/tmp/feynman_test_consequence_inventory")


def _corpus() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    corpus = TMP / "corpus.jsonl"
    rows = []
    for src in ("费恩曼物理学讲义.md｜1-1", "费恩曼物理学讲义.md｜1-2"):
        for cons in ("温度升高", "亮度增加", "压力降低", "折射率增大"):
            # 后果放倒数第二个分句：句末分句带"。"过不了 _is_cjk，与真实语料同构
            rows.append({"source": src, "section": src.split("｜")[1],
                         "text": f"你看，电子发生偏转，我们前面已经讨论过这个现象并且记住它的基本规律，{cons}，这一点很重要。"})
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                      encoding="utf-8")
    return corpus


def main() -> None:
    corpus = _corpus()
    full = build_vault(corpus, p8=True)
    assert len(full["consequences"]) >= 2, full["consequences"]

    capped = build_vault(corpus, p8=True, slot_limits={"consequence": 1})
    assert len(capped["consequences"]) == 1, capped["consequences"]
    assert capped["consequences"][0]["word"] == full["consequences"][0]["word"]

    # 未指定的槽仍按默认配额，且槽内容不变
    assert [g["word"] for g in capped["hooks"]] == [g["word"] for g in full["hooks"]]
    assert [g["word"] for g in capped["entities"]] == [g["word"] for g in full["entities"]]

    # 配额扩大时后果槽可以超过默认 60 条的上限逻辑：用小语料验证不截断
    bigger = build_vault(corpus, p8=True, slot_limits={"consequence": 50})
    assert len(bigger["consequences"]) == len(full["consequences"]) >= 4, bigger["consequences"]

    # 策展库存模式：整槽替换，条目字段原样保留，其余槽不受影响
    inv = TMP / "inventory.json"
    items = [
        {"word": "温度升高", "slot": "consequence", "count": 7, "amp": 1.0,
         "domain": "physics", "style": "lecture", "phase_deg": 20.0,
         "sources": ["费恩曼物理学讲义.md｜1-1｜1-1"], "example": "温度升高", "tier": 1},
        {"word": "光线变暗", "slot": "consequence", "count": 2, "amp": 0.95,
         "domain": "physics", "style": "lecture", "phase_deg": 21.0,
         "sources": ["费恩曼物理学讲义.md｜1-2｜1-2"], "example": "光线变暗", "tier": 2},
    ]
    inv.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    curated = build_vault(corpus, p8=True, consequence_inventory=inv)
    assert [g["word"] for g in curated["consequences"]] == ["温度升高", "光线变暗"], \
        curated["consequences"]
    assert curated["consequences"][0]["tier"] == 1
    assert curated["consequences"][1]["tier"] == 2
    assert curated["consequences"][0]["sources"] == ["费恩曼物理学讲义.md｜1-1｜1-1"]
    assert [g["word"] for g in curated["hooks"]] == [g["word"] for g in full["hooks"]]

    bad = TMP / "bad_inventory.json"
    bad.write_text(json.dumps([{"word": "x", "slot": "entity"}], ensure_ascii=False),
                   encoding="utf-8")
    try:
        build_vault(corpus, p8=True, consequence_inventory=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("非法库存条目应抛 ValueError")

    print("✅ P9 后果槽扩表 7 项回归通过")


if __name__ == "__main__":
    main()
