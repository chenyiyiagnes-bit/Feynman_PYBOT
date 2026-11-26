"""P15 语料外书籍补边：build_vault(edge_supplement=...) 只加不减。

背景：P15 的空档实体从句大多来自桌面物理书，但这些句子不在 p8 语料里，
_build_cooccurrence 看不见它们，E→C 边会是 0，检索永远取不到。
补边文件只补"语料外观察到的同句共现"，且必须落在已入库的字粒上。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_p15_edge_supplement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.feynman_vault import build_vault

TMP = Path("/tmp/feynman_test_p15_edge_supplement")


def _corpus() -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    corpus = TMP / "corpus.jsonl"
    rows = []
    for src in ("费恩曼物理学讲义.md｜1-1", "费恩曼物理学讲义.md｜1-2",
                "费恩曼物理学讲义.md｜1-3"):
        for cons in ("温度升高", "亮度增加"):
            # 实体放名词短语（电子的偏转），后果放倒数第二个分句
            rows.append({"source": src, "section": src.split("｜")[1],
                         "text": f"电子的偏转告诉我们，我们前面已经讨论过这个现象并且记住它的基本规律，{cons}，这一点很重要。"})
    corpus.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                      encoding="utf-8")
    return corpus


def _inventory() -> Path:
    inv = TMP / "inventory.json"
    items = [
        {"word": "温度升高", "slot": "consequence", "count": 8, "amp": 1.0,
         "domain": "physics", "style": "lecture", "phase_deg": 20.0,
         "sources": ["费恩曼物理学讲义.md｜1-1"], "example": "温度升高", "tier": 1},
        {"word": "磁场强度会缓慢下降", "slot": "consequence", "count": 3, "amp": 0.95,
         "domain": "physics", "style": "lecture", "phase_deg": 40.0,
         "sources": ["书｜电磁学"], "example": "磁场强度会缓慢下降", "tier": 2},
    ]
    inv.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return inv


def test_supplement_adds_only_listed_pair():
    corpus, inv = _corpus(), _inventory()
    base = build_vault(corpus, p8=True, consequence_inventory=inv)
    base_cooc = dict(base["cooccurrence"])
    assert "电子|磁场强度会缓慢下降" not in base_cooc

    sup = TMP / "edges.json"
    sup.write_text(json.dumps({"电子|磁场强度会缓慢下降": 3}, ensure_ascii=False),
                   encoding="utf-8")
    supd = build_vault(corpus, p8=True, consequence_inventory=inv,
                       edge_supplement=sup)
    cooc = supd["cooccurrence"]

    assert cooc["电子|磁场强度会缓慢下降"] == 3
    for k, v in base_cooc.items():
        assert cooc[k] == v, f"既有边被改动：{k} {v}→{cooc[k]}"
    extra = {k for k in cooc if k not in base_cooc}
    assert extra == {"电子|磁场强度会缓慢下降"}, extra

    # 未涉及的其他槽逐项不变
    assert [g["word"] for g in supd["entities"]] == [g["word"] for g in base["entities"]]
    assert [g["word"] for g in supd["actions"]] == [g["word"] for g in base["actions"]]
    assert [g["word"] for g in supd["consequences"]] == [g["word"] for g in base["consequences"]]


def test_supplement_existing_pair_is_incremental():
    corpus, inv = _corpus(), _inventory()
    base = build_vault(corpus, p8=True, consequence_inventory=inv)
    existing = "电子|温度升高"
    assert base["cooccurrence"][existing] > 0
    sup = TMP / "edges_existing.json"
    sup.write_text(json.dumps({existing: 5}, ensure_ascii=False), encoding="utf-8")
    supd = build_vault(corpus, p8=True, consequence_inventory=inv,
                       edge_supplement=sup)
    assert supd["cooccurrence"][existing] == base["cooccurrence"][existing] + 5
    for k, v in base["cooccurrence"].items():
        if k != existing:
            assert supd["cooccurrence"][k] == v


def test_supplement_rejects_unknown_words():
    corpus, inv = _corpus(), _inventory()
    bad = TMP / "edges_bad.json"
    bad.write_text(json.dumps({"不存在的实体|温度升高": 2}, ensure_ascii=False),
                   encoding="utf-8")
    try:
        build_vault(corpus, p8=True, consequence_inventory=inv, edge_supplement=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("未知实体应抛 ValueError")


def test_supplement_rejects_non_consequence_target():
    corpus, inv = _corpus(), _inventory()
    bad = TMP / "edges_bad2.json"
    # 电子 是实体，不是后果槽字粒
    bad.write_text(json.dumps({"温度升高|电子": 2}, ensure_ascii=False),
                   encoding="utf-8")
    try:
        build_vault(corpus, p8=True, consequence_inventory=inv, edge_supplement=bad)
    except ValueError:
        pass
    else:
        raise AssertionError("指向非后果字粒应抛 ValueError")


def test_supplement_rejects_bad_counts_and_keys():
    corpus, inv = _corpus(), _inventory()
    for payload in ({"电子|温度升高": 0}, {"电子|温度升高": -2},
                    {"电子|温度升高": "3"}, {"电子温度升高": 1}):
        p = TMP / "edges_bad3.json"
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        try:
            build_vault(corpus, p8=True, consequence_inventory=inv,
                        edge_supplement=p)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法补边应抛 ValueError：{payload}")


def test_no_supplement_is_byte_identical():
    corpus, inv = _corpus(), _inventory()
    a = build_vault(corpus, p8=True, consequence_inventory=inv)
    b = build_vault(corpus, p8=True, consequence_inventory=inv,
                    edge_supplement=None)
    assert json.dumps(a, ensure_ascii=False, sort_keys=True) == \
           json.dumps(b, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"✅ P15 补边 {len(fns)} 项回归全部通过")
