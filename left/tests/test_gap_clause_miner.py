"""P15 空档实体·跨书从句挖掘器。

与 entity_clause_miner 的区别：不要求实体作主语。中文物理书里
坐标系/直线/常数/函数这些词大多作状语或宾语（沿直线、在平面内、
测得常数），主语式挖掘对它们近乎零产出。这里改为"从句包含实体"，
仍过同一套确定性门禁 + P8 后果门。

用法（测试）：
    PYTHONPATH=. .venv/bin/python -B tests/test_gap_clause_miner.py
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

from plugins.gap_clause_miner import (  # noqa: E402
    _frame_invariance_ok,
    aggregate_books,
    iter_book_sentences,
    mine_entity_clauses,
)


def _clauses(text, ents):
    return [r["clause"] for r in mine_entity_clauses(text, set(ents))]


def test_mines_entity_as_object():
    # 实体不作主语也要能挖到（沿直线/在平面内）
    got = _clauses("平面上的直线也随之弯曲了。", ["直线"])
    assert got == ["平面上的直线也随之弯曲了"]


def test_mines_entity_as_subject():
    got = _clauses("线圈转动，穿过它的磁通量便将发生改变。", ["线圈"])
    assert got == ["线圈转动"]


def test_one_candidate_per_clause_longest_entity():
    # 同分句两个实体：取更长的那个，且至多一条
    got = _clauses("量子力学效应往往变得微不足道。", ["力学", "量子力学"])
    assert len(got) <= 1
    assert got and got[0].startswith("量子力学")


def test_rejects_without_effect_verb():
    # P8 后果门：没有变化类动词的纯定义/指称句不要
    assert _clauses("量子力学是一门学科。", ["量子力学"]) == []


def test_rejects_figure_reference():
    assert _clauses("如图9-8所示，曲线相当迅速地下降至零。", ["曲线"]) == []
    assert _clauses("图 3 中的直线斜率不同。", ["直线"]) == []


def test_rejects_deixis():
    assert _clauses("上述曲线变得平缓。", ["曲线"]) == []
    assert _clauses("前一章讨论的线圈发生转动。", ["线圈"]) == []


def test_rejects_report_endings():
    assert _clauses("线圈的变化表明。", ["线圈"]) == []


def test_rejects_question_and_pronoun():
    assert _clauses("曲线的斜率为什么发生突变？", ["曲线"]) == []
    assert _clauses("它的颜色发生了变化。", ["颜色"]) == []


def test_length_window():
    assert _clauses("长度变。", ["长度"]) == []            # 太短
    assert _clauses("这一条特别特别长的直线在某种极不寻常的条件下确实会慢慢地弯曲起来。",
                    ["直线"]) == []                        # 超过 24 字


def test_iter_book_sentences_filters_non_chinese():
    p = TMP / "book.md"
    p.write_text("直线在引力场中发生了弯曲，光也跟着偏转。\n\n" + "This is an English paragraph about straight lines. " * 5,
                 encoding="utf-8")
    sents = [s for s in iter_book_sentences(p)]
    assert any("弯曲" in s for s in sents)
    assert not any("English" in s for s in sents)


def test_aggregate_counts_and_provenance():
    a = TMP / "a.md"
    b = TMP / "b.md"
    a.write_text("引力场里的空间弯曲了，平面上的直线也随之弯曲了。", encoding="utf-8")
    b.write_text("引力场里的空间弯曲了，平面上的直线也随之弯曲了。", encoding="utf-8")
    rows = aggregate_books([a, b], {"直线"}, min_books=1)
    hit = [r for r in rows if "直线" in r["clause"]]
    assert hit and hit[0]["count"] >= 2 and hit[0]["books"] >= 2
    assert hit[0]["clause"] in hit[0]["example"]


def test_aggregate_min_books():
    a = TMP / "c.md"
    a.write_text("线圈中就产生感应电动势。", encoding="utf-8")
    rows = aggregate_books([a], {"线圈"}, min_books=2)
    assert rows == []


def test_deterministic():
    a = TMP / "d.md"
    a.write_text("曲线的斜率发生突变。曲线的斜率发生突变。", encoding="utf-8")
    r1 = aggregate_books([a], {"曲线"}, min_books=1)
    r2 = aggregate_books([a], {"曲线"}, min_books=1)
    assert json.dumps(r1, ensure_ascii=False) == json.dumps(r2, ensure_ascii=False)


def test_invariance_gate_accepts_clean_statements():
    for c in ("物理规律与坐标系无关",
              "直线具有和坐标系无关的意义",
              "两个坐标系具有相同的原点",
              "所有坐标系必须等效"):
        assert _frame_invariance_ok(c), c


def test_invariance_gate_rejects_non_invariance():
    # 定义句、无不变性标记、代词主语都不收
    assert not _frame_invariance_ok("坐标系是一个数学工具")
    assert not _frame_invariance_ok("坐标系发生转动")
    assert not _frame_invariance_ok("它与坐标系无关")


def test_invariance_gate_rejects_long_and_conditional():
    assert not _frame_invariance_ok("空间中的两个点之间的绝对距离不依赖于坐标系的变化")
    assert not _frame_invariance_ok("若坐标系与惯性系重合则规律与坐标系无关")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 空档实体跨书挖掘器 {len(tests)} 项回归全部通过")
