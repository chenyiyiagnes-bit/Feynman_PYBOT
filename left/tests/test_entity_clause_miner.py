"""实体专属后果挖掘器：从句级门禁的回归测试。

目标不是"实体出现过"，而是"实体在主语位置、后接谓词"。共现池里的
行星→"电子在外面绕着原子核旋转"这类噪声必须被挡掉。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.entity_clause_miner import aggregate, mine_clauses  # noqa: E402

ENTS = {"电子", "光子", "行星", "引力", "力", "质量", "时空", "学生"}


def _clauses(text, entities=ENTS):
    return [c["clause"] for c in mine_clauses(text, entities)]


def test_subject_position_kept():
    got = _clauses("电子被加速了")
    assert got == ["电子被加速了"]


def test_object_position_rejected():
    # 电子是宾语，不是主语
    assert "电子被加速了" not in _clauses("我们观察电子被加速了")


def test_first_entity_wins_and_other_entity_not_mined():
    got = mine_clauses("电子在外面绕着原子核旋转", ENTS)
    assert [c["entity"] for c in got] == ["电子"]
    assert "行星" not in [c["entity"] for c in got]


def test_quantifier_prefix_allowed():
    assert _clauses("一个电子发射一个光子") == ["一个电子发射一个光子"]


def test_connector_prefix_stripped():
    c = mine_clauses("所以电子被加速", ENTS)[0]
    assert c["entity"] == "电子"
    assert c["clause"] == "电子被加速"


def test_possessive_rejected():
    assert _clauses("电子的密度增大") == []


def test_no_predicate_rejected():
    assert _clauses("电子和光子") == []
    assert _clauses("电子") == []


def test_bad_chars_and_pronouns_rejected():
    assert _clauses("电子$a$被加速") == []
    assert _clauses("电子让我们感到奇怪") == []


def test_length_bounds():
    assert _clauses("电子动") == []
    long = "电子" + "在" * 30 + "加速"
    assert _clauses(long) == []


def test_clause_split_and_punctuation():
    got = _clauses("光子被吸收，电子发射一个光子。质量增加")
    assert "光子被吸收" in got
    assert "电子发射一个光子" in got
    assert "质量增加" in got


def test_deterministic():
    t = "所以电子被加速，光子被吸收。"
    assert mine_clauses(t, ENTS) == mine_clauses(t, ENTS)


def test_aggregate_counts_sections_and_provenance():
    sents = [
        {"text": "电子被加速。", "section": "S1"},
        {"text": "所以电子被加速。", "section": "S1"},
        {"text": "电子被加速，然后发光。", "section": "S2"},
        {"text": "光子被吸收。", "section": "S3"},
    ]
    rows = aggregate(sents, ENTS, min_count=2, min_sections=2)
    top = [r for r in rows if r["entity"] == "电子"]
    assert len(top) == 1
    r = top[0]
    assert r["clause"] == "电子被加速"
    assert r["count"] == 3
    assert set(r["sections"]) == {"S1", "S2"}
    assert r["example"] == "电子被加速。"  # 首次出现，按语料顺序，逐字节可查


def test_aggregate_thresholds_drop_thin_candidates():
    sents = [{"text": "电子被加速。", "section": "S1"}]
    assert aggregate(sents, ENTS, min_count=2, min_sections=1) == []




# ---- 二次质量门禁：候选聚合后的残句/元指代过滤 ----

from plugins.entity_clause_miner import filter_candidates  # noqa: E402


def _row(entity, clause, count=3):
    return {"entity": entity, "clause": clause, "count": count,
            "sections": ["S1", "S2"], "example": "x" + clause + "y"}


def test_filter_rejects_reporting_fragments():
    rows = [_row("量子力学", "量子力学表明"), _row("科学家", "科学家们发现")]
    assert filter_candidates(rows) == []


def test_filter_rejects_adverbial_fragments():
    rows = [_row("光子", "一个光子射到玻璃上时"), _row("光子", "光子在时间中倒行时")]
    assert filter_candidates(rows) == []


def test_filter_rejects_unbalanced_brackets():
    rows = [_row("加速度", "加速度要由弹簧的性质来决定）"),
            _row("光子", "光子经前表面透射（无旋转")]
    assert filter_candidates(rows) == []


def test_filter_rejects_discourse_deixis():
    rows = [_row("分子", "那些分子都要像上面所说的那样分离开来")]
    assert filter_candidates(rows) == []


def test_filter_keeps_good_clauses():
    rows = [_row("电子", "电子被加速"), _row("角动量", "角动量是守恒的"),
            _row("温度", "温度足够高")]
    assert len(filter_candidates(rows)) == 3


def test_filter_excludes_narrative_entities():
    rows = [_row("父亲", "父亲带着费曼到树林中散步"),
            _row("学生", "许多学生害怕进入教室")]
    assert filter_candidates(rows) == []


def test_filter_drops_substring_duplicates():
    rows = [_row("光子", "光子被吸收"), _row("光子", "一个光子被吸收", count=5)]
    kept = filter_candidates(rows)
    assert [r["clause"] for r in kept] == ["一个光子被吸收"]


def test_filter_is_deterministic():
    rows = [_row("电子", "电子被加速"), _row("量子力学", "量子力学表明")]
    assert filter_candidates(rows) == filter_candidates(rows)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 实体从句挖掘器 {len(tests)} 项回归全部通过")
