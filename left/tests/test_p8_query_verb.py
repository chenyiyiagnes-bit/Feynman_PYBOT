"""P8 第二轮修复回归：

1. 问句动词注入动作槽（"发光/吸/浮"直接当动作字，不再退回"产生/消失"）；
2. 问句实体在语料里没见过时，动作要落在有 A→C 出边的字上（不给相位凑数的空动作）；
3. 时间残片不当后果（"事故发生前不久"）；
4. 纯招呼语返回模板招呼，不拿语料排版冒充回答。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_vault import is_complete_consequence  # noqa: E402
from plugins.feynman_typesetter import (  # noqa: E402
    PrintRequest,
    Vault,
    detect_smalltalk,
    extract_query_verbs,
    run_feynman_print,
)

FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{verb}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [
        {"word": "你看", "slot": "hook", "count": 20, "amp": 1.0, "domain": "everyday",
         "style": "speech", "phase_deg": 60.0, "sources": ["样｜1"], "example": ""},
    ],
    "entities": [
        {"word": "电子", "slot": "entity", "count": 50, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 20.0, "sources": ["样｜1"], "example": ""},
        {"word": "温度", "slot": "entity", "count": 20, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 30.0, "sources": ["样｜3"], "example": ""},
    ],
    "actions": [
        # 观察有 A→C 出边；旋转没有，但相位最近——用来验证"未知实体不拿空动作凑数"
        {"word": "观察", "slot": "action", "count": 22, "amp": 1.0, "domain": "everyday",
         "style": "lecture", "phase_deg": 90.0, "sources": ["样｜1"], "example": "",
         "order": "pre", "valency": "vo"},
        {"word": "出现", "slot": "action", "count": 18, "amp": 1.0, "domain": "everyday",
         "style": "dialogue", "phase_deg": 30.0, "sources": ["样｜2"], "example": "",
         "order": "post", "valency": "sv"},
        {"word": "旋转", "slot": "action", "count": 16, "amp": 1.0, "domain": "everyday",
         "style": "speech", "phase_deg": 68.0, "sources": ["样｜3"], "example": "",
         "order": "post", "valency": "sv"},
    ],
    "consequences": [
        {"word": "温度升高", "slot": "consequence", "count": 12, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 40.0, "sources": ["样｜1"], "example": ""},
        {"word": "位移变大", "slot": "consequence", "count": 5, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 35.0, "sources": ["样｜4"], "example": ""},
    ],
    "entangled_pairs": [],
    "cooccurrence": {
        "电子|观察": 6, "电子|旋转": 9, "观察|温度升高": 5, "观察|位移变大": 3,
    },
}
FAKE_PATH = TMP / "feynman_style_vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def test_extract_query_verbs():
    assert "发光" in extract_query_verbs("电子为什么会发光")
    assert "吸" in extract_query_verbs("磁铁为什么会吸铁")
    assert "浮" in extract_query_verbs("冰为什么会浮在水面上")
    assert extract_query_verbs("你好") == []
    # 系动词/情态词不是过程动词，不能当动作字印出来
    assert "是" not in extract_query_verbs("天空为什么是蓝的")
    assert "会" not in extract_query_verbs("电子为什么会发光")


def test_query_verb_is_injected_into_action_slot():
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=7, user_query="电子为什么会发光", conditioned_chain=True))
    assert out["user_verbs"] == ["发光"]
    assert any(p["glyphs"]["action"] == "发光" for p in out["paths"])
    assert "发光" in out["printed_collapse"]


def test_entity_verb_overlap_keeps_the_other_verb():
    """“摩擦”既被抽成实体又被抽成动词（vn）：另一个谓语在场时，“摩擦”归实体。"""
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=11, user_query="摩擦为什么会生热", conditioned_chain=True))
    assert out["user_entities"] == ["摩擦"]
    assert out["user_verbs"] == ["生热"]
    assert any(p["glyphs"]["action"] == "生热" for p in out["paths"])
    # 反向：只有一个谓语且它同时被当成实体（“发光”）→ 仍然注入动作槽
    out2 = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=12, user_query="电子为什么会发光", conditioned_chain=True))
    assert out2["user_entities"] == ["电子"]
    assert out2["user_verbs"] == ["发光"]


def test_unknown_entity_actions_prefer_wired():
    """冰在字库里没有 E→A 证据：动作不能拿相位最近但无出边的"旋转"顶上来。"""
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=8, user_query="冰", conditioned_chain=True))
    ice = [p for p in out["paths"] if p["glyphs"]["entity"] == "冰"]
    assert ice
    top = max(ice, key=lambda p: (p["probability"], p["line"]))
    assert top["glyphs"]["action"] == "观察"
    assert VAULT.cooc_count(top["glyphs"]["action"], top["glyphs"]["consequence"]) > 0


def test_temporal_fragment_is_not_consequence():
    assert not is_complete_consequence("事故发生前不久")
    assert not is_complete_consequence("事故发生三分钟后")
    assert not is_complete_consequence("爆炸之前")
    assert not is_complete_consequence("密度已经变成")
    assert is_complete_consequence("温度升高")
    assert is_complete_consequence("宇宙膨胀")


def test_greeting_returns_template_not_corpus():
    talk = detect_smalltalk("你好！")
    assert talk and "你好" in talk
    assert detect_smalltalk("电子为什么会发光") is None

    out = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=9,
                                                 user_query="你好", conditioned_chain=True))
    assert out["greeting"] is True
    assert out["printed_collapse"] == talk
    assert out["paths"] == []
    assert out["paths_total"] == 0


def test_greeting_paragraph_has_frontend_keys():
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
    out = weave_paragraph(VAULT, ParagraphRequest(user_query="你好。", style_deg=60, nonce=1))
    assert out["greeting"] is True
    assert out["paragraph"] == detect_smalltalk("你好。")
    assert len(out["sentences"]) == 1
    # 前端要用的键不能缺
    for key in ("pool_size", "seed", "latency_ms", "style_band", "sentences", "complete"):
        assert key in out


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P8 问句动词/门禁/招呼语 {len(tests)} 项回归全部通过")
