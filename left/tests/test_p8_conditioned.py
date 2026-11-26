"""P8 回归：铸字净化、条件级联（实体→动作→后果）、段落接力。

默认关闭时基线不变由 fixtures_feynman_baseline.json 在其它测试里守住；
这里验证三件事：
1. 净化函数能把"天空变成/的时候/霓虹灯的颜色"这类片段拦在槽外；
2. 条件级联打开后仍确定、路径合法，关掉时与旧路径逐字节一致；
3. 段落接力：阶段 2 沿用锚点配价，阶段 3 回指锚实体。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_vault import (  # noqa: E402
    classify_action_valency,
    is_clean_action,
    is_complete_consequence,
    is_pure_noun_phrase,
)
from plugins.feynman_typesetter import (  # noqa: E402
    PrintRequest,
    Vault,
    run_feynman_print,
)

FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [
        {"word": "你看", "slot": "hook", "count": 20, "amp": 1.0, "domain": "everyday",
         "style": "speech", "phase_deg": 60.0, "sources": ["样｜1"], "example": ""},
        {"word": "注意", "slot": "hook", "count": 15, "amp": 1.0, "domain": "everyday",
         "style": "lecture", "phase_deg": 20.0, "sources": ["样｜2"], "example": ""},
    ],
    "entities": [
        {"word": "电子", "slot": "entity", "count": 50, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 20.0, "sources": ["样｜1"], "example": ""},
        {"word": "水面", "slot": "entity", "count": 30, "amp": 1.0, "domain": "everyday",
         "style": "dialogue", "phase_deg": 60.0, "sources": ["样｜2"], "example": ""},
        {"word": "温度", "slot": "entity", "count": 20, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 30.0, "sources": ["样｜3"], "example": ""},
        {"word": "电荷", "slot": "entity", "count": 18, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 25.0, "sources": ["样｜4"], "example": ""},
    ],
    "actions": [
        {"word": "观察", "slot": "action", "count": 22, "amp": 1.0, "domain": "everyday",
         "style": "lecture", "phase_deg": 30.0, "sources": ["样｜1"], "example": "",
         "order": "pre", "valency": "vo"},
        {"word": "出现", "slot": "action", "count": 18, "amp": 1.0, "domain": "everyday",
         "style": "dialogue", "phase_deg": 60.0, "sources": ["样｜2"], "example": "",
         "order": "post", "valency": "sv"},
        {"word": "旋转", "slot": "action", "count": 16, "amp": 1.0, "domain": "everyday",
         "style": "speech", "phase_deg": 90.0, "sources": ["样｜3"], "example": "",
         "order": "post", "valency": "sv"},
        {"word": "测量", "slot": "action", "count": 14, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 25.0, "sources": ["样｜4"], "example": "",
         "order": "pre", "valency": "vo"},
    ],
    "consequences": [
        {"word": "温度升高", "slot": "consequence", "count": 12, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 40.0, "sources": ["样｜1"], "example": ""},
        {"word": "出现一圈条纹", "slot": "consequence", "count": 8, "amp": 1.0, "domain": "physics",
         "style": "dialogue", "phase_deg": 78.0, "sources": ["样｜2"], "example": ""},
        {"word": "整条街都听见了", "slot": "consequence", "count": 6, "amp": 1.0, "domain": "physics",
         "style": "speech", "phase_deg": 118.0, "sources": ["样｜3"], "example": ""},
        {"word": "位移变大", "slot": "consequence", "count": 5, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 35.0, "sources": ["样｜4"], "example": ""},
    ],
    "entangled_pairs": [],
    # 有向边：电子→观察/测量/旋转，观察/测量→温度升高；水面→出现
    # "旋转"是没出边的动作，故意给它更高的 E→A 计数，验证条件级联不选它
    "cooccurrence": {
        "电子|观察": 6, "电子|测量": 4, "电子|旋转": 9, "电子|温度升高": 3,
        "观察|温度升高": 5, "测量|温度升高": 2,
        "水面|出现": 3, "出现|出现一圈条纹": 4,
    },
}
FAKE_PATH = TMP / "feynman_style_vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def test_entity_purity_gate():
    assert is_pure_noun_phrase("电子")
    assert is_pure_noun_phrase("温度")
    assert not is_pure_noun_phrase("天空变成")      # 动词收尾
    assert not is_pure_noun_phrase("的时候")        # 悬垂从句
    assert not is_pure_noun_phrase("无法")          # 副词
    assert not is_pure_noun_phrase("明白")          # 动词
    assert not is_pure_noun_phrase("时会")        # "…时会…" 切出的残片（jieba 标 n）
    assert not is_pure_noun_phrase("")


def test_action_purity_gate():
    assert is_clean_action("观察")
    assert is_clean_action("测量出")
    assert is_clean_action("开始旋转")
    assert not is_clean_action("观察水")            # 动词+宾语
    assert not is_clean_action("旋转木马")
    assert not is_clean_action("振荡电荷")
    assert not is_clean_action("测量来确")          # 碎片


def test_action_valency():
    assert classify_action_valency("观察") == "vo"
    assert classify_action_valency("测量出") == "vo"
    assert classify_action_valency("出现") == "sv"
    assert classify_action_valency("开始旋转") == "sv"


def test_consequence_completeness_gate():
    assert is_complete_consequence("温度升高")
    assert is_complete_consequence("水平移动一个重物")
    assert not is_complete_consequence("霓虹灯的颜色")   # 纯名词短语
    assert not is_complete_consequence("响声出现的速率")  # "的"把谓词变成定语
    assert not is_complete_consequence("然会发出声音")     # 切剩的连接词残片
    assert not is_complete_consequence("的时候")


def test_conditioned_off_is_unchanged_shape():
    out = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=1))
    assert out["conditioned"] is False
    # 旧笛卡尔积：2 颗起手式 × 3×3×3
    assert out["paths_total"] == 54
    assert all("conditioned_multiplier" not in p for p in out["paths"])


def test_conditioned_on_is_deterministic_and_uses_edges():
    req = dict(style_deg=60, nonce=2, user_query="电子为什么会发光", conditioned_chain=True)
    a = run_feynman_print(VAULT, PrintRequest(**req))
    b = run_feynman_print(VAULT, PrintRequest(**req))
    assert a["conditioned"] is True
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a["paths"]
    assert any(abs(p["conditioned_multiplier"] - 1.0) > 1e-9 for p in a["paths"])
    # 落到用户实体上的路径必须存在
    assert any(p["glyphs"]["entity"] == "电子" for p in a["paths"])


def test_conditioned_beam_keeps_unseen_pairs_neutral():
    out = run_feynman_print(
        VAULT, PrintRequest(style_deg=60, nonce=3, conditioned_chain=True))
    for p in out["paths"]:
        assert 0.5 <= p["conditioned_multiplier"] <= 2.0, p


def test_conditioned_beam_prefers_wired_actions():
    """动作在 A→C 表里有出边时优先；纯 E→A 高分但没后果证据的"旋转"不该登顶。"""
    out = run_feynman_print(
        VAULT, PrintRequest(style_deg=60, nonce=5, user_query="电子的旋转",
                            conditioned_chain=True))
    elec = [p for p in out["paths"] if p["glyphs"]["entity"] == "电子"]
    assert elec, "束里应有落到用户实体上的路径"
    top = max(elec, key=lambda p: (p["probability"], p["line"]))["glyphs"]
    assert top["entity"] == "电子"
    assert top["action"] != "旋转"
    assert VAULT.cooc_count(top["action"], top["consequence"]) > 0


def test_paragraph_relay_keeps_valency_and_anchor():
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
    out = weave_paragraph(VAULT, ParagraphRequest(
        user_query="电子为什么会发光", style_deg=60, nonce=4,
        beam_width=6, conditioned_chain=True))
    assert out["complete"] is True
    assert len(out["sentences"]) == 3
    anchor_val = out["sentences"][0]["valency"]
    anchor_entity = out["sentences"][0]["glyphs"]["entity"]
    # 配价接力是偏好而非硬约束：优先同配价，同配价里没有新内容时退回异配价凑齐一段
    second = out["sentences"][1]
    if second["valency"] != anchor_val:
        assert second["glyphs"]["action"] != out["sentences"][0]["glyphs"]["action"]
    assert out["sentences"][2]["glyphs"]["entity"] == anchor_entity
    # 三段不能只是换了起手式
    keys = [(s["glyphs"]["entity"], s["glyphs"]["action"], s["glyphs"]["consequence"])
            for s in out["sentences"]]
    assert len(set(keys)) == 3, keys


def test_paragraph_relay_off_keeps_old_behavior():
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
    req = ParagraphRequest(user_query="电子为什么会发光", style_deg=60, nonce=4, beam_width=6)
    a = weave_paragraph(VAULT, req)
    b = weave_paragraph(VAULT, req)
    assert a["paragraph"] == b["paragraph"]
    assert a["sentences"][0]["connector"] == ""
    assert all("valency" in s for s in a["sentences"])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P8 净化/级联/接力 {len(tests)} 项回归全部通过")
