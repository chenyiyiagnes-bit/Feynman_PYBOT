"""P10b：检索池质量门禁（主语错配 / 方向矛盾 / 残句）。

P10 检索把后果从句绑到实体的语料证据上，但仍会捞到三类坏后果：
- 主语错配：实体"原子"命中同段里以"原子弹"为主语的"原子弹爆炸"（字符串包含 ≠ 同一主语）；
- 方向矛盾：问"温度升高以后"，检索到"温度降低"；
- 残句：库存里的"空间是弯曲"（缺"的"）。
三者都是确定性可判的，放在检索池过滤里，全部默认关。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_typesetter import (  # noqa: E402
    PrintRequest,
    Vault,
    _retrieve_consequences,
    _retrieval_candidate_ok,
    run_feynman_print,
)


def _g(word, slot, phase, tier=1):
    return {"word": word, "slot": slot, "count": 10, "amp": 1.0, "domain": "physics",
            "style": "lecture", "phase_deg": phase, "sources": ["样｜1"], "example": "",
            "tier": tier}


FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [_g("你看", "hook", 60.0)],
    "entities": [_g("原子", "entity", 20.0), _g("温度", "entity", 30.0),
                 _g("时空", "entity", 40.0)],
    "actions": [_g("观察", "action", 30.0)],
    "consequences": [
        _g("原子弹爆炸", "consequence", 20.0, tier=2),
        _g("能量减少", "consequence", 25.0),
        _g("温度降低", "consequence", 30.0),
        _g("能量保持不变", "consequence", 35.0),
        _g("空间是弯曲", "consequence", 40.0, tier=2),
        _g("位置变化", "consequence", 45.0),
    ],
    "entangled_pairs": [],
    "cooccurrence": {
        "原子|原子弹爆炸": 6, "原子|能量减少": 1,
        "温度|温度降低": 6, "温度|能量保持不变": 2,
        "时空|空间是弯曲": 6, "时空|位置变化": 1,
    },
}
FAKE_PATH = TMP / "vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)
POOL = VAULT.by_slot["consequence"]


def test_subject_mismatch_rejected():
    # 无门禁：支撑最高的"原子弹爆炸"入选
    raw = _retrieve_consequences(VAULT, "原子", POOL, 40.0)
    assert raw[0]["word"] == "原子弹爆炸", [g["word"] for g in raw]
    ok, reason = _retrieval_candidate_ok("原子弹爆炸", "原子", "原子里面有什么")
    assert not ok and reason == "subject_mismatch"
    gated = _retrieve_consequences(VAULT, "原子", POOL, 40.0, gates=True,
                                   query="原子里面有什么")
    assert [g["word"] for g in gated] == ["能量减少"]


def test_contradicting_direction_rejected():
    ok, reason = _retrieval_candidate_ok("温度降低", "温度", "温度升高以后会怎样")
    assert not ok and reason == "contradiction"
    gated = _retrieve_consequences(VAULT, "温度", POOL, 50.0, gates=True,
                                   query="温度升高以后会怎样")
    assert [g["word"] for g in gated] == ["能量保持不变"]


def test_malformed_clause_rejected():
    ok, reason = _retrieval_candidate_ok("空间是弯曲", "时空", "时空为什么会弯曲")
    assert not ok and reason == "malformed"
    gated = _retrieve_consequences(VAULT, "时空", POOL, 60.0, gates=True,
                                   query="时空为什么会弯曲")
    assert [g["word"] for g in gated] == ["位置变化"]


def test_good_candidates_survive_and_queries_without_antonyms_keep_pool():
    ok, _ = _retrieval_candidate_ok("能量减少", "原子", "原子里面有什么")
    assert ok
    ok, _ = _retrieval_candidate_ok("温度降低", "温度", "温度是什么")
    assert ok, "问句里没有相反方向时不该拦"


def test_all_gated_falls_back_not_empty():
    # 实体只有一个候选且被门禁拦下：宁可给回退池，也不能把后果槽清空
    FAKE2 = dict(FAKE, cooccurrence={"原子|原子弹爆炸": 6})
    p = TMP / "vault2.json"
    p.write_text(json.dumps(FAKE2, ensure_ascii=False), encoding="utf-8")
    v2 = Vault(p)
    got = _retrieve_consequences(v2, "原子", v2.by_slot["consequence"], 40.0,
                                 gates=True, query="原子里面有什么")
    assert got, "全被拦时退回第一个候选，不能空"


def test_flag_off_keeps_p10_behavior_and_gates_on_changes_output():
    base = dict(style_deg=60, nonce=7, user_query="原子里面有什么", top_k=50,
                retrieve_consequence=True)
    on = run_feynman_print(VAULT, PrintRequest(**base))["paths"]
    gated = run_feynman_print(VAULT, PrintRequest(**base, retrieval_gates=True))["paths"]
    cons_on = {p["glyphs"]["consequence"] for p in on if p["glyphs"]["entity"] == "原子"}
    cons_g = {p["glyphs"]["consequence"] for p in gated if p["glyphs"]["entity"] == "原子"}
    assert "原子弹爆炸" in cons_on
    assert "原子弹爆炸" not in cons_g and cons_g, (cons_on, cons_g)
    # 没有用户实体的普通请求：默认关时两条路径集合相同
    a = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=8))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=8))
    assert [p["line"] for p in a["paths"]] == [p["line"] for p in b["paths"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P10b 检索质量门禁 {len(tests)} 项回归全部通过")
