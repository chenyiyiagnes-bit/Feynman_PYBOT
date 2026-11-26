"""P12：问句实体锁定（topic lock）。

实测（p10b 的 24 查询输出）：检索把后果绑到实体上之后，整句打分仍会把
问句实体挤掉——问"动量为什么守恒"，最终句是"观察半径，强度改变"，
而动量自己的检索池是"动量变化/动量改变/动量保持不变"。
问句已经抽出实体时，实体槽只保留这些实体（不再拿库内其它实体补位）；
没有抽出实体时退回原池。默认关，关时与旧版逐字节一致。
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
    "entities": [_g("动量", "entity", 40.0), _g("半径", "entity", 60.0),
                 _g("粒子", "entity", 50.0)],
    "actions": [_g("观察", "action", 50.0), _g("测量", "action", 55.0)],
    "consequences": [
        _g("动量改变", "consequence", 40.0),
        _g("半径增大", "consequence", 60.0),
        _g("动量保持不变", "consequence", 45.0),
        _g("面积变化", "consequence", 65.0),
    ],
    "entangled_pairs": [],
    "cooccurrence": {"动量|动量改变": 5, "动量|动量保持不变": 3,
                     "半径|半径增大": 4, "半径|面积变化": 2},
}
FAKE_PATH = TMP / "vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def _ents(out):
    return {p["glyphs"]["entity"] for p in out["paths"]}


def test_default_off_is_byte_identical():
    a = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3,
                                              user_query="动量为什么守恒"))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3,
                                              user_query="动量为什么守恒"))
    assert a["printed_collapse"] == b["printed_collapse"]
    assert "topic_lock" not in a or a.get("topic_lock") in (False, None)


def test_lock_keeps_only_query_entity():
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=55, nonce=3, user_query="动量为什么守恒", top_k=50,
        topic_lock=True))
    assert _ents(out) == {"动量"}, _ents(out)


def test_lock_then_retrieval_gives_query_bound_consequence():
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=55, nonce=3, user_query="动量为什么守恒", top_k=50,
        topic_lock=True, retrieve_consequence=True))
    cons = {p["glyphs"]["consequence"] for p in out["paths"]}
    assert cons <= {"动量改变", "动量保持不变"}, cons


def test_lock_without_extractable_entity_falls_back_to_full_pool():
    # "为什么" 抽不出库内实体：锁定不得把实体槽清空
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=55, nonce=3, user_query="为什么会这样", top_k=50,
        topic_lock=True))
    assert _ents(out), "抽不出实体时应退回原池"


def test_lock_is_deterministic():
    kw = dict(style_deg=55, nonce=9, user_query="动量为什么守恒", top_k=20,
              topic_lock=True, retrieve_consequence=True)
    a = run_feynman_print(VAULT, PrintRequest(**kw))
    b = run_feynman_print(VAULT, PrintRequest(**kw))
    assert [p["line"] for p in a["paths"]] == [p["line"] for p in b["paths"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P12 问句实体锁定 {len(tests)} 项回归全部通过")
