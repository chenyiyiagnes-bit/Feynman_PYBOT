"""P14：检索时 specific（tier=2）从句优先于通用原子。

实测问题：动量的 wired 后果 top-3 全是"动量变化/动量改变/动量保持不变"
这类通用原子，语料里挖到的"动量却继续增加"排在第 4 以后，永远用不上。
检索的目的是给实体配专属从句，因此 tier=2 应在排序里先于 tier=1。
默认关；关时逐字节不变。
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
    run_feynman_print,
)


def _g(word, slot, phase, tier=1, count=10):
    return {"word": word, "slot": slot, "count": count, "amp": 1.0,
            "domain": "physics", "style": "lecture", "phase_deg": phase,
            "sources": ["样｜1"], "example": "", "tier": tier}


FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [_g("你看", "hook", 60.0)],
    "entities": [_g("动量", "entity", 50.0)],
    "actions": [_g("观察", "action", 55.0)],
    "consequences": [
        _g("动量改变", "consequence", 58.0, tier=1, count=40),
        _g("动量变化", "consequence", 59.0, tier=1, count=30),
        _g("动量保持不变", "consequence", 57.0, tier=1, count=20),
        _g("动量却继续增加", "consequence", 56.0, tier=2, count=4),
    ],
    "entangled_pairs": [],
    "cooccurrence": {
        "动量|动量改变": 9, "动量|动量变化": 7, "动量|动量保持不变": 5,
        "动量|动量却继续增加": 2,
        "观察|动量改变": 9, "观察|动量变化": 7, "观察|动量保持不变": 5,
        "观察|动量却继续增加": 2,
    },
}
FAKE_PATH = TMP / "vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def _pool():
    return VAULT.by_slot["consequence"]


def test_default_off_keeps_count_order():
    top = _retrieve_consequences(VAULT, "动量", _pool(), 70.0)
    assert [g["word"] for g in top] == ["动量改变", "动量变化", "动量保持不变"]


def test_specific_pref_lifts_tier2_clause():
    top = _retrieve_consequences(VAULT, "动量", _pool(), 70.0, specific_pref=True)
    assert top[0]["word"] == "动量却继续增加"
    assert len(top) == 3


def test_specific_pref_does_not_invent_edges():
    # 时空没有 E→C 边：退回原池，排序与关时一致，不把专属从句凭空提到最前
    a = [g["word"] for g in _retrieve_consequences(VAULT, "时空", _pool(), 70.0)]
    b = [g["word"] for g in _retrieve_consequences(VAULT, "时空", _pool(), 70.0,
                                                   specific_pref=True)]
    assert a == b


def test_flag_off_is_byte_identical():
    a = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3, top_k=20))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=55, nonce=3, top_k=20,
                                              specific_pref=False))
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a.get("specific_pref") in (False, None)


def test_flag_on_changes_retrieval_result():
    kw = dict(style_deg=55, nonce=3, top_k=20, retrieve_consequence=True,
              specific_pref=True)
    a = run_feynman_print(VAULT, PrintRequest(**kw))
    cons = [p["glyphs"]["consequence"] for p in a["paths"]]
    assert "动量却继续增加" in cons


def test_deterministic():
    kw = dict(style_deg=55, nonce=9, top_k=20, retrieve_consequence=True,
              specific_pref=True)
    a = run_feynman_print(VAULT, PrintRequest(**kw))
    b = run_feynman_print(VAULT, PrintRequest(**kw))
    assert [p["line"] for p in a["paths"]] == [p["line"] for p in b["paths"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P14 specific 优先 {len(tests)} 项回归全部通过")
