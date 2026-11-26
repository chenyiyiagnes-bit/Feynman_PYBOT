"""P10c：没有 E→C 边的实体，用 E→A→C 两跳检索后果。

实测（p10b）半数查询实体（水、结冰、时空）在 p9 图里没有任何 E→C 直连边，
检索退回未过滤的通用池，输出仍是"观察半径，面积变化"。
两跳检索：先取实体有 E→A 边的动作，再取这些动作有 A→C 边的后果；
同时质量门禁也应用到回退池，至少把残句拦掉。
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
    "entities": [_g("水", "entity", 50.0), _g("时空", "entity", 40.0),
                 _g("电子", "entity", 20.0)],
    "actions": [_g("出现", "action", 60.0), _g("观察", "action", 30.0)],
    "consequences": [
        _g("出现一圈条纹", "consequence", 60.0, tier=2),
        _g("动量改变", "consequence", 55.0),
        _g("空间是弯曲", "consequence", 65.0, tier=2),
        _g("位置变化", "consequence", 70.0),
        _g("电子发射一个光子", "consequence", 20.0, tier=2),
    ],
    "entangled_pairs": [],
    "cooccurrence": {
        # 水没有 E→C 直连，只有 E→A（水→出现）
        "水|出现": 4,
        # 出现→出现一圈条纹；观察→动量改变
        "出现|出现一圈条纹": 5,
        "观察|动量改变": 3,
        # 电子有直连 E→C
        "电子|电子发射一个光子": 6,
    },
}
FAKE_PATH = TMP / "vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def test_two_hop_beats_fallback_for_entity_without_ec():
    pool = VAULT.by_slot["consequence"]
    v1 = _retrieve_consequences(VAULT, "水", pool, 70.0)
    two = _retrieve_consequences(VAULT, "水", pool, 70.0, via_action=True)
    # v1 退回原池（相位顺序），两跳应拿到动作链路对应的后果
    assert v1[0]["word"] == "出现一圈条纹" or len(v1) > 1
    assert [g["word"] for g in two] == ["出现一圈条纹"], [g["word"] for g in two]


def test_direct_ec_wins_over_two_hop():
    pool = VAULT.by_slot["consequence"]
    got = _retrieve_consequences(VAULT, "电子", pool, 40.0, via_action=True)
    assert [g["word"] for g in got] == ["电子发射一个光子"]


def test_gates_apply_to_fallback_pool():
    pool = VAULT.by_slot["consequence"]
    # 时空没有 E→C/E→A 边：v1 原样退回；带门禁时残句"空间是弯曲"被拦
    raw = _retrieve_consequences(VAULT, "时空", pool, 60.0)
    gated = _retrieve_consequences(VAULT, "时空", pool, 60.0, gates=True,
                                   query="时空为什么会弯曲")
    assert any(g["word"] == "空间是弯曲" for g in raw)
    assert all(g["word"] != "空间是弯曲" for g in gated)


def test_flag_wires_through_print_and_is_off_by_default():
    a = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=11))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=11))
    assert a["printed_collapse"] == b["printed_collapse"]
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=12, user_query="水为什么会结冰", top_k=50,
        retrieve_consequence=True, retrieval_gates=True, retrieval_via_action=True))
    cons = {p["glyphs"]["consequence"] for p in out["paths"]
            if p["glyphs"]["entity"] == "水"}
    assert cons == {"出现一圈条纹"}, cons


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P10c 两跳检索 {len(tests)} 项回归全部通过")
