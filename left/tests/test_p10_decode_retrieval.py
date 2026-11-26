"""P10 回归：解码期检索约束——后果从句按实体的 E→C 语义边从语料里取，
不再按相位自由拼。默认关时与旧版逐字节一致；没有边的实体退回原池。

背景：P9 扩表把 1 分"荒谬句"压到 11%，但剩下的多是"通顺但空洞"。
瓶颈是槽间独立重配，不是后果词表大小。P10 让后果候选由实体在真实语料里的
共现证据决定，而不是八个槽各自按相位取最近的词。
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
    # E→C 语义边只挂在"电子"上：温度升高(3)、位移变大(2)。
    # "出现一圈条纹"相位最近（78 vs 目标 80）但和电子没有证据——检索模式下不能出现。
    # 水面/温度没有任何 E→C 边，用来验证退回原池。
    "cooccurrence": {
        "电子|观察": 6, "电子|旋转": 9,
        "电子|温度升高": 3, "电子|位移变大": 2,
        "观察|温度升高": 5,
        "水面|出现": 3, "出现|出现一圈条纹": 4,
    },
}
FAKE_PATH = TMP / "feynman_style_vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)

WIRED = {"温度升高", "位移变大"}


def _cons_by_entity(out, entity):
    return {p["glyphs"]["consequence"] for p in out["paths"]
            if p["glyphs"]["entity"] == entity}


def test_default_flag_off_is_byte_identical():
    a = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=1))
    b = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=1,
                                              retrieve_consequence=False))
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a["paths_total"] == b["paths_total"]
    assert a["retrieval"] is False


def test_retrieval_restricts_cartesian_paths_to_entity_edges():
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=2, user_query="电子为什么会发光", top_k=50,
        retrieve_consequence=True))
    assert out["retrieval"] is True
    elec = _cons_by_entity(out, "电子")
    assert elec == WIRED, elec
    # 对照：同一请求关掉检索时，相位最近的"出现一圈条纹"会拼到电子上
    off = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=2, user_query="电子为什么会发光", top_k=50))
    assert "出现一圈条纹" in _cons_by_entity(off, "电子")


def test_retrieval_falls_back_when_entity_has_no_edges():
    on = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=3, top_k=50, retrieve_consequence=True))
    off = run_feynman_print(VAULT, PrintRequest(style_deg=60, nonce=3, top_k=50))
    for ent in ("水面", "温度"):
        assert _cons_by_entity(on, ent) == _cons_by_entity(off, ent), ent
        assert len(_cons_by_entity(on, ent)) > 1, ent


def test_retrieval_applies_in_conditioned_beam():
    out = run_feynman_print(VAULT, PrintRequest(
        style_deg=60, nonce=4, user_query="电子为什么会发光",
        conditioned_chain=True, retrieve_consequence=True))
    assert out["conditioned"] is True
    elec = _cons_by_entity(out, "电子")
    assert elec, "束里应有用户实体路径"
    assert elec <= WIRED, elec
    assert "出现一圈条纹" not in elec


def test_retrieval_is_deterministic():
    req = dict(style_deg=60, nonce=5, user_query="电子为什么会发光",
               conditioned_chain=True, retrieve_consequence=True)
    a = run_feynman_print(VAULT, PrintRequest(**req))
    b = run_feynman_print(VAULT, PrintRequest(**req))
    assert a["printed_collapse"] == b["printed_collapse"]
    assert [p["line"] for p in a["paths"]] == [p["line"] for p in b["paths"]]


def test_retrieval_without_any_ec_edges_is_neutral():
    empty = dict(FAKE, cooccurrence={
        k: v for k, v in FAKE["cooccurrence"].items() if k not in
        ("电子|温度升高", "电子|位移变大")})
    path = TMP / "no_ec.json"
    path.write_text(json.dumps(empty, ensure_ascii=False), encoding="utf-8")
    vault = Vault(path)
    on = run_feynman_print(vault, PrintRequest(
        style_deg=60, nonce=6, retrieve_consequence=True))
    off = run_feynman_print(vault, PrintRequest(style_deg=60, nonce=6))
    assert on["printed_collapse"] == off["printed_collapse"]
    assert on["paths_total"] == off["paths_total"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P10 解码期检索 {len(tests)} 项回归全部通过")
