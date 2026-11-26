"""用户实体注入 + 会话记忆回归（P5.2 / P5.3）。

用仓库真字库；字库缺失则跳过真库相关断言，只跑纯逻辑部分。
"""

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("QBOT_CACHE_TTL_SEC", "0")

from fastapi.testclient import TestClient

from bus.main import app
from plugins.feynman_memory import SessionMemory
from plugins.feynman_typesetter import (
    PROTOTYPE_MAP,
    SAFE_UNIVERSAL_ACTIONS,
    OperatorWeights,
    PrintRequest,
    Vault,
    extract_entities,
    run_feynman_print,
)

client = TestClient(app)
REAL_VAULT = Path("data/feynman_style_vault.json")
BASELINE = Path("tests/fixtures_feynman_baseline.json")
HAS_REAL = REAL_VAULT.is_file()


def _vault():
    # 基线夹具由 base 字库生成；产品默认库（.env 指向 P15）不影响本测试
    return Vault(REAL_VAULT)


def _print(**kw):
    body = {"style_deg": 60.0, "nonce": 0, "top_k": 8, **kw}
    resp = client.post("/feynman/print", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_extract_entities_drops_meta_discourse_and_jargon():
    assert extract_entities("这个角度的问题在于结论和说法") == []
    assert extract_entities("我们来看看这个例子") == []
    ents = extract_entities("量子纠缠是什么", ["量子纠缠"])
    assert "量子纠缠" not in ents
    assert extract_entities("") == []


def test_extract_entities_keeps_user_nouns_in_order():
    ents = extract_entities("我的咖啡杯为什么凉得这么快")
    assert ents[:1] == ["咖啡杯"]
    ents = extract_entities("汽车在冰面上刹车时轮胎发生了什么")
    assert ents[0] == "汽车" and len(ents) <= 3


def test_baseline_output_unchanged_without_user_fields():
    if not (HAS_REAL and BASELINE.is_file()):
        return
    v = _vault()
    for key, exp in json.loads(BASELINE.read_text(encoding="utf-8")).items():
        style, nonce = map(int, key.split("|"))
        out = run_feynman_print(v, PrintRequest(style_deg=style, nonce=nonce))
        assert out["printed_collapse"] == exp["line"], key
        assert abs(out["paths"][0]["weight"] - exp["weight"]) < 1e-9, key
        assert out["user_entities"] == []


def test_user_entity_is_injected_and_tops_paths():
    if not HAS_REAL:
        return
    v = _vault()
    out = run_feynman_print(v, PrintRequest(style_deg=60, nonce=3,
                                            user_query="我的咖啡杯为什么凉得这么快"))
    assert out["user_entities"] == ["咖啡杯"]
    assert out["paths"][0]["glyphs"]["entity"] == "咖啡杯"
    assert out["paths"][0]["sources"]["entity"] == ["用户提问"]
    assert any("落在你问的东西上" in n for n in out["paths"][0]["notes"])


def test_incompatible_action_is_penalized_not_banned():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _evaluate_path, _user_glyph

    v = _vault()
    bad_action = next(
        g["word"] for g in v.by_slot["action"]
        if g["word"] not in SAFE_UNIVERSAL_ACTIONS
        and v.pair_support("咖啡杯", g["word"]) == 0
        and not v.pair_support(PROTOTYPE_MAP.get("咖啡杯", ""), g["word"])
    )
    act = next(g for g in v.by_slot["action"] if g["word"] == bad_action)
    path = {
        "hook": v.by_slot["hook"][0],
        "entity": _user_glyph("咖啡杯", 60.0, "咖啡杯为什么凉得这么快"),
        "action": act,
        "consequence": v.by_slot["consequence"][0],
    }
    out = _evaluate_path(path, 60.0, v.jargon, set(), vault=v, weights=OperatorWeights())
    assert out["weight"] > 0, "扣权不是封杀"
    assert any("没有共现" in n for n in out["notes"])


def test_analogy_keeps_user_entity_in_jiuxiang_slot():
    if not HAS_REAL:
        return
    v = _vault()
    out = run_feynman_print(v, PrintRequest(
        style_deg=92, nonce=1, user_query="咖啡杯为什么凉得这么快",
        operators=OperatorWeights(cross=2.0),
    ))
    assert out["skeleton"]["pattern_type"] == "analogy", out["skeleton"]
    assert "就像咖啡杯" in out["top_resonant_line"], out["top_resonant_line"]


def test_request_never_writes_the_vault_file():
    if not HAS_REAL:
        return
    before = hashlib.sha256(REAL_VAULT.read_bytes()).hexdigest()
    mtime = REAL_VAULT.stat().st_mtime
    _print(style_deg=60, user_query="电子为什么有质量", session_id="write-guard")
    time.sleep(0.01)
    assert hashlib.sha256(REAL_VAULT.read_bytes()).hexdigest() == before
    assert REAL_VAULT.stat().st_mtime == mtime


def test_session_memory_constructive_and_pruning():
    m = SessionMemory()
    m.update(60.0)
    assert m.magnitude() == 1.0 and m.turns == 1
    m.update(70.0)
    assert m.magnitude() > 1.0
    for _ in range(3):
        m.update(70.0)
    assert m.magnitude() <= 2.0
    # 反向一轮：旧方向被抵消到阈值以下，旧线索退相干
    m.update(250.0)
    assert m.pruned is True
    assert m.magnitude() == 1.0


def test_session_bias_is_bounded():
    m = SessionMemory()
    m.update(110.0)
    for _ in range(4):
        m.update(110.0)
    biased = m.biased_style(10.0)
    assert 10.0 < biased <= 25.0


def test_session_endpoint_tracks_turns_and_amplitude():
    a = _print(style_deg=60, nonce=1, session_id="p53-alpha")
    b = _print(style_deg=60, nonce=1, session_id="p53-alpha")
    c = _print(style_deg=60, nonce=1, session_id="p53-beta")
    assert a["session"]["turns"] == 1
    assert b["session"]["turns"] == 2
    assert b["session"]["amplitude"] >= a["session"]["amplitude"]
    assert c["session"]["turns"] == 1
    # 同一会话、同一 nonce：记忆已变，但结果仍然确定
    d = _print(style_deg=60, nonce=1, session_id="p53-beta")
    assert c["seed"] != a["seed"]
    assert d["session"]["turns"] == 2


def test_requests_without_session_have_no_session_field():
    out = _print(style_deg=60)
    assert "session" not in out


def test_new_operators_are_exposed():
    meta = client.get("/feynman/operators").json()
    assert {"jargon_strip", "least_action", "cargo_cult"} <= set(meta["labels"])
    assert len(meta["presets"]) >= 7
    assert {"剥掉术语", "最小作用量", "货物崇拜检验"} <= set(meta["presets"])


def _glyph(word, phase, domain="physics", slot="entity", order="post"):
    return {"word": word, "phase_deg": phase, "domain": domain, "amp": 1.0,
            "order": order, "sources": ["测试"]}


def test_jargon_strip_penalizes_namedropping():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _evaluate_path

    v = _vault()
    clean = {
        "hook": _glyph("注意", 20.0), "entity": _glyph("电子", 20.0),
        "action": _glyph("旋转", 30.0, order="post"), "consequence": _glyph("温度升高", 40.0),
    }
    tainted = json.loads(json.dumps(clean))
    tainted["action"]["word"] = "赋能"
    w = OperatorWeights(jargon_strip=2.0)
    a = _evaluate_path(clean, 30.0, v.jargon, set(), vault=v, weights=w)
    b = _evaluate_path(tainted, 30.0, v.jargon, set(), vault=v, weights=w)
    assert a["weight"] > b["weight"]
    assert b["operator_scores"]["jargon_strip"] < a["operator_scores"]["jargon_strip"]


def test_least_action_prefers_invariants():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _evaluate_path

    v = _vault()
    invariant = {
        "hook": _glyph("注意", 20.0), "entity": _glyph("电子", 20.0),
        "action": _glyph("旋转", 30.0, order="post"), "consequence": _glyph("温度不变", 38.0),
    }
    drifting = json.loads(json.dumps(invariant))
    drifting["consequence"]["word"] = "温度降低得很快而且到处都在变化"
    w = OperatorWeights(least_action=2.0)
    a = _evaluate_path(invariant, 30.0, v.jargon, set(), vault=v, weights=w)
    b = _evaluate_path(drifting, 30.0, v.jargon, set(), vault=v, weights=w)
    assert a["weight"] > b["weight"]
    assert a["operator_scores"]["least_action"] > b["operator_scores"]["least_action"]


def test_least_action_votes_for_short_skeletons():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _choose_skeleton

    v = _vault()
    sk = _choose_skeleton(v, 30.0, OperatorWeights(least_action=2.0))
    assert sk["pattern_type"] in {"conditional", "causal"}
    sk = _choose_skeleton(v, 30.0, OperatorWeights(cargo_cult=2.0))
    assert sk["pattern_type"] in {"retro", "contrast"}


def test_cargo_cult_rewards_falsifiable_consequences():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _evaluate_path

    v = _vault()
    testable = {
        "hook": _glyph("注意", 20.0), "entity": _glyph("电子", 20.0),
        "action": _glyph("测量", 30.0, order="pre"), "consequence": _glyph("实验读数的误差变小", 42.0),
    }
    vague = json.loads(json.dumps(testable))
    vague["consequence"]["word"] = "一切都很和谐"
    w = OperatorWeights(cargo_cult=2.0)
    a = _evaluate_path(testable, 30.0, v.jargon, set(), vault=v, weights=w)
    b = _evaluate_path(vague, 30.0, v.jargon, set(), vault=v, weights=w)
    assert a["weight"] > b["weight"]
    assert a["operator_scores"]["cargo_cult"] > b["operator_scores"]["cargo_cult"]


def test_new_operators_zero_keeps_baseline_weight():
    if not HAS_REAL:
        return
    from plugins.feynman_typesetter import _evaluate_path

    v = _vault()
    path = {
        "hook": _glyph("注意", 20.0), "entity": _glyph("电子", 20.0),
        "action": _glyph("旋转", 30.0, order="post"), "consequence": _glyph("温度升高", 40.0),
    }
    base = _evaluate_path(path, 30.0, v.jargon, set(), vault=v)
    zero = _evaluate_path(path, 30.0, v.jargon, set(), vault=v, weights=OperatorWeights())
    assert abs(base["weight"] - zero["weight"]) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 实体注入与会话记忆 {len(tests)} 项回归全部通过")
