"""受控进化闭环回归（P6.6）：显式反馈、隐式延续、钳位、剪枝、GC 与隔离。"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import plugins.domain_expander as de
import plugins.evolution_loop as ev
from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print

REAL_VAULT = Path("data/feynman_style_vault.json")
HAS_REAL = REAL_VAULT.is_file()
ALLOWED = None  # 由真实字库填充的动作白名单


def _setup(tmpdir: Path):
    de.EXT_PATH = tmpdir / "domain_extensions.json"
    return de.EXT_PATH


def _load():
    return de.load_domain_extensions_with_hash()


def test_sanitize_entity_rejects_injection_and_short_words():
    assert ev.sanitize_entity("股票") == "股票"
    assert ev.sanitize_entity(" 通货膨胀 ") == "通货膨胀"
    for bad in ("", "股", "1234", "股票；rm -rf", "a|b", "x\ny", "九" * 9, None):
        assert ev.sanitize_entity(bad) is None, bad


def test_sanitize_rejects_action_outside_whitelist():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        h = ev.apply_evolution_update(["股票"], "做空", "", 0.3, "test",
                                      allowed_actions={"观察", "计算"})
        assert h == de.EMPTY_EXT_HASH
        assert not _load()[0].get("edge_scores"), "非法动作不得入库"
        h = ev.apply_evolution_update(["股票"], "观察", "", 0.3, "test",
                                      allowed_actions={"观察", "计算"})
        assert h != de.EMPTY_EXT_HASH


def test_upvote_promotes_edge_and_clamps():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        for _ in range(3):
            ev.apply_evolution_update(["股票"], "观察", "declarative", 0.3, "up", {"观察"})
        data, h = _load()
        assert data["edge_scores"]["股票|观察"] == 1.5, "三次 +0.3 应钳到 1.5"
        assert ["股票", "观察"] in data["cooccurrence_edges"], "过 1.15 应升级为共现边"
        assert data["skeleton_scores"]["declarative"] == 1.45
        assert data["ext_hash"] == h


def test_downvotes_prune_edge_and_cooc():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        ev.apply_evolution_update(["股票"], "观察", "", 0.3, "up", {"观察"})
        assert "股票|观察" in _load()[0]["edge_scores"]
        ev.apply_evolution_update(["股票"], "观察", "", -0.45, "down", {"观察"})
        ev.apply_evolution_update(["股票"], "观察", "", -0.45, "down", {"观察"})
        assert _load()[0]["edge_scores"]["股票|观察"] == 0.4, "两踩应降到 0.4 但仍在"
        ev.apply_evolution_update(["股票"], "观察", "", -0.45, "down", {"观察"})
        data, _ = _load()
        assert "股票|观察" not in data["edge_scores"], "跌破 0.15 必须剪掉"
        assert ["股票", "观察"] not in data["cooccurrence_edges"]


def test_gc_keeps_highest_scores_not_alphabetical():
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        scores = {f"e{i:03d}|观察": round(0.2 + i * 0.001, 3) for i in range(600)}
        scores["zzz|观察"] = 1.4  # 字典序最后但分数最高，必须留下
        de.atomic_save_extensions({"edge_scores": scores,
                                   "cooccurrence_edges": [["zzz", "观察"]]})
        ev.apply_evolution_update(["股票"], "观察", "", 0.3, "gc", {"观察"})
        data, _ = _load()
        assert len(data["edge_scores"]) == ev.MAX_EDGE_ENTRIES
        assert data["edge_scores"]["zzz|观察"] == 1.4
        assert data["edge_scores"]["e599|观察"] == 0.799, "应按分数保留高位而非按键名"


def test_implicit_followup_scores_previous_turn_only():
    tr = ev.ImplicitSignalTracker()
    assert tr.observe(None, ["股票"], "观察", "declarative", 60) is None
    assert tr.observe("s1", ["股票"], "观察", "declarative", 60) is None
    sig = tr.observe("s1", ["股价"], "计算", "causal", 63)
    assert sig and sig["reason"] == "implicit_followup" and sig["delta"] > 0
    assert sig["entities"] == ["股票"] and sig["action"] == "观察"
    drift = tr.observe("s1", ["合同"], "检查", "retro", 30)
    assert drift and drift["reason"] == "implicit_topic_shift" and drift["delta"] < 0
    assert drift["entities"] == ["股价"]


def test_implicit_overlap_beats_phase_distance():
    tr = ev.ImplicitSignalTracker()
    tr.observe("s2", ["股票", "股价"], "观察", "declarative", 90)
    sig = tr.observe("s2", ["股价"], "计算", "causal", 10)
    assert sig["reason"] == "implicit_followup", "实体延续优先于相位漂移"


def test_learned_multiplier_only_touches_user_entity_paths():
    if not HAS_REAL:
        return
    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        v = Vault()
        req = PrintRequest(style_deg=60, nonce=1, user_query="股票回购是什么")
        base = run_feynman_print(v, req)
        assert base["collapsed"]
        target = next(p for p in base["paths"]
                      if p["glyphs"]["entity"] in base["user_entities"])
        ent, act = target["glyphs"]["entity"], target["glyphs"]["action"]
        # 只给这一条用户实体路径加乘子，选路（骨架/字粒）不变
        boosted = run_feynman_print(v, req, learned_edges={f"{ent}|{act}": 1.5})
        key = "，".join(target["glyphs"][s] for s in ("hook", "entity", "action", "consequence"))
        b_line = {p["line"]: p for p in boosted["paths"]}.get(key)
        assert b_line, "加权路径应仍在候选里"
        assert abs(b_line["weight"] / target["weight"] - 1.5) < 1e-3
        assert any("反馈习得" in n for n in b_line["notes"])
        # 基线路径（无用户实体）绝不受这个乘子影响
        plain = PrintRequest(style_deg=60, nonce=1)
        p1 = run_feynman_print(v, plain)
        p2 = run_feynman_print(v, plain, learned_edges={f"{ent}|{act}": 1.5,
                                                         "咖啡杯|旋转": 1.5})
        assert p1["paths"][0]["weight"] == p2["paths"][0]["weight"]


def test_feedback_endpoint_validates_and_persists():
    from fastapi.testclient import TestClient
    from bus.main import app

    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        good = client.post("/feynman/feedback", json={
            "vote": "up", "entities": ["股票"], "action": "观察",
            "skeleton": "declarative", "session_id": "s1"})
        assert good.status_code == 200, good.text
        body = good.json()
        assert body["ok"] and body["ext_hash"] != de.EMPTY_EXT_HASH
        stored = json.loads(de.EXT_PATH.read_text())
        assert stored["edge_scores"]["股票|观察"] == 1.3
        assert stored["ext_hash"] == body["ext_hash"]

        for bad in (
            {"vote": "meh", "entities": ["股票"], "action": "观察"},
            {"vote": "up", "entities": ["股票"], "action": "做空"},
            {"vote": "up", "entities": [], "action": "观察"},
            {"vote": "up", "entities": ["股票；rm"], "action": "观察"},
        ):
            r = client.post("/feynman/feedback", json=bad)
            assert r.status_code == 400, (bad, r.status_code, r.text)


def test_print_exposes_entity_weights_and_writes_nothing_without_feedback():
    from fastapi.testclient import TestClient
    from bus.main import app

    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        r = TestClient(app).post("/feynman/print", json={
            "style_deg": 60, "nonce": 0, "user_query": "股票回购是什么"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ext_hash"] == de.EMPTY_EXT_HASH
        assert body["entity_weights"], "应有用户实体的经验乘子视图"
        assert not de.EXT_PATH.exists(), "无反馈不得创建扩展库"


def test_feedback_roundtrip_shifts_print_but_stays_deterministic():
    from fastapi.testclient import TestClient
    from bus.main import app

    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        payload = {"style_deg": 60, "nonce": 2, "user_query": "股票回购是什么"}
        before = client.post("/feynman/print", json=payload).json()
        ent = before["user_entities"][0]
        act = before["paths"][0]["glyphs"]["action"]
        client.post("/feynman/feedback", json={"vote": "up", "entities": [ent],
                                               "action": act})
        after1 = client.post("/feynman/print", json=payload).json()
        after2 = client.post("/feynman/print", json=payload).json()
        assert after1["ext_hash"] != before["ext_hash"]
        assert after1["seed"] == after2["seed"] != before["seed"]
        assert after1["entity_weights"][ent] == 1.3


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 进化闭环 {len(tests)} 项回归全部通过")
