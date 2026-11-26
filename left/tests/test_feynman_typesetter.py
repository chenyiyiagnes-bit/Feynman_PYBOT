"""费曼心智活字回归：出处可查、确定性、语体敏感、术语闸门、留白、骨架、算子。

不依赖 700MB 原始语料：测试用一份小型合成字库验证引擎逻辑；
仓库里存在真字库（data/feynman_style_vault.json）时，再跑一遍真实的出处与质量检查。
"""

import json
import os
import tempfile
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
FAKE_VAULT = TMP / "feynman_style_vault.json"

HOOKS = [
    {"word": "你看", "slot": "hook", "count": 30, "amp": 1.0, "domain": "everyday", "style": "speech",
     "phase_deg": 100.0, "sources": ["《别闹了，费曼先生》｜第一章", "《发现的乐趣》｜科学是什么"], "example": ""},
    {"word": "假设", "slot": "hook", "count": 25, "amp": 0.98, "domain": "everyday", "style": "dialogue",
     "phase_deg": 60.0, "sources": ["《费恩曼物理学讲义》｜第一卷", "《QED》｜第一章"], "example": ""},
    {"word": "注意", "slot": "hook", "count": 20, "amp": 0.96, "domain": "everyday", "style": "lecture",
     "phase_deg": 18.0, "sources": ["《费恩曼物理学讲义》｜第二卷", "《物理定律的本性》｜第三章"], "example": ""},
]
ENTITIES = [
    {"word": "水面", "slot": "entity", "count": 40, "amp": 1.0, "domain": "everyday", "style": "dialogue",
     "phase_deg": 58.0, "sources": ["《费恩曼物理学讲义》｜第一卷", "《QED》｜第二章"], "example": ""},
    {"word": "电子", "slot": "entity", "count": 50, "amp": 0.99, "domain": "physics", "style": "lecture",
     "phase_deg": 20.0, "sources": ["《费恩曼物理学讲义》｜第三卷", "《QED》｜第一章"], "example": ""},
    {"word": "咖啡杯", "slot": "entity", "count": 12, "amp": 0.95, "domain": "everyday", "style": "speech",
     "phase_deg": 98.0, "sources": ["《别闹了，费曼先生》｜第二章", "《发现的乐趣》｜科学是什么"], "example": ""},
]
ACTIONS = [
    {"word": "旋转", "slot": "action", "count": 22, "amp": 0.98, "domain": "everyday", "style": "lecture",
     "phase_deg": 30.0, "sources": ["《费恩曼物理学讲义》｜第一卷", "《物理定律的本性》｜第二章"], "example": ""},
    {"word": "穿过", "slot": "action", "count": 18, "amp": 0.97, "domain": "everyday", "style": "dialogue",
     "phase_deg": 68.0, "sources": ["《QED》｜第三章", "《费恩曼物理学讲义》｜第二卷"], "example": ""},
    {"word": "想象", "slot": "action", "count": 16, "amp": 0.96, "domain": "everyday", "style": "speech",
     "phase_deg": 110.0, "sources": ["《发现的乐趣》｜科学是什么", "《别闹了，费曼先生》｜第三章"], "example": ""},
]
CONSEQUENCES = [
    {"word": "温度升高", "slot": "consequence", "count": 12, "amp": 0.97, "domain": "physics", "style": "lecture",
     "phase_deg": 40.0, "sources": ["《费恩曼物理学讲义》｜第一卷", "《物理定律的本性》｜第一章"], "example": ""},
    {"word": "出现一圈条纹", "slot": "consequence", "count": 8, "amp": 0.95, "domain": "physics", "style": "dialogue",
     "phase_deg": 78.0, "sources": ["《QED》｜第一章", "《费恩曼物理学讲义》｜第三卷"], "example": ""},
    {"word": "整条街都听见了", "slot": "consequence", "count": 6, "amp": 0.94, "domain": "physics", "style": "speech",
     "phase_deg": 118.0, "sources": ["《别闹了，费曼先生》｜第四章", "《发现的乐趣》｜科学是什么"], "example": ""},
]
SKELETONS = [
    {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
     "phase_deg": 60.0, "freq": 40,
     "provenance": ["《费恩曼物理学讲义》｜第一卷｜（样例句）"]},
    {"pattern_type": "conditional", "template": "{hook}，假设{entity}{action}，{consequence}。",
     "phase_deg": 25.0, "freq": 20,
     "provenance": ["《费恩曼物理学讲义》｜第二卷｜（样例句）"]},
    {"pattern_type": "retro", "template": "{consequence}——之所以会这样，是因为{hook}，{entity}{action}。",
     "phase_deg": 105.0, "freq": 12,
     "provenance": ["《别闹了，费曼先生》｜第三章｜（样例句）"]},
]
FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 100,
    "slot_offsets": {"hook": -15.0, "entity": 2.0, "action": 9.0, "consequence": 18.0},
    "skeletons": SKELETONS,
    "hooks": HOOKS, "entities": ENTITIES, "actions": ACTIONS, "consequences": CONSEQUENCES,
    "entangled_pairs": [{"everyday": "水面", "physics": "出现一圈条纹", "count": 5}],
    "cooccurrence": {"水面|出现一圈条纹": 5, "电子|温度升高": 2, "旋转|出现一圈条纹": 1},
}
FAKE_VAULT.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_VAULT"] = str(FAKE_VAULT)

from fastapi.testclient import TestClient

from bus.main import app
from plugins.feynman_typesetter import (
    OperatorWeights,
    PrintRequest,
    Vault,
    _choose_skeleton,
    _evaluate_path,
    _select_per_slot,
    run_feynman_print,
)

client = TestClient(app)
vault = Vault()
REAL_VAULT = Path("data/feynman_style_vault.json")


def _print(**kw):
    body = {"style_deg": 60.0, "nonce": 0, "top_k": 8, **kw}
    resp = client.post("/feynman/print", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_vault_stats_and_api():
    stats = client.get("/feynman/vault").json()
    assert stats["glyphs"] == {"hook": 3, "entity": 3, "action": 3, "consequence": 3}
    assert stats["bank_id"].startswith("feynman-")
    assert stats["slot_offsets"]["consequence"] == 18.0
    assert len(stats["skeletons"]) == 3
    assert stats["cooccurrence_pairs"] == 3


def test_operators_metadata_endpoint():
    meta = client.get("/feynman/operators").json()
    assert set(meta["presets"]) >= {"均衡费曼全脑", "黑板物理推演", "酒吧科普神比喻", "侦探由果溯因"}
    assert meta["max_weight"] == 2.0


def test_print_is_deterministic_for_same_seed():
    a = _print(style_deg=60, nonce=7)
    b = _print(style_deg=60, nonce=7)
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a["seed"] == b["seed"]
    assert a["collapsed"] is True


def test_style_dial_changes_the_printed_line():
    tops = [_print(style_deg=s)["printed_collapse"] for s in (10, 60, 110)]
    assert len(set(tops)) == 3, tops


def test_printed_line_uses_glyphs_from_the_vault():
    out = _print(style_deg=60)
    words = {g["word"] for slot in ("hooks", "entities", "actions", "consequences") for g in FAKE[slot]}
    line = out["top_resonant_line"]
    assert line.endswith("。")
    for word in out["paths"][0]["glyphs"].values():
        assert word in line, (word, line)
        assert word in words
    for src in out["paths"][0]["sources"].values():
        assert src and "｜" in src[0]
    # 本次坍缩句也必须是完整骨架句
    assert out["printed_collapse"].endswith("。")


def test_every_glyph_keeps_provenance():
    for slot in ("hooks", "entities", "actions", "consequences"):
        for g in FAKE[slot]:
            assert g["sources"], g["word"]
            assert all("｜" in s for s in g["sources"])


def test_skeleton_selected_by_style_and_rendered():
    mid = _print(style_deg=60)
    assert mid["skeleton"]["pattern_type"] == "declarative"
    low = _print(style_deg=10)
    assert low["skeleton"]["pattern_type"] == "conditional"
    high = _print(style_deg=110)
    assert high["skeleton"]["pattern_type"] == "retro"


def test_measured_slot_offsets_drive_selection():
    assert vault.slot_offsets == {"hook": -15.0, "entity": 2.0, "action": 9.0, "consequence": 18.0}
    picked = _select_per_slot(vault, 60.0, 3, "")
    # 后果目标 60+18=78°，最贴的是 78° 的"出现一圈条纹"
    assert picked["consequence"][0]["word"] == "出现一圈条纹"


def test_operator_weights_change_weight_and_emit_badges():
    path = {
        "hook": HOOKS[0], "entity": ENTITIES[0], "action": ACTIONS[0], "consequence": CONSEQUENCES[1],
    }
    base = _evaluate_path(path, 60.0, vault.jargon, set(), vault=vault, weights=OperatorWeights())
    hot = _evaluate_path(
        path, 60.0, vault.jargon, set(), vault=vault,
        weights=OperatorWeights(micro=2, cross=2, boundary=2, reversal=2, anti_intuition=2),
    )
    assert base["operator_scores"] and hot["operator_scores"]
    assert hot["weight"] != base["weight"]
    assert hot["badges"], hot["operator_scores"]


def test_zero_operators_equal_baseline():
    a = _print(style_deg=60, nonce=3)
    b = _print(style_deg=60, nonce=3, operators={})
    assert a["printed_collapse"] == b["printed_collapse"]
    assert a["paths"][0]["weight"] == b["paths"][0]["weight"]


def test_skeleton_votes_follow_operator_weights():
    retro = _choose_skeleton(vault, 100.0, OperatorWeights(reversal=2.0))
    assert retro["pattern_type"] == "retro"
    conditional = _choose_skeleton(vault, 25.0, OperatorWeights(boundary=2.0))
    assert conditional["pattern_type"] == "conditional"


def test_purity_and_killed_fields_present():
    out = _print(style_deg=60)
    assert 0.0 <= out["purity"] <= 1.0
    assert isinstance(out["killed"], list)
    assert out["operators"] == OperatorWeights().model_dump()


def test_jargon_stack_is_vetoed():
    path = {
        "hook": {"word": "注意", "phase_deg": 20.0, "domain": "everyday"},
        "entity": {"word": "认知维度", "phase_deg": 20.0, "domain": "everyday"},
        "action": {"word": "赋能", "phase_deg": 20.0, "domain": "everyday"},
        "consequence": {"word": "温度升高", "phase_deg": 40.0, "domain": "physics"},
    }
    out = _evaluate_path(path, 30.0, vault.jargon, set())
    assert out["weight"] == 0.0
    assert out["pre_veto_weight"] > 0
    assert "🚫" in " ".join(out["notes"])
    assert len(out["jargon_hits"]) >= 2


def test_single_jargon_term_is_penalized_not_banned():
    clean = {
        "hook": {"word": "注意", "phase_deg": 20.0, "domain": "everyday"},
        "entity": {"word": "电子", "phase_deg": 20.0, "domain": "physics"},
        "action": {"word": "旋转", "phase_deg": 30.0, "domain": "everyday"},
        "consequence": {"word": "温度升高", "phase_deg": 40.0, "domain": "physics"},
    }
    tainted = json.loads(json.dumps(clean))
    tainted["action"]["word"] = "赋能"
    a = _evaluate_path(clean, 30.0, vault.jargon, set())
    b = _evaluate_path(tainted, 30.0, vault.jargon, set())
    assert a["weight"] > 0 and b["weight"] > 0
    assert b["weight"] < a["weight"]


def test_unrelated_topic_prints_blank_glyph():
    out = _print(style_deg=60, topic="股票回购与货币政策")
    assert out["collapsed"] is False
    assert out["blank"] is True
    assert "没有可测的东西" in out["printed_collapse"]


def test_unknown_bank_id_returns_404():
    resp = client.post("/feynman/print", json={"bank_id": "nope", "style_deg": 60})
    assert resp.status_code == 404


def test_style_out_of_range_is_rejected():
    resp = client.post("/feynman/print", json={"style_deg": 200})
    assert resp.status_code == 422


def test_operator_out_of_range_is_rejected():
    resp = client.post("/feynman/print", json={"style_deg": 60, "operators": {"micro": 5}})
    assert resp.status_code == 422


def test_print_latency_under_budget():
    run_feynman_print(vault, PrintRequest(style_deg=60))  # warm
    started = time.perf_counter()
    for _ in range(5):
        run_feynman_print(vault, PrintRequest(style_deg=60))
    elapsed = (time.perf_counter() - started) / 5
    assert elapsed < 0.05, f"单次排版 {elapsed:.3f}s 超过 50ms"


def test_real_vault_is_corpus_grounded():
    if not REAL_VAULT.is_file():
        print("   （跳过：仓库内无真字库，先运行 plugins/feynman_vault.py）")
        return
    real = Vault(REAL_VAULT)
    stats = real.stats()
    assert stats["sections"] > 500
    for slot in ("hook", "entity", "action", "consequence"):
        # 后果槽按"可观测状态"硬过滤（动词开头的片段一律不要），所以数量本就最少
        minimum = 15 if slot == "consequence" else 20
        assert stats["glyphs"][slot] >= minimum, (slot, stats["glyphs"])
    assert len(stats["skeletons"]) >= 3, stats["skeletons"]
    assert stats["cooccurrence_pairs"] >= 50, stats["cooccurrence_pairs"]
    for slot, offset in stats["slot_offsets"].items():
        assert -25.0 <= offset <= 25.0, (slot, offset)
    # 字粒出处必须落在费曼本人的作品上。审计库用几十种英文篇目标签（口述史、
    # FLP 选章、各场讲座），逐项白名单不现实；反过来守红线：讲费曼的书、
    # 书评、讣告、他人著作一律不许出现，且每个字粒都要有出处。
    from plugins.feynman_vault import EXCLUDED_SOURCE_MARKERS

    checked = 0
    for slot, items in real.by_slot.items():
        for g in items:
            assert g["sources"], (slot, g["word"])
            assert not any(
                m.lower() in s.lower()
                for s in g["sources"] for m in EXCLUDED_SOURCE_MARKERS
            ), g
            if slot == "entity":
                assert not g["word"].endswith(("性", "化")), g["word"]
            checked += 1
    assert checked >= 100
    out = run_feynman_print(real, PrintRequest(style_deg=60, nonce=3))
    assert out["collapsed"] is True
    assert out["skeleton"] and out["skeleton"]["provenance"]
    assert 0.0 <= out["purity"] <= 1.0


def test_real_vault_has_no_meta_discourse_or_death_content():
    if not REAL_VAULT.is_file():
        print("   （跳过：仓库内无真字库，先运行 plugins/feynman_vault.py）")
        return
    from plugins.feynman_vault import DEATH_MARKERS, META_DISCOURSE_NOUNS, META_HOOK_STOP

    real = Vault(REAL_VAULT)
    banned = set(META_DISCOURSE_NOUNS) | set(META_HOOK_STOP)
    for slot in ("hook", "entity", "consequence"):
        for g in real.by_slot[slot]:
            assert g["word"] not in banned, (slot, g["word"])
            assert not any(m in g["word"] for m in DEATH_MARKERS), g["word"]
    # 50 个 nonce 印出的句子，四槽里都不许再出现元话语名词
    for nonce in range(50):
        out = run_feynman_print(real, PrintRequest(style_deg=60, nonce=nonce))
        for word in out["paths"][0]["glyphs"].values():
            assert word not in banned, (nonce, word)
    # 语料过滤层面：一行讣告/悼念段落都不该进来
    from plugins.feynman_vault import iter_sections
    assert all(
        not any(m in text for m in DEATH_MARKERS)
        for _src, _section, text in iter_sections()
    )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 费曼心智活字 {len(tests)} 项回归全部通过")
