"""P21 同构空间：产物结构、乘子边界、默认关逐字节一致、段落实体锁。

只读测试，不改任何数据文件。运行时数秒。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_p21_isomorphism.py
"""

import json
import math
import os
from pathlib import Path

os.environ.setdefault("QBOT_FEYNMAN_VAULT", "data/feynman_style_vault_p15.json")
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

from plugins.feynman_typesetter import PrintRequest, Vault, run_feynman_print
from plugins.isomorphism_p21 import (
    COMPAT_HI,
    COMPAT_LO,
    COMPAT_PERC_HI,
    COMPAT_PERC_LO,
    P21Context,
    load_engine,
)
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph

ARTIFACT = Path("data/isomorphism_p21.json")
BANDWIDTH = Path("data/phase_bandwidth_p21.json")
VAULT = Path("data/feynman_style_vault_p15.json")


def test_artifact_structure_and_sign_lock():
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    meta = d["meta"]
    assert meta["k"] == 12, meta
    assert meta["words"] >= 650, meta
    assert len(d["vectors"]) == meta["words"]
    assert len(d["axes_audit"]) == 12
    assert "corpus_sha256" in meta and "vault_sha256" in meta
    for w, vec in d["vectors"].items():
        assert len(vec) == 12, (w, len(vec))
    # 符号锁是按轴做的：每维载荷最大的那个字粒必须为正（整轴未被翻错方向）。
    for axis in range(12):
        loads = [(abs(v[axis]), v[axis]) for v in d["vectors"].values()]
        peak_abs, peak = max(loads)
        assert peak > 0, f"第 {axis} 维符号锁失效"


def test_engine_load_and_bounds():
    engine = load_engine()
    assert engine is not None
    vec = engine.vec("电子")
    assert vec is not None and abs(math.sqrt(sum(x * x for x in vec)) - 1.0) < 1e-6
    # 乘子必须在 [0.6, 1.6]，且对 score 单调
    ctx = P21Context(engine, ["股票"], ["下跌"], {}, None)
    last = -1.0
    for c in ("振幅减小", "能量保持不变", "动量改变"):
        m, info = ctx.multiplier("股票", "旋转", c)
        assert COMPAT_LO - 1e-9 <= m <= COMPAT_HI + 1e-9, (m, info)
        assert info["score"] is not None
        assert m >= last - 1e-9 or info.get("clamped")
        last = m


def test_oov_sparsification():
    engine = load_engine()
    vec, info = engine.oov_vector("股票", ["下跌"], None, "液体")
    assert info["source"] == "oov"
    assert vec is not None
    nz = [i for i, x in enumerate(vec) if abs(x) > 1e-9]
    assert len(nz) <= 3, nz


def test_bandwidth_alignment_bounded():
    engine = load_engine()
    for w in ("电子", "振幅", "轨道"):
        for target in (15.0, 60.0, 95.0):
            a = engine.aligned(w, 40.0, target)
            assert -1.0 - 1e-9 <= a <= 1.0 + 1e-9, (w, target, a)
    # 无带宽记录时退回纯余弦
    assert abs(engine.aligned("不存在的词", 10.0, 40.0)
               - math.cos(math.radians(10.0 - 40.0))) < 1e-9


def test_flag_off_is_byte_identical():
    vault = Vault(VAULT)
    base = PrintRequest(user_query="为什么电子会旋转", style_deg=60.0, nonce=11)
    off = run_feynman_print(vault, base)
    explicit = run_feynman_print(
        vault, PrintRequest(user_query="为什么电子会旋转", style_deg=60.0,
                            nonce=11, use_isomorphism=False))
    assert off["printed_collapse"] == explicit["printed_collapse"]
    assert [p["line"] for p in off["paths"]] == [p["line"] for p in explicit["paths"]]
    assert "p21_diagnostics" not in off


def test_flag_on_records_diagnostics_only():
    vault = Vault(VAULT)
    out = run_feynman_print(vault, PrintRequest(
        user_query="股票为什么会下跌", style_deg=60.0, nonce=5, use_isomorphism=True))
    diag = out.get("p21_diagnostics")
    assert diag, "开启后应有诊断字段"
    for row in diag:
        assert 0.0 <= row["residual"] <= 1.0 + 1e-6
        assert 0.0 <= row["entropy"] <= 1.0 + 1e-6
    # 诊断只记录：概率字段仍是合法概率且按权重降序（归一在全部存活路径上做，
    # 返回的只是 top_k，不能要求它们的和等于 1）
    probs = [p["probability"] for p in out["paths"] if p.get("probability")]
    assert probs and all(0.0 <= x <= 1.0 for x in probs)
    assert probs == sorted(probs, reverse=True)


def test_paragraph_entity_lock():
    vault = Vault(VAULT)
    out = weave_paragraph(vault, ParagraphRequest(
        user_query="股票为什么会下跌", style_deg=70.0, nonce=5,
        beam_width=6, use_isomorphism=True))
    assert out["complete"], out["paragraph"]
    entities = [(s.get("glyphs") or {}).get("entity") for s in out["sentences"]]
    assert entities[0] == "股票", entities
    assert entities[2] == entities[0], entities


def test_oov_verb_projection_sparse():
    engine = load_engine()
    # 语料内的谓语直接取库向量；语料外走字面重叠投影，仍要稀疏
    assert engine.verb_vector("旋转")[1]["source"] == "corpus"
    vec, info = engine.verb_vector("下跌")
    assert info["source"] in ("oov_verb", "corpus"), info
    if info["source"] == "oov_verb":
        nz = [i for i, x in enumerate(vec) if abs(x) > 1e-9]
        assert 1 <= len(nz) <= 3, nz


def test_query_mechanism_combines_entity_and_verb():
    engine = load_engine()
    ctx = P21Context(engine, ["股票"], ["下跌"], {}, None)
    vec, info = ctx.query_mechanism("股票")
    assert vec is not None and len(vec) == 12
    assert info["entity"] == "oov"
    assert "下跌" in info["verbs"]
    # 缓存：同一谓语第二次调用不重算（对象同一个）
    assert ctx.verb_vector("下跌")[0] is ctx.verb_vector("下跌")[0]


def test_pool_percentile_multiplier_bounds():
    vault = Vault(VAULT)
    out = run_feynman_print(vault, PrintRequest(
        user_query="咖啡为什么会变凉", style_deg=60.0, nonce=5,
        use_isomorphism=True, top_k=50))
    pool = out.get("p21_pool") or {}
    assert pool.get("sim_scored", 0) > 0, pool
    scored = [p for p in out["paths"] if p.get("p21") and p["p21"].get("percentile_pool")]
    for p in scored:
        info = p["p21"]
        assert 0.0 <= info["percentile"] <= 1.0, info
        assert COMPAT_PERC_LO - 1e-9 <= info["multiplier"] <= COMPAT_PERC_HI + 1e-9, info
    for p in out["paths"]:
        if p.get("p21_pool"):
            assert 0.85 <= p["p21_pool"]["multiplier"] <= 1.15, p["p21_pool"]


def test_pool_single_path_is_neutral():
    engine = load_engine()
    ctx = P21Context(engine, ["股票"], ["下跌"], {}, None)
    path = {"glyphs": {"entity": "股票", "action": "旋转", "consequence": "动量改变"},
            "weight": 1.0, "p21": {"score": 0.1}}
    ctx.rerank_pool([path], "股票")
    assert abs(path["p21"]["multiplier"] - 1.0) < 1e-9, path["p21"]


def test_paragraph_flag_off_unchanged():
    vault = Vault(VAULT)
    base = dict(user_query="为什么光会弯曲", style_deg=70.0, nonce=3, beam_width=6)
    a = weave_paragraph(vault, ParagraphRequest(**base))
    b = weave_paragraph(vault, ParagraphRequest(**base, use_isomorphism=False))
    assert a["paragraph"] == b["paragraph"]
    assert [s["line"] for s in a["sentences"]] == [s["line"] for s in b["sentences"]]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P21 同构测试 {len(tests)} 项全部通过")
