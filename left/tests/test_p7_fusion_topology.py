"""P7 回归：自由能算子融合、拓扑层（PPMI/SVD/三角形）、类比传输、段落织构。

不联网、不写项目数据：扩展库指向临时文件，拓扑图在临时目录重建。
"""

import json
import os
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")
os.environ["QBOT_FEYNMAN_TOPOLOGY"] = str(TMP / "topology_graph.npz")
os.environ.pop("FEYNMAN_ENABLE_TOPOLOGY", None)

import numpy as np
from pydantic import ValidationError

from plugins.feynman_typesetter import (
    OperatorWeights,
    PrintRequest,
    Vault,
    _operator_multiplier,
    run_feynman_print,
)

HAS_REAL = Path("data/feynman_style_vault.json").is_file()
BASELINE = Path("tests/fixtures_feynman_baseline.json")
_VAULT = None
_ENGINE = None


def _vault():
    global _VAULT
    if _VAULT is None:
        _VAULT = Vault()
    return _VAULT


def _engine():
    global _ENGINE
    if _ENGINE is None:
        from plugins.topology_builder import TopologyManifoldEngine, build_topology
        build_topology(out_path=Path(os.environ["QBOT_FEYNMAN_TOPOLOGY"]))
        _ENGINE = TopologyManifoldEngine.load(Path(os.environ["QBOT_FEYNMAN_TOPOLOGY"]))
    return _ENGINE


# --- P7.1 自由能融合 -------------------------------------------------------

def test_boltzmann_zero_weights_exact_one():
    scores = {"micro": 0.0, "cross": 0.2, "boundary": 0.9, "reversal": 0.3,
              "anti_intuition": 0.1, "jargon_strip": 0.0, "least_action": 0.4,
              "cargo_cult": 0.8}
    assert _operator_multiplier(scores, OperatorWeights(), fusion="boltzmann") == 1.0
    # 线性模式在全零权重下同样恰好为 1.0
    assert _operator_multiplier(scores, OperatorWeights(), fusion="linear") == 1.0


def test_boltzmann_energy_is_monotone_and_clamped():
    low = {k: 0.0 for k in OperatorWeights().model_dump()}
    high = {k: 1.0 for k in OperatorWeights().model_dump()}
    w = OperatorWeights(**{k: 0.5 for k in low})
    assert _operator_multiplier(high, w, fusion="boltzmann") == 1.0
    assert _operator_multiplier(low, w, fusion="boltzmann") < 0.5
    # 满权重低分：E = 16；Boltzmann 不设 0.25 下限（下限会把整池截成常数），
    # 只保证 >0 且继续随能量单调下降。
    w2 = OperatorWeights(**{k: 2.0 for k in low})
    m2 = _operator_multiplier(low, w2, fusion="boltzmann")
    assert 0.0 < m2 < 0.01, m2


def test_boltzmann_baseline_matches_linear():
    if not (HAS_REAL and BASELINE.is_file()):
        return
    v = _vault()
    fixture = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key, exp in fixture.items():
        style, nonce = map(int, key.split("|"))
        out = run_feynman_print(v, PrintRequest(style_deg=style, nonce=nonce,
                                                fusion="boltzmann"))
        assert out["printed_collapse"] == exp["line"], key
        assert abs(out["paths"][0]["weight"] - exp["weight"]) < 1e-9, key


def test_invalid_fusion_rejected():
    try:
        PrintRequest(fusion="tensor")
    except ValidationError:
        return
    raise AssertionError("非法融合方式应被拒绝")


# --- P7.2 拓扑层 -----------------------------------------------------------

def test_topology_build_reproducible():
    from plugins.topology_builder import build_topology
    p1, p2 = TMP / "t1.npz", TMP / "t2.npz"
    m1 = build_topology(out_path=p1)
    m2 = build_topology(out_path=p2)
    assert m1["topo_hash"] == m2["topo_hash"]
    assert m1["dim"] == 64
    assert m1["words"] > 50
    assert m1["triangles"] > 0, "同句三元组不应为空"
    assert m1["edges"] > 0
    assert m1["analogy_pairs"] >= 30


def test_engine_vectors_and_geodesic_contract():
    e = _engine()
    assert e is not None
    # 词表里有只出现在起手式、没有任何共现边的字粒（向量为零，返回 None）
    word = next(w for w in e.words if e.get_vector(w) is not None)
    v = e.get_vector(word)
    assert v is not None and abs(float(np.linalg.norm(v)) - 1.0) < 1e-4
    assert e.get_vector("不存在的词") is None
    # 未知词：没有几何证据，乘数恒 1.0
    mult, simplex = e.compute_geodesic_multiplier("未知甲", "未知乙", "未知丙", "未知丁")
    assert mult == 1.0 and simplex is False
    # 已知实心三角形：标记为 simplex，乘数在钳位区间内
    triple = next(iter(e.simples))
    mult, simplex = e.compute_geodesic_multiplier("", *triple)
    assert simplex is True
    assert 0.25 <= mult <= 1.80


def test_topology_disabled_by_default():
    from plugins.topology_builder import get_engine
    os.environ.pop("FEYNMAN_ENABLE_TOPOLOGY", None)
    assert get_engine() is None


# --- P7.3 类比传输 ---------------------------------------------------------

def test_procrustes_matrix_is_orthogonal():
    e = _engine()
    r = e._r
    assert r is not None and r.size
    assert np.allclose(r.T @ r, np.eye(r.shape[0]), atol=1e-3)
    assert abs(float(np.linalg.det(r)) - 1.0) < 1e-3
    assert 0.0 < e.analogy_alpha <= 0.85


def test_analogy_alignment_range():
    e = _engine()
    sim = e.compute_analogy_alignment("电子", "质量")
    assert sim is not None and -1.0001 <= sim <= 1.0001
    assert e.compute_analogy_alignment("不存在的词", "质量") is None


# --- P7.4 段落织构 ---------------------------------------------------------

def test_weave_paragraph_three_distinct_stages():
    if not HAS_REAL:
        return
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
    req = ParagraphRequest(user_query="为什么天空是蓝的", style_deg=70, nonce=3,
                           beam_width=6)
    out = weave_paragraph(_vault(), req)
    assert out["complete"] is True
    assert [s["connector"] for s in out["sentences"]] == ["", "打个比方，", "说到底，"]
    lines = [s["line"] for s in out["sentences"]]
    assert len(set(lines)) == 3
    assert out["paragraph"] == "".join(s["connector"] + s["line"] for s in out["sentences"])
    again = weave_paragraph(_vault(), req)
    assert again["paragraph"] == out["paragraph"]
    assert again["seed"] == out["seed"]


def test_weave_paragraph_with_topology_reports_hash():
    if not HAS_REAL:
        return
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph
    e = _engine()
    out = weave_paragraph(_vault(), ParagraphRequest(user_query="股票为什么涨",
                                                     nonce=5, beam_width=4),
                          topology=e)
    assert out["complete"] is True
    assert out["topology_hash"] == e.topo_hash


def test_paragraph_endpoint_contract():
    from fastapi.testclient import TestClient
    from bus.main import app
    client = TestClient(app)
    body = client.post("/feynman/paragraph", json={
        "user_query": "为什么天空是蓝的", "style_deg": 70, "nonce": 2,
        "beam_width": 5}).json()
    assert body["complete"] is True and len(body["sentences"]) == 3
    assert body["sentences"][0]["connector"] == ""
    topo = client.get("/feynman/topology").json()
    assert topo["enabled"] is False and topo["loaded"] is False
    assert not Path(os.environ["QBOT_FEYNMAN_EXT"]).exists(), "只读路径不得创建扩展库"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P7 融合/拓扑/段落 {len(tests)} 项回归全部通过")
