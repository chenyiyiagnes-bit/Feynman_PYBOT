"""P9 回归：算子分数连续化、Boltzmann 温度、拓扑证据分级与池内归一。

背景（2026-11-25 实测 p8b）：
- Boltzmann 在预设权重下 top-8 顺序完全不变，因为同一实体/动作的路径算子
  分数逐位相同，exp(−E) 退化成常数；
- 拓扑乘子在真实候选上 100% 来自测地曲率（~0.3），三角形/空洞因子从不触发，
  等于给全池乘一个常数，只在概率第 4 位小数上撬动近并列——是噪声不是证据。

这里守住三件事：
1. 分数对"长短/共现强度"连续，且仍在 [0,1]；
2. Boltzmann 有温度、乘数 ≤1（E≥0），零权重仍恰好 1.0；
3. 拓扑证据分级存在，池内按中位数归一，均匀项被约掉。
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

TMP = Path(tempfile.mkdtemp())
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_FEYNMAN_EXT"] = str(TMP / "domain_extensions.json")

from plugins.feynman_typesetter import (  # noqa: E402
    OperatorWeights,
    PrintRequest,
    Vault,
    _normalize_topology_pool,
    _operator_multiplier,
    _operator_scores,
)

FAKE = {
    "corpus": "(synthetic)", "sections": 1, "sentences": 60,
    "slot_offsets": {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0},
    "skeletons": [
        {"pattern_type": "declarative", "template": "{hook}，{entity}{action}，{consequence}。",
         "phase_deg": 55.0, "freq": 40, "provenance": ["样例句"]},
    ],
    "hooks": [
        {"word": "你看", "slot": "hook", "count": 20, "amp": 1.0, "domain": "everyday",
         "style": "speech", "phase_deg": 60.0, "sources": ["样｜1"], "example": ""},
    ],
    "entities": [
        {"word": "电子", "slot": "entity", "count": 50, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 20.0, "sources": ["样｜1"], "example": ""},
    ],
    "actions": [
        {"word": "观察", "slot": "action", "count": 22, "amp": 1.0, "domain": "physics",
         "style": "lecture", "phase_deg": 30.0, "sources": ["样｜1"], "example": "",
         "order": "pre", "valency": "vo"},
    ],
    "consequences": [
        {"word": "温度升高", "slot": "consequence", "count": 12, "amp": 1.0,
         "domain": "physics", "style": "lecture", "phase_deg": 40.0,
         "sources": ["样｜1"], "example": ""},
        {"word": "温度升高然后一直持续下去", "slot": "consequence", "count": 3, "amp": 1.0,
         "domain": "physics", "style": "lecture", "phase_deg": 40.0,
         "sources": ["样｜2"], "example": ""},
    ],
    "entangled_pairs": [],
    "cooccurrence": {
        "电子|观察": 6, "观察|温度升高": 5, "电子|温度升高": 5,
    },
}
FAKE_PATH = TMP / "feynman_style_vault.json"
FAKE_PATH.write_text(json.dumps(FAKE, ensure_ascii=False), encoding="utf-8")
VAULT = Vault(FAKE_PATH)


def _path(consequence: str) -> dict:
    g = lambda slot, word: next(x for x in VAULT.by_slot[slot] if x["word"] == word)
    return {"hook": g("hook", "你看"), "entity": g("entity", "电子"),
            "action": g("action", "观察"), "consequence": g("consequence", consequence)}


def test_scores_stay_bounded_and_continuous():
    short = _operator_scores(_path("温度升高"), VAULT, set())
    long = _operator_scores(_path("温度升高然后一直持续下去"), VAULT, set())
    for scores in (short, long):
        assert all(0.0 <= v <= 1.0 for v in scores.values()), scores
    # 最小作用量：话越长越不像费曼，且不再只有"≤6 字"一档
    assert short["least_action"] > long["least_action"]
    # 反直觉：共现越多越陈词，但不再只有 0.2/0.6/0.75/1.0 四个点
    assert short["anti_intuition"] < long["anti_intuition"]
    assert 0.0 < long["least_action"] < 1.0


def test_boltzmann_temperature_and_cap():
    low = {k: 0.0 for k in OperatorWeights().model_dump()}
    w = OperatorWeights(**{k: 0.125 for k in low})  # E = 8×0.125 = 1.0
    warm = _operator_multiplier(low, w, fusion="boltzmann", tau=0.5)
    cool = _operator_multiplier(low, w, fusion="boltzmann", tau=2.0)
    # 无 0.25 下限：绝对能量概率归一后消失，下限只会把整池截成常数
    assert 0.0 < warm < cool <= 1.0, (warm, cool)
    # E≥0 ⇒ 乘数不可能 >1（提案里的 4.0 上限不可达，已删）
    high = {k: 1.0 for k in low}
    assert _operator_multiplier(high, w, fusion="boltzmann") == 1.0
    # 零权重仍然精确等于 1.0（基线不变）
    assert _operator_multiplier(low, OperatorWeights(), fusion="boltzmann") == 1.0


def test_boltzmann_discriminates_real_alternatives():
    w = OperatorWeights(**{k: 0.15 for k in OperatorWeights().model_dump()})
    # 权重太大时两条路径都会被 0.25 下限截平，测不出区分度；取小权重
    # 让 exp 项落在未截断区间。
    a = _operator_scores(_path("温度升高"), VAULT, set())
    b = _operator_scores(_path("温度升高然后一直持续下去"), VAULT, set())
    ma = _operator_multiplier(a, w, fusion="boltzmann")
    mb = _operator_multiplier(b, w, fusion="boltzmann")
    assert abs(ma - mb) > 1e-3, "同实体同动作、不同后果的路径必须能被区分"


def test_use_topology_request_field():
    assert PrintRequest().use_topology is False
    assert PrintRequest(use_topology=True).use_topology is True


def test_topology_pool_normalization_removes_uniform_decay():
    # 真实流程里 weight 已经乘过原始拓扑乘子，归一换掉的是 raw→norm 这一段
    paths = [{"weight": m, "topology": {"multiplier": m}}
             for m in (0.30, 0.31, 0.32, 0.33, 0.60)]
    _normalize_topology_pool(paths)
    norms = [p["topology"]["multiplier"] for p in paths]
    assert abs(float(np.median(norms)) - 1.0) < 0.05, norms
    assert all(0.5 <= n <= 2.0 for n in norms), norms
    # 原始值留档，排序不变
    assert paths[0]["topology"]["multiplier_raw"] == 0.30
    assert norms[0] < norms[-1]
    # 权重乘的是归一后的比值：最终权重与乘子同序
    assert paths[0]["weight"] < paths[-1]["weight"]


def test_hole_requires_strong_pair_support():
    from plugins.topology_builder import TopologyManifoldEngine
    words = ["甲", "乙", "丙", "丁"]
    vecs = np.eye(4, dtype=np.float64)
    edges = {("甲", "乙"), ("甲", "丙"), ("乙", "丙")}
    strong = TopologyManifoldEngine(
        words, vecs, {}, edges, None, 0.0, 0, "test0001",
        edge_counts={("甲", "乙"): 3, ("甲", "丙"): 3, ("乙", "丙"): 3})
    weak = TopologyManifoldEngine(
        words, vecs, {}, edges, None, 0.0, 0, "test0002",
        edge_counts={("甲", "乙"): 1, ("甲", "丙"): 1, ("乙", "丙"): 1})
    ms, _ = strong.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    mw, _ = weak.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    assert ms < mw, "三对边都很强却从未同处才叫空洞；弱证据不该被罚"


def test_topology_window_tier_boosts_above_neutral():
    from plugins.topology_builder import TopologyManifoldEngine
    words = ["甲", "乙", "丙", "丁"]
    vecs = np.eye(4, dtype=np.float64)
    base = TopologyManifoldEngine(words, vecs, {}, set(), None, 0.0, 0, "t1")
    tier3 = TopologyManifoldEngine(
        words, vecs, {("甲", "乙", "丙"): 1}, set(), None, 0.0, 0, "t2",
        simplex_tiers={("甲", "乙", "丙"): 3})
    m0, is0 = base.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    m3, is3 = tier3.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    assert is0 is False and is3 is True
    assert m3 > m0, "三句窗内的三元组是弱证据，应当高于中性"


def test_topology_section_tier_beats_neutral_and_collects():
    """同段共现：三元组分散在同一节的不同句子里，三句窗抓不到，段落层抓到。

    实测 p8b 语料：同句 31、两句窗 103、三句窗 168、段落 2440；
    候选路径 480 条里段落层命中 51 条（覆盖 42/48 查询），前三级为 0。
    """
    from plugins.topology_builder import (
        TopologyManifoldEngine,
        collect_section_triples,
        collect_tiered_triples,
    )
    vocab = {slot: [{"word": w, "slot": slot, "count": 1, "amp": 1.0,
                     "domain": "physics", "style": "lecture", "phase_deg": 0.0,
                     "sources": ["样｜1"], "example": ""}]
             for slot, w in (("entity", "甲"), ("action", "乙"), ("consequence", "丙"))}
    vocab.update({"hooks": [], "entities": vocab["entity"], "actions": vocab["action"],
                  "consequences": vocab["consequence"], "skeletons": [],
                  "entangled_pairs": [], "cooccurrence": {}, "corpus": "(synthetic)"})
    text = ("甲是第一个。中间一。中间二。中间三。中间四。"
            "乙是第五个。中间五。中间六。中间七。丙是第八个。")
    corp = TMP / "section_tier_corpus.jsonl"
    corp.write_text(json.dumps({"source": "费恩曼物理学讲义.md｜1-1", "section": "s",
                               "text": text}, ensure_ascii=False) + "\n",
                    encoding="utf-8")

    tiers = collect_tiered_triples(vocab, corp)
    assert all(("甲", "乙", "丙") not in tiers[t] for t in (1, 2, 3)), \
        "三句窗内不该抓到跨越七句的共现"
    sections = collect_section_triples(vocab, corp)
    assert ("甲", "乙", "丙") in sections

    words = ["甲", "乙", "丙", "丁"]
    vecs = np.eye(4, dtype=np.float64)
    base = TopologyManifoldEngine(words, vecs, {}, set(), None, 0.0, 0, "s0")
    sec = TopologyManifoldEngine(words, vecs, {}, set(), None, 0.0, 0, "s1",
                                 section_simplexes={("甲", "乙", "丙")})
    tier3 = TopologyManifoldEngine(words, vecs, {("甲", "乙", "丙"): 1}, set(),
                                   None, 0.0, 0, "s2",
                                   simplex_tiers={("甲", "乙", "丙"): 3})
    m0, is0 = base.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    m1, is1 = sec.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    m3, _ = tier3.compute_geodesic_multiplier("丁", "甲", "乙", "丙")
    assert is0 is False and is1 is True
    assert m1 > m0, "段落共现是弱证据，但必须高于中性"
    assert m1 <= m3, "段落证据不得强于三句窗"


def test_pair_evidence_rewards_strong_pairs_only():
    from plugins.topology_builder import TopologyManifoldEngine
    words = ["甲", "乙", "丙"]
    vecs = np.eye(3, dtype=np.float64)
    # 两条边强、一条弱（<HOLE_MIN_SUPPORT）：走成对共现加成，不算空洞
    strong = TopologyManifoldEngine(
        words, vecs, {}, set(), None, 0.0, 0, "p1",
        edge_counts={("甲", "乙"): 60, ("甲", "丙"): 60, ("乙", "丙"): 1})
    weak = TopologyManifoldEngine(
        words, vecs, {}, set(), None, 0.0, 0, "p2",
        edge_counts={("甲", "乙"): 1})
    none = TopologyManifoldEngine(words, vecs, {}, set(), None, 0.0, 0, "p3")
    se = strong.pair_evidence("甲", "乙", "丙")
    we = weak.pair_evidence("甲", "乙", "丙")
    ne = none.pair_evidence("甲", "乙", "丙")
    assert se["score"] > 0.5 and se["tilt"] > 1.0
    # 弱证据/无证据都不许扣分：没有证据不等于反证据
    assert we["tilt"] == 1.0 and ne["tilt"] == 1.0
    ms, _ = strong.compute_geodesic_multiplier("甲", "乙", "丙", "甲")
    mw, _ = weak.compute_geodesic_multiplier("甲", "乙", "丙", "甲")
    assert ms > mw, "强成对共现必须比弱证据拿到更高乘子"
    # 三对边全强却无同处证据：仍是空洞，应被压低（压过 +10% 加成）
    hole = TopologyManifoldEngine(
        words, vecs, {}, set(), None, 0.0, 0, "p4",
        edge_counts={("甲", "乙"): 60, ("甲", "丙"): 60, ("乙", "丙"): 60})
    mh, _ = hole.compute_geodesic_multiplier("甲", "乙", "丙", "甲")
    assert mh < ms, "强边空洞必须低于部分支撑的可组合路径"


def test_double_penalty_arbitration_keeps_milder():
    from plugins.feynman_typesetter import _arbitrate_double_penalty
    both_penalized = {"weight": round(1.0 * 0.8 * 0.6, 6),
                      "conditioned_multiplier": 0.6,
                      "topology": {"multiplier": 0.8}}
    rewarded = {"weight": round(1.2 * 0.6, 6),
                "conditioned_multiplier": 0.6, "topology": {"multiplier": 1.2}}
    paths = [both_penalized, rewarded]
    assert _arbitrate_double_penalty(paths) == 1
    # 权重退还较重那条：只剩 0.8 的惩罚
    assert abs(both_penalized["weight"] - 0.8) < 1e-6
    assert both_penalized["topology"]["double_penalty_arbitrated"] is True
    # 奖励照常叠加，不动
    assert abs(rewarded["weight"] - 0.72) < 1e-6
    assert "double_penalty_arbitrated" not in rewarded["topology"]


def test_topology_health_endpoint_fields():
    from fastapi.testclient import TestClient
    from bus.main import app
    body = TestClient(app).get("/feynman/topology").json()
    assert set(("enabled", "loaded", "health")).issubset(body), body


def test_frontend_has_topology_toggle():
    src = Path("frontend/dashboard.py").read_text(encoding="utf-8")
    assert "拓扑层" in src
    assert "use_topology" in src
    assert '"use_topology": feynman_topology' in src


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ P9 连续分数/温度/拓扑分级与归一 {len(tests)} 项回归全部通过")
