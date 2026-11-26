"""适应性改造回归：相对排序补救、确定性归并、离线诚实降级。"""
import math
import os

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["QBOT_CACHE_TTL_SEC"] = "0"

from fastapi.testclient import TestClient

from bus.main import app, rescue_by_relative_rank, _boundary_underspecified
from evaluator.mapreduce_auditor import SingleMapAudit, _cluster_by_claim
from shared.protocol import AmplitudePacket

client = TestClient(app)


def _packet(claim, phase_deg, bot="PyBot-F1", magnitude=1.0):
    """按 θ = sign·π·loss^1.2 反推实部/虚部。"""
    theta = math.radians(phase_deg)
    return AmplitudePacket(
        bot_id=bot, wave_direction="RETARDED_FORWARD", perspective="t",
        conclusion_key=claim, detail="", critical_flaw="",
        real=round(math.cos(theta) * magnitude, 4),
        imag=round(math.sin(theta) * magnitude, 4),
        magnitude=magnitude, phase_rad=theta,
        phase_deg=phase_deg, interference_regime="",
    )


def _audit(claim):
    return SingleMapAudit(
        claim_summary=claim, critical_flaw="", evidence_strength=0.9,
        assumption_burden=0.1, boundary_fragility=0.1, logical_leap=0.1,
    )


def test_rescue_ranks_by_smallest_relative_loss():
    packets = [
        _packet("几乎全反相", 175.0, "PyBot-F1"),
        _packet("中等反相", 120.0, "PyBot-F2"),
        _packet("最接近自洽", 95.0, "PyBot-B1"),
    ]
    spectrum = rescue_by_relative_rank(packets)
    assert spectrum[0]["conclusion"] == "最接近自洽"
    assert spectrum[0]["rescued"] is True
    assert abs(sum(s["probability"] for s in spectrum) - 100.0) < 0.5


def test_rescue_prefers_stronger_evidence_on_tie():
    a = _packet("弱证据", 100.0, "PyBot-F1", magnitude=0.5)
    b = _packet("强证据", 100.0, "PyBot-B1", magnitude=1.0)
    assert rescue_by_relative_rank([a, b])[0]["conclusion"] == "强证据"


def test_deterministic_cluster_merges_synonyms():
    audits = [_audit("同时落地"), _audit("同时落地。"), _audit("两弹同时落地")]
    out = _cluster_by_claim(audits)
    assert len(set(out)) == 1, out


def test_deterministic_cluster_keeps_negation_flips_apart():
    audits = [_audit("读数不变"), _audit("读数改变")]
    out = _cluster_by_claim(audits)
    assert out[0] != out[1], out


def test_deterministic_cluster_keeps_opposite_directions_apart():
    audits = [_audit("熵增不可逆"), _audit("熵减可逆")]
    out = _cluster_by_claim(audits)
    assert out[0] != out[1], out


def test_boundary_underspecified_detection():
    assert _boundary_underspecified("一架无人机在卡车里飞起来，卡车会变轻吗？")
    assert _boundary_underspecified("水箱里的鱼悬停时秤读数会变吗？")
    assert not _boundary_underspecified("一辆完全密封的厢式货车停在电子地磅上，车内无人机悬停，读数如何变化？")
    assert not _boundary_underspecified("敞篷货车里的无人机悬停，地磅读数如何变化？")


def test_offline_uncovered_problem_abstains_instead_of_fabricating():
    unseen = "请推演 P 与 NP 是否相等，并给出复杂度类之间的严格分离证明。"
    data = client.post("/solve", json={"problem": unseen}).json()
    assert data["collapse_status"] == "SECOND_NOTEBOOK_PENDING"
    assert data["packets"] == [] and data["spectrum"] == []
    assert data["second_notebook"]["status"] == "OFFLINE_UNCOVERED"
    assert data["second_notebook"]["clarifying_questions"]


def test_offline_covered_problem_still_answers():
    data = client.post(
        "/solve",
        json={"problem": "一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？"},
    ).json()
    assert data["collapse_status"] == "COLLAPSED_VALID"
    assert data["spectrum"][0]["probability"] >= 60.0


if __name__ == "__main__":
    for fn in [
        test_rescue_ranks_by_smallest_relative_loss,
        test_rescue_prefers_stronger_evidence_on_tie,
        test_deterministic_cluster_merges_synonyms,
        test_deterministic_cluster_keeps_negation_flips_apart,
        test_deterministic_cluster_keeps_opposite_directions_apart,
        test_boundary_underspecified_detection,
        test_offline_uncovered_problem_abstains_instead_of_fabricating,
        test_offline_covered_problem_still_answers,
    ]:
        fn()
    print("✅ 适应性改造 8 项回归全部通过")
