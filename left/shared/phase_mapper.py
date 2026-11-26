import math
import cmath
import hashlib
from shared.protocol import WorkerPathEvaluation


def _get_decoherence_sign(seed_text: str) -> int:
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return 1 if int(digest, 16) % 2 == 0 else -1


def compute_symmetric_amplitude(
    eval_data: WorkerPathEvaluation,
    weights: tuple[float, float, float] = (0.3, 0.5, 0.2),
    gamma: float = 1.2
) -> dict:
    """将多维逻辑作用量损耗映射为 [-pi, pi] 对称随机退相干复数振幅"""
    w1, w2, w3 = weights
    raw_loss = (
        w1 * eval_data.assumption_burden +
        w2 * eval_data.boundary_fragility +
        w3 * eval_data.logical_leap
    )
    clamped_loss = max(0.0, min(1.0, raw_loss))
    base_theta = math.pi * (clamped_loss ** gamma)

    sign = _get_decoherence_sign(f"{eval_data.bot_id}:{eval_data.detail}")
    theta = sign * base_theta

    r = eval_data.evidence_strength
    amp = cmath.rect(r, theta)
    abs_theta = abs(theta)

    if abs_theta < math.pi / 6:
        regime = "相长共鸣区 (高度自洽)"
    elif abs_theta < math.pi / 2:
        regime = "温和增益区 (局部近似)"
    elif math.isclose(abs_theta, math.pi / 2, abs_tol=0.1):
        regime = "正交叠加态 (待定存疑)"
    elif abs_theta < 5 * math.pi / 6:
        regime = "部分相消区 (存在漏洞)"
    else:
        regime = "全额反相区 (致命悖论)"

    return {
        "real": amp.real,
        "imag": amp.imag,
        "magnitude": r,
        "phase_rad": round(theta, 4),
        "phase_deg": round(math.degrees(theta), 1),
        "interference_regime": regime
    }
