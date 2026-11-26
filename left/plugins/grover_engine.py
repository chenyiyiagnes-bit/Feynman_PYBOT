import asyncio
import cmath
import math
from typing import Dict
from pydantic import BaseModel


class GroverRequest(BaseModel):
    target_state: str = "|10>"
    step: int = 2


async def _pybot_oracle_check(basis_state: str, current_amp: complex, target: str) -> complex:
    await asyncio.sleep(0.03)
    if basis_state == target:
        return current_amp * cmath.rect(1.0, math.pi)
    return current_amp


async def run_grover_simulation(req: GroverRequest) -> Dict:
    states = ["|00>", "|01>", "|10>", "|11>"]
    bot_names = ["PyBot-00", "PyBot-01", "PyBot-10", "PyBot-11"]
    amplitudes = {s: complex(0.5, 0.0) for s in states}
    stage_desc = "Step 0: Hadamard 门初始化 —— 4 个 PyBot 处于均匀叠加态 (各占 25.0%)"

    if req.step >= 1:
        flipped_amps = await asyncio.gather(*(_pybot_oracle_check(s, amplitudes[s], req.target_state) for s in states))
        amplitudes = dict(zip(states, flipped_amps))
        stage_desc = f"Step 1: Oracle 相位标记 —— {req.target_state} 节点触发 180° 反相翻转 (振幅变为 -0.50)"

    if req.step >= 2:
        mean_amp = sum(amplitudes.values()) / len(amplitudes)
        for s in states:
            amplitudes[s] = 2 * mean_amp - amplitudes[s]
        stage_desc = f"Step 2: 扩散算子干涉放大 —— 非目标态相消归零 (0%)，目标态 {req.target_state} 振幅跃升至 +1.0 (100%)"

    packets, spectrum = [], []
    for b_id, s in zip(bot_names, states):
        amp = amplitudes[s]
        mag = abs(amp)
        prob = round((mag ** 2) * 100, 2)
        phase_deg = round(math.degrees(cmath.phase(amp)), 1) if mag > 1e-6 else 0.0
        is_target = (s == req.target_state)

        packets.append({
            "bot_id": b_id,
            "perspective": "🎯 目标基矢节点" if is_target else "普通叠加态节点",
            "conclusion_key": f"量子态 {s}",
            "real": round(amp.real, 4),
            "imag": round(amp.imag, 4),
            "magnitude": round(mag, 4),
            "phase_deg": phase_deg,
            "interference_regime": "相长共鸣 (100%锁定)" if prob > 90 else ("180°反相标记" if amp.real < -0.1 else ("相消归零" if prob == 0 else "均匀叠加"))
        })
        spectrum.append({
            "conclusion": f"量子态 {s}" + (" [目标]" if is_target else ""),
            "probability": prob,
            "net_real": round(amp.real, 4),
            "net_imag": round(amp.imag, 4),
            "net_magnitude": round(mag, 4)
        })

    return {
        "stage_desc": stage_desc,
        "target_state": req.target_state,
        "step": req.step,
        "packets": packets,
        "spectrum": spectrum
    }
