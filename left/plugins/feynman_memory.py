"""会话量子记忆：把每轮坍缩出的语体方向留成一个带阻尼的复振幅。

只记"上一轮把话说到哪个语体/主题方向"，不记原话，也不当事实库用。
演化规则（与 PYBOT2 方案同构，但去掉不可执行的装饰）：

- 同向（cos Δθ > 0）才保留旧振幅，强度乘 0.85·cos Δθ；
- 新方向作为单位相量加上去（相长叠加），振幅自然增长；
- |振幅| < 0.15 视为退相干，清零；
- |振幅| 上限 2.0，避免会话越聊越"固执"。

`biased_style` 把记忆方向以最多 ±15° 的力度拉向旋钮，给出下一轮的排版相位。
没有会话 id 时调用方根本不建 SessionMemory，行为与无记忆完全一致。
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

DECAY = 0.85
PRUNE_BELOW = 0.15
AMP_CEIL = 2.0
MAX_BIAS_DEG = 15.0
BIAS_GAIN = 0.2


@dataclass
class SessionMemory:
    amplitude: complex = 0j
    phase_deg: float = 60.0
    turns: int = 0
    pruned: bool = False
    last_entities: List[str] = field(default_factory=list)

    def _clamp(self) -> None:
        mag = abs(self.amplitude)
        if mag > AMP_CEIL:
            self.amplitude *= AMP_CEIL / mag
        if mag < PRUNE_BELOW:
            self.amplitude = 0j
            self.pruned = True

    def update(self, phase_deg: float, entities: Tuple[str, ...] = ()) -> None:
        """吸收一轮坍缩：返回后 amplitude/phase_deg 反映新状态。"""
        self.turns += 1
        mag = abs(self.amplitude)
        if mag == 0:
            self.amplitude = complex(math.cos(math.radians(phase_deg)),
                                     math.sin(math.radians(phase_deg))) * 1.0
            self.phase_deg = phase_deg
            self.pruned = False
        else:
            dtheta = abs(phase_deg - self.phase_deg) % 360.0
            dtheta = min(dtheta, 360.0 - dtheta)
            survive = DECAY * max(0.0, math.cos(math.radians(dtheta)))
            carried = mag * survive
            # 旧方向被这一轮抵消到阈值以下：旧线索退相干，只留下新方向
            self.pruned = carried < PRUNE_BELOW
            self.amplitude *= survive
            self.amplitude += complex(math.cos(math.radians(phase_deg)),
                                      math.sin(math.radians(phase_deg)))
            self.phase_deg = math.degrees(math.atan2(self.amplitude.imag,
                                                     self.amplitude.real)) % 360.0
        self.last_entities = [e for e in entities][:3]
        self._clamp()

    def magnitude(self) -> float:
        return round(abs(self.amplitude), 4)

    def biased_style(self, style_deg: float) -> float:
        """记忆只做轻推：强度 min(|A|,1)，力度 ≤15°，不越出 [0,120]。"""
        mag = abs(self.amplitude)
        if mag < PRUNE_BELOW:
            return style_deg
        diff = self.phase_deg - style_deg
        bias = max(-MAX_BIAS_DEG, min(MAX_BIAS_DEG, BIAS_GAIN * min(mag, 1.0) * diff))
        return max(0.0, min(120.0, style_deg + bias))

    def snapshot(self) -> dict:
        return {
            "turns": self.turns,
            "amplitude": self.magnitude(),
            "phase_deg": round(self.phase_deg, 1),
            "pruned": self.pruned,
            "last_entities": list(self.last_entities),
        }
