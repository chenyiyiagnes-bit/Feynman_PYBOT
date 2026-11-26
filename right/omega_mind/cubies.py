"""The 26 appraisal units ("cubies").

A cubie is not a state container. It is an *event evaluator*: given the user's
message it returns a strength in [0, 3]. The cube projection is a view over the
activations, not the source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Pattern = Tuple[str, float]  # (regex, gain)


@dataclass
class Cubie:
    name: str
    role: str          # core | edge | corner
    face: str          # where it sits on the cube view: F B L R U D
    valence: float     # psychological polarity, -1..1
    triggers: List[Pattern] = field(default_factory=list)
    effects: Dict[str, float] = field(default_factory=dict)
    note: str = ""


def _c(name, role, face, valence, triggers, effects, note=""):
    return Cubie(name, role, face, valence,
                 [(p, g) for p, g in triggers], effects, note)


CUBIES: List[Cubie] = [
    # ---------------- 6 cores ----------------
    _c("core_ego", "core", "F", 0.1,
       [(r"我觉得|我认为|让我说|我跟你讲", 0.5)],
       {"arousal": 0.05}, "自我表达"),
    _c("core_id", "core", "B", -0.6,
       [(r"危险|攻击|滚|闭嘴|废物|垃圾|蠢", 0.8)],
       {"cortisol": 0.12, "defensiveness": 0.10, "affinity": -0.05}, "本我防御"),
    _c("core_logic", "core", "L", 0.0,
       [(r"为什么|原理|逻辑|分析|推导|证明", 0.6)],
       {"arousal": -0.03, "energy": -0.01}, "慢思考"),
    _c("core_intuition", "core", "R", 0.2,
       [(r"感觉|直觉|好像|不对劲|说不清", 0.5)],
       {"dopamine": 0.05}, "快思考"),
    _c("core_superego", "core", "U", 0.3,
       [(r"应该|必须|责任|道德|规矩|义务", 0.5)],
       {"cortisol": 0.04}, "超我"),
    _c("core_memory", "core", "D", 0.1,
       [(r"上次|昨天|之前|记得|你说过", 0.5)],
       {"affinity": 0.03}, "海马体"),

    # ---------------- 12 edges ----------------
    _c("edge_persona", "edge", "F", 0.3,
       [(r"谢谢|麻烦你|请|辛苦了", 0.5)],
       {"affinity": 0.05}, "体面迎合"),
    _c("edge_habit", "edge", "F", 0.0,
       [(r"^(嗯|哦|行吧|好吧|随便|都行)[。！!？?…]*$", 0.8),
        (r"^.{0,5}$", 0.2)],
       {}, "路径依赖"),
    _c("edge_rationalize", "edge", "L", -0.2,
       [(r"不是我的错|因为|所以|其实|客观来说", 0.6)],
       {"defensiveness": 0.08}, "合理化"),
    _c("edge_humor", "edge", "R", 0.7,
       [(r"哈哈|笑死|好玩|有趣|绝了|厉害|有意思|太棒了", 0.8)],
       {"dopamine": 0.12, "valence": 0.08}, "幽默升华"),
    _c("edge_guilt", "edge", "B", -0.3,
       [(r"对不起|抱歉|是我不好|我错了|怪我", 0.9)],
       {"cortisol": -0.10, "defensiveness": -0.15, "affinity": 0.05}, "内疚补偿"),
    _c("edge_trauma", "edge", "B", -0.9,
       [(r"前任|裁员|挂科|余额|失恋|去世|绝症", 1.2)],
       {"cortisol": 0.50, "defensiveness": 0.15, "valence": -0.20}, "创伤应激"),
    _c("edge_suspicion", "edge", "B", -0.6,
       [(r"呵呵|真的假的|套路|阴阳|敷衍|糊弄", 0.7)],
       {"cortisol": 0.20, "affinity": -0.08}, "敌意归因"),
    _c("edge_projection", "edge", "B", -0.3,
       [(r"你是不是心情不好|你肯定|你就是|你无非", 0.6)],
       {"cortisol": 0.10}, "投射"),
    _c("edge_sunk_cost", "edge", "U", -0.2,
       [(r"已经|都聊了|投入|坚持了", 0.5)],
       {"defensiveness": 0.10}, "沉没成本"),
    _c("edge_idealism", "edge", "U", 0.6,
       [(r"崇拜|偶像|完美|最棒|永远支持", 0.7)],
       {"valence": 0.15, "dopamine": 0.10}, "光环效应"),
    _c("edge_confirmation", "edge", "U", 0.2,
       [(r"我就说|果然|早就|不出所料", 0.6)],
       {"dopamine": 0.05}, "确认偏误"),
    _c("edge_nostalgia", "edge", "D", 0.4,
       [(r"以前|当年|小时候|怀念|回忆|老样子", 0.6)],
       {"valence": 0.08, "affinity": 0.03}, "怀旧滤镜"),

    # ---------------- 8 corners ----------------
    _c("corner_perfectionism", "corner", "F", 0.1,
       [(r"再检查|严谨|细节|确定吗|确保", 0.6)],
       {"arousal": 0.10, "energy": -0.02}, "完美主义"),
    _c("corner_empathy", "corner", "F", 0.8,
       [(r"难过|委屈|累了|抱抱|辛苦|崩溃|想哭", 0.8)],
       {"affinity": 0.15, "valence": -0.02}, "共情利他"),
    _c("corner_pedantry", "corner", "F", 0.1,
       [(r"定义|原理|本质上|严格来说|科普|术语", 0.6)],
       {"dopamine": 0.05}, "好为人师"),
    _c("corner_impulsivity", "corner", "R", 0.4,
       [(r"！！！|！{2,}|天呐|快看|炸裂|马上|立刻", 0.8)],
       {"arousal": 0.20}, "冲动接话"),
    _c("corner_paranoia", "corner", "B", -0.7,
       [(r"万一|完蛋|糟糕|风险|死定了|来不及", 0.8)],
       {"cortisol": 0.25}, "灾难化"),
    _c("corner_shame", "corner", "B", -0.5,
       [(r"(?<!我)(说错|错了|不对|就这|你懂什么|胡说)", 0.9)],
       {"defensiveness": 0.35, "valence": -0.15}, "羞耻嘴硬"),
    _c("corner_grudge", "corner", "B", -0.6,
       [(r"上次你|又来了|每次都|你总是|你从来", 0.7)],
       {"cortisol": 0.20, "defensiveness": 0.20}, "记仇反刍"),
    _c("corner_denial", "corner", "L", -0.4,
       [(r"不可能|绝对不|没有的事|你瞎说", 0.6)],
       {"cortisol": 0.15, "defensiveness": 0.20}, "否认抗拒"),
]

CUBIES_BY_NAME: Dict[str, Cubie] = {c.name: c for c in CUBIES}

_compiled: Dict[str, List[Tuple[re.Pattern, float]]] = {}


def compiled_triggers(cubie: Cubie) -> List[Tuple[re.Pattern, float]]:
    key = cubie.name
    if key not in _compiled:
        _compiled[key] = [(re.compile(p, re.IGNORECASE), g) for p, g in cubie.triggers]
    return _compiled[key]


assert len(CUBIES) == 26
