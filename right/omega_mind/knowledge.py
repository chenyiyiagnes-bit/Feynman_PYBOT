"""A deliberately tiny built-in left brain.

This is the fallback used when PyBot is not mounted. It exists so the state
engine can be exercised on its own; it is not meant to be a knowledge base.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

_FACTS: List[Tuple[str, str]] = [
    (r"^\s*(你好|您好|嗨|哈喽|早上好|晚上好)[！!。.~～\s]*$",
     "我在。想聊点什么？"),
    (r"火|燃烧|木头",
     "火是太阳光在木头里存了好几年的能量：树把碳拆开存起来，点火只是让氧重新扑回去和碳抱在一起，把当年晒到的热量放出来。"),
    (r"橡皮筋|弹性",
     "橡皮筋变紧不是因为原子被拉长了，而是长长的分子链原来乱成一团；你一拉，它们被迫排队，一松手热运动又把它们撞回乱糟糟的样子。"),
    (r"光|光子|双缝|干涉",
     "光子像个不肯走直线的顽皮鬼：它到终点前把所有可能的路径都试了一遍，最后把路径的箭头加起来，才决定落在屏幕哪里。"),
    (r"磁|磁铁",
     "磁力不是隔空施法，更像带电粒子不停交换看不见的“虚光子”小球，扔球和接球把人推开或拉近。"),
    (r"重力|引力|苹果",
     "地球没伸绳子拽苹果，它只是把周围的时空床垫压出一个坑，苹果顺着坡度滑下去。"),
    (r"原子|物质|虚空",
     "把一杯水里的原子放大到房间那么大，原子核只有中央一粒微尘那么小，剩下的几乎全是空的；你没穿墙，靠的是电子之间的排斥。"),
    (r"时间|相对论|光速",
     "每个人在时空里穿行的总速度永远是光速。空间里跑得越快，分给时间流逝的那部分就越少，所以飞船里的钟走得慢。"),
    (r"熵|时间箭头|打碎",
     "墨水能散开、不能自己缩回去，不是定律禁止，而是“混乱”的排法比“整齐”多出天文数字，大自然只是走了概率最大的那条路。"),
    (r"电|电流|电子",
     "电线里的电子走得比蜗牛还慢，但你一按开关灯就亮，因为整条队伍同时被推了一下——像一列挤满人的管子，这头一推，那头立刻动。"),
    (r"热|温度|分子",
     "热不是什么神秘液体，只是数不清的原子在原地疯狂抖动、互相碰撞；温度越高，它们跳得越疯。"),
]

_FALLBACK = "嗯……这个问题我得再翻翻记忆库。"


_NAME = re.compile(r"^(?:你好[，,]?\s*)?我叫([一-鿿A-Za-z·]{2,8})[。.!！]*$")
_EN_NAME = re.compile(r"^(?:hi[,.!]?\s*)?(?:my name is|i am|i'm|call me)\s+([A-Za-z][A-Za-z .'-]{0,30}?)[.!]?$",
                      re.IGNORECASE)
_EN_GREETING = re.compile(r"^(?:hi|hello|hey|good morning|good evening)[!.\s]*$",
                          re.IGNORECASE)


def answer_en(text: str) -> Tuple[str, str]:
    """English builtin: greeting, name, and an explicit fallback."""
    stripped = text.strip()
    match = _EN_NAME.match(stripped)
    if match:
        return f"Got it, {match.group(1).strip()}. What would you like to talk about?", "builtin_en"
    if _EN_GREETING.match(stripped):
        return "I'm here. What's on your mind?", "builtin_en"
    return "I'm not confident about that one.", "fallback_en"


def answer(text: str) -> Tuple[str, str]:
    """Return (reply, source)."""
    match = _NAME.match(text.strip())
    if match:
        return f"记住了，{match.group(1)}。想聊点什么？", "builtin_kb"
    for pattern, reply in _FACTS:
        if re.search(pattern, text):
            return reply, "builtin_kb"
    return _FALLBACK, "fallback"
