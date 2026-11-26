"""Fact-skepticism reflex: the persona's first move on a grand claim.

The hook is deliberately narrow. It fires only on recognizable "someone has
proved X" patterns, not on ordinary questions about X, and it answers with the
evidence questions Feynman would ask: where is the paper, what is the explicit
construction, has anyone reproduced it, and does it hold in the general case?

The 2.25 rule exists because the matrix-multiplication exponent is a live
research subject, not a settled constant. The reply must not repeat the rumour
as fact: it corrects it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class SkepticCheck:
    rule: str
    reply: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "reply": self.reply}


_OMEGA_ZH = re.compile(
    r"(矩阵乘法|矩阵乘法的指数|矩阵乘法复杂度).{0,40}(2\.25|二点二五)"
    r"|(2\.25|二点二五).{0,40}(矩阵乘法|矩阵乘法的指数)"
    r"|(openai|OpenAI).{0,30}(证明|搞出|解决).{0,30}(矩阵乘法|矩阵乘法的指数)",
    re.IGNORECASE,
)
_OMEGA_EN = re.compile(
    r"matrix\s+multiplication.{0,80}(2\.25)"
    r"|(2\.25).{0,80}matrix\s+multiplication"
    r"|(openai|researchers?|someone|they).{0,40}(proved|proves|solved|showed|proven)"
    r".{0,60}matrix\s+multiplication",
    re.IGNORECASE,
)

_PNP = re.compile(
    r"P\s*(?:=|equals)\s*NP.{0,40}(证明|解决|proved|solved|settled)"
    r"|(证明|解决|proved|solved|settled).{0,40}P\s*(?:=|equals)\s*NP",
    re.IGNORECASE,
)
_PERPETUAL = re.compile(
    r"(永动机|perpetual\s+motion).{0,40}(造出|实现|成功|works|built|achieved|confirmed)",
    re.IGNORECASE,
)
_SUPERCONDUCTOR = re.compile(
    r"(常温超导|室温超导|room[-\s]?temperature\s+superconduct).{0,50}"
    r"(实现|突破|证明|成功|achieved|breakthrough|proved|confirmed)",
    re.IGNORECASE,
)

_OMEGA_REPLY = {
    "zh": (
        "等一下，这个说法不成立。就我所知，OpenAI 并没有证明“任意域上矩阵乘法的指数"
        "精确等于 2.25”。公开的最好上界在 2.37 附近（具体数字以最新文献为准），精确值仍是开放问题；"
        "2.25 在这个程序里只是我自己的一个动力学设计常数，不是定理。"
        "先别急着信——把 arXiv 论文、具体的张量分解构造、以及别人能复现的检验拿出来。"
        "没有这三样，它就只是个好听的故事。"
    ),
    "en": (
        "Hold on. That claim does not hold up. As far as I know, OpenAI has not proved "
        "that the matrix-multiplication exponent over arbitrary fields is exactly 2.25. "
        "The published upper bounds I know are around 2.37 (check the latest literature for the exact number), and the exact value is still "
        "open; 2.25 is only a dynamics constant I chose for this program, not a theorem. "
        "Before believing it, show me the arXiv paper, the explicit tensor decomposition, "
        "and a check someone else can reproduce. Without those three, it is just a good story."
    ),
}

_PNP_REPLY = {
    "zh": (
        "先别把标题当证明。P 是否等于 NP 至今没有被解决；谁要说证明了，"
        "就让他给出形式化证明、让同行逐行检查，而不是一段采访或一张图。"
        "这地方最容易出现“看起来像证明”的东西。"
    ),
    "en": (
        "Don't mistake a headline for a proof. Whether P equals NP is still open. "
        "Anyone claiming a proof should hand over a formal argument that peers can "
        "check line by line, not an interview or a diagram. This is exactly the kind "
        "of claim that produces things that merely look like proofs."
    ),
}

_PERPETUAL_REPLY = {
    "zh": (
        "永动机要成立，先得说清楚能量从哪儿来、又到哪儿去。"
        "把输入、输出和每一处损耗都列出来；如果账对不上，那不是突破，是漏记了一项。"
    ),
    "en": (
        "For a perpetual-motion machine to work, first say where the energy comes from "
        "and where it goes. List the input, the output, and every loss. If the books "
        "don't balance, that is not a breakthrough; it is a missing line item."
    ),
}

_SUPERCONDUCTOR_REPLY = {
    "zh": (
        "常温超导这种消息，先看三件事：样品、测量曲线、独立复现。"
        "没有别人按同样步骤做出来，它就还只是一个待核实的声称，不是事实。"
    ),
    "en": (
        "For a room-temperature superconductor claim, check three things first: the "
        "sample, the measurement curves, and independent replication. Until someone "
        "else gets the same result by the same steps, it is an unverified claim, not a fact."
    ),
}

_RULES = {
    "omega_2_25": (
        {"zh": _OMEGA_ZH, "en": _OMEGA_EN},
        _OMEGA_REPLY,
    ),
    "p_vs_np": (
        {"zh": _PNP, "en": _PNP},
        _PNP_REPLY,
    ),
    "perpetual_motion": (
        {"zh": _PERPETUAL, "en": _PERPETUAL},
        _PERPETUAL_REPLY,
    ),
    "room_temperature_superconductor": (
        {"zh": _SUPERCONDUCTOR, "en": _SUPERCONDUCTOR},
        _SUPERCONDUCTOR_REPLY,
    ),
}


def check(text: str, language: str = "zh") -> Optional[SkepticCheck]:
    if not text:
        return None
    for rule, (patterns, replies) in _RULES.items():
        pattern = patterns.get(language) or patterns["zh"]
        if pattern.search(text):
            return SkepticCheck(rule=rule, reply=replies.get(language, replies["zh"]))
    return None


# Amount a correction injects into the affect state and the resonance loop.
DOPAMINE_JOLT = 0.25
CORTISOL_JOLT = 0.30
RESONANCE_JOLT = 2.25
