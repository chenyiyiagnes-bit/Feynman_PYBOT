"""P20 五条对话优化：全部默认关闭，开启时只加字段，不改段落正文。

五条各做一件小事，且都只用"固定模板 + 字库既有字粒"，不生成新的整句：
1. 苏格拉底反问：拿锚点里的实体/动作填一个固定问句；
2. 比喻画像：把三句的四槽字粒与相位画成一张字卡；
3. 自动相位：按本段实际相位给出下一轮建议语体档，只建议不生效；
4. 轮间呼吸：按轮次给一句固定停顿提示与停顿秒数；
5. 人机共创：只列出可学的候选边，真正写入必须走 P19 那道双闸通道。

关闭时 `apply_dialogue_modes` 返回空 dict，调用方不应添加任何键，
基线响应因此逐字节不变。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

BREATH_LINES = (
    "先停一拍：这段话里，哪一个词最不像物理？",
    "先喝口水：如果只能留下一个词，你留哪个？",
    "别急着往下问：刚才哪一句是你本来没想到的？",
)
BREATH_SECONDS = (3, 5, 8)


class DialogueModes(BaseModel):
    """五条开关，全默认关闭。"""
    socratic: bool = False
    portrait: bool = False
    auto_phase: bool = False
    breathing: bool = False
    cocreate: bool = False

    def any_on(self) -> bool:
        return any(self.model_dump().values())


def _pick(seq, nonce: int):
    return seq[int(nonce) % len(seq)] if seq else None


def _anchor_glyphs(result: dict) -> Dict[str, str]:
    anchor = result.get("anchor") or {}
    glyphs = anchor.get("glyphs")
    if isinstance(glyphs, dict) and glyphs:
        return {k: str(v) for k, v in glyphs.items() if v}
    for s in result.get("sentences") or []:
        g = s.get("glyphs")
        if isinstance(g, dict) and g.get("entity") and g.get("action"):
            return {k: str(v) for k, v in g.items() if v}
    return {}


def socratic_question(result: dict, nonce: int = 0) -> Optional[str]:
    """固定句架，只填空槽；实体或动作缺一个就不出。"""
    g = _anchor_glyphs(result)
    ent, act = g.get("entity"), g.get("action")
    if not ent or not act:
        return None
    templates = (
        "如果{ent}根本没有{act}，你会看到什么不一样？",
        "假设你是{ent}，{act}的时候你最先感觉到什么？",
        "{act}这件事，少了{ent}还成立吗？为什么？",
    )
    return _pick(templates, nonce).format(ent=ent, act=act)


def metaphor_portrait(result: dict, vault=None) -> Optional[str]:
    """四槽 + 相位画成字卡；只排版既有字粒，不写新内容。"""
    sentences = [s for s in (result.get("sentences") or [])
                 if isinstance(s.get("glyphs"), dict) and s["glyphs"]]
    if not sentences:
        return None
    rows: List[str] = ["🖼 比喻画像"]
    for s in sentences:
        g = s["glyphs"]
        phase = s.get("phase_mean")
        head = f"阶段{s.get('stage', 0) + 1}"
        if phase is not None:
            head += f" · {float(phase):.0f}°"
        rows.append(head)
        for slot in ("hook", "entity", "action", "consequence"):
            word = g.get(slot)
            if word:
                rows.append(f"  {slot:<12}{word}")
    return "\n".join(rows)


def auto_phase_hint(result: dict) -> Optional[dict]:
    """只建议下一轮的语体档：取本段实际相位均值，向 60° 回拉一档。"""
    phases = [float(s["phase_mean"]) for s in (result.get("sentences") or [])
              if s.get("phase_mean") is not None]
    if not phases:
        return None
    mean = sum(phases) / len(phases)
    suggested = round(max(0.0, min(120.0, mean * 0.6 + 60.0 * 0.4)), 1)
    return {"suggested_style_deg": suggested,
            "observed_phase_mean": round(mean, 1),
            "why": "按本段实际相位向 60° 回拉 40%，只建议、不自动生效"}


def breathing_line(turn: int, nonce: int = 0) -> dict:
    return {"line": _pick(BREATH_LINES, turn + nonce),
            "seconds": _pick(BREATH_SECONDS, turn + nonce)}


def cocreate_offer(result: dict) -> Optional[dict]:
    """只给候选边，不写；真正落地走 /feynman/digest 的 write 双闸。"""
    g = _anchor_glyphs(result)
    ent, act = g.get("entity"), g.get("action")
    if not ent or not act:
        return None
    return {"candidate_edges": [f"{ent}|{act}"],
            "channel": "/feynman/digest",
            "requires": "write=true 且 FEYNMAN_ENABLE_SENSORY_WRITE=1",
            "note": "这里只提议；不会自行修改扩展库。"}


def apply_dialogue_modes(vault, result: dict, modes: Optional[DialogueModes],
                         nonce: int = 0, turn: int = 0) -> dict:
    """只返回开启项；全部关闭时返回空 dict。"""
    if modes is None or not modes.any_on():
        return {}
    out: dict = {}
    if modes.socratic:
        q = socratic_question(result, nonce)
        if q:
            out["socratic"] = q
    if modes.portrait:
        p = metaphor_portrait(result, vault)
        if p:
            out["portrait"] = p
    if modes.auto_phase:
        h = auto_phase_hint(result)
        if h:
            out["auto_phase"] = h
    if modes.breathing:
        out["breathing"] = breathing_line(turn, nonce)
    if modes.cocreate:
        c = cocreate_offer(result)
        if c:
            out["cocreate"] = c
    return out
