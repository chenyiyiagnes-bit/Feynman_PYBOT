"""P19 只读感知接入：外界丢进来一段文字，本机只读取、只回一段费曼式读后感。

三条硬规矩：
1. 默认只读：不写扩展库、不改任何分数，响应里的 `learned` 恒为 None；
2. 可选写入必须二次开闸：请求带 `write=true` 且环境变量
   `FEYNMAN_ENABLE_SENSORY_WRITE=1`，否则如实回"未开闸"，不偷偷写；
3. 写入也只走现有受控通道 `apply_evolution_update`：实体过 `sanitize_entity`、
   动作必须在字库动作表里、最多 3 条边、`reason="sensory_ingest"`，
   只动经验乘子，不新增字粒、不改原典。

不引入 jieba：实体/动作匹配用字库自带的词表做最长匹配，离线、确定、
不需要分词模型；匹配不到就老实说没匹配到。
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

MAX_DIGEST_CHARS = 2000
MAX_EDGES = 3
WRITE_ENV = "FEYNMAN_ENABLE_SENSORY_WRITE"
WRITE_HINT = f"{WRITE_ENV}=1"
MIN_MATCH_LEN = 2

# 读后感只做语气提示，不参与事实判断：文字里有这些词就算"厚实/单薄"一点
ABSORB_POSITIVE = ("实验", "观测", "因为", "守恒", "证明", "所以", "精确", "测量")
ABSORB_NEGATIVE = ("大概", "也许", "听说", "据说", "感觉", "玄学", "反正")

_BAD_SEGMENT = re.compile(r"[。！？!?；;\n]+")


def digest_nonce(text: str) -> int:
    """稳定 nonce：同一段文字永远排同一版，便于复现与测试。"""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 10000


def clamp_chars(text: str, limit: int = MAX_DIGEST_CHARS) -> str:
    return (text or "").strip()[:limit]


def _words_from(vault, slot: str) -> List[str]:
    try:
        return [str(g["word"]) for g in vault.by_slot.get(slot, []) if g.get("word")]
    except Exception:
        return []


def _match_all(text: str, words: Iterable[str]) -> List[str]:
    """按出现顺序（再按长度降序）收集出现过的词；纯子串匹配，无分词依赖。"""
    hits = []
    seen = set()
    for w in sorted({str(x) for x in words if len(str(x)) >= MIN_MATCH_LEN},
                    key=lambda s: (-len(s), s)):
        idx = text.find(w)
        if idx >= 0 and w not in seen:
            seen.add(w)
            hits.append((idx, w))
    return [w for _, w in sorted(hits)]


def matched_entities(text: str, vault, extra: Optional[Iterable[str]] = None) -> List[str]:
    words = _words_from(vault, "entity")
    if extra:
        words += [str(w) for w in extra if w]
    return _match_all(text, words)


def matched_action(text: str, vault) -> Optional[str]:
    acts = _match_all(text, _words_from(vault, "action"))
    return acts[0] if acts else None


def absorption_score(text: str) -> float:
    """0.0–1.0 的粗糙"信息厚度"，只用于可选写入的乘子幅度，不是事实判断。"""
    pos = sum(1 for t in ABSORB_POSITIVE if t in text)
    neg = sum(1 for t in ABSORB_NEGATIVE if t in text)
    total = pos + neg
    if not total:
        return 0.0
    return round(pos / total, 3)


def segments(text: str) -> List[str]:
    return [s.strip() for s in _BAD_SEGMENT.split(text or "") if s.strip()]


def write_requested(write: bool) -> bool:
    return bool(write) and os.getenv(WRITE_ENV, "").strip() in ("1", "true", "yes")


def build_digest(text: str, vault, ext_hash: str = "00000000",
                 extra_entities: Optional[Iterable[str]] = None,
                 style_deg: float = 60.0) -> dict:
    """只读路径：匹配 → 走既有段落编织器 → 原样返回，不写任何东西。"""
    from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph

    body = clamp_chars(text)
    nonce = digest_nonce(body)
    ents = matched_entities(body, vault, extra_entities)[:MAX_EDGES]
    action = matched_action(body, vault)
    segs = segments(body)
    query = segs[0][:120] if segs else body[:120]

    paragraph = {"sentences": [], "paragraph": ""}
    if query:
        try:
            paragraph = weave_paragraph(
                vault, ParagraphRequest(user_query=query, style_deg=style_deg,
                                        nonce=nonce, beam_width=6))
        except Exception:
            paragraph = {"sentences": [], "paragraph": "", "degraded": True}

    return {
        "version": 1,
        "read_only": True,
        "chars": len(body),
        "nonce": nonce,
        "matched_entities": ents,
        "matched_action": action,
        "absorption": absorption_score(body),
        "reflection": (paragraph.get("paragraph") or "").strip(),
        "sentences": paragraph.get("sentences") or [],
        "ext_hash": ext_hash or "00000000",
        "learned": None,
    }


def plan_write(digest: dict, allowed_actions: Sequence[str]) -> Optional[dict]:
    """把一次性读后感翻译成受控更新参数；不满足条件就返回 None（什么都不写）。"""
    ents = [e for e in (digest.get("matched_entities") or [])][:MAX_EDGES]
    action = digest.get("matched_action")
    if not ents or not action or action not in set(allowed_actions):
        return None
    # 厚度偏正面才加分，偏负面才减分；只在 ±0.10 内，且与隐式信号同一个钳位通道
    score = float(digest.get("absorption") or 0.0)
    delta = round((score - 0.5) * 0.2, 3)
    if abs(delta) < 0.01:
        return None
    return {"entities": ents, "action": action, "delta": delta,
            "reason": "sensory_ingest"}


def apply_digest_write(plan: dict, allowed_actions: set,
                       vault_id: Optional[str] = None) -> str:
    """实际写入口：调用方必须已经过 `write_requested` 开闸。"""
    from plugins.evolution_loop import apply_evolution_update

    return apply_evolution_update(
        plan["entities"], plan["action"], "", plan["delta"],
        plan["reason"], set(allowed_actions), vault_id)


def render_digest_reply(result: dict) -> str:
    """给网关用的一行摘要：四处都只说实际发生了什么。"""
    ents = "、".join(result.get("matched_entities") or []) or "无"
    action = result.get("matched_action") or "无"
    learned = result.get("learned")
    if learned:
        tail = f"学到 {len(learned.get('entities') or [])} 条搭配（±{learned.get('delta')}）"
    elif result.get("write_blocked"):
        tail = "写入未开闸，本次只读"
    else:
        tail = "只读"
    return f"📖 认出实体：{ents}｜动作：{action}｜{tail}"
