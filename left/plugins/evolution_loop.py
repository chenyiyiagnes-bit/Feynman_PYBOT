"""受控自进化闭环：只学"用户点过赞/顺着追问"的搭配，且严格钳位、可一键清空。

和 PYBOT3 草图的三处不同（草图照抄会学坏）：
- 隐式信号不再拿单句四槽相干度冒充跨轮相干度；同一 session 下看实体延续与相位漂移。
- GC 按分数排序，不按字典序（草图按 key 排序等于随机淘汰）。
- 所有读-改-写走 `update_extensions` 的进程锁，后台任务并发不丢更新；
  写入前校验实体与动作，客户端塞什么字符串都不会进库。

学到的只是 `(实体, 动作)` 的经验乘子在排版里生效，费曼原典字库始终只读。
"""

import re
import time
from collections import OrderedDict
from typing import Dict, List, Optional

MIN_EDGE_SCORE = 0.15
MAX_EDGE_SCORE = 1.50
MAX_TRACKED_SESSIONS = 1000
PROMOTE_EDGE_SCORE = 1.15  # 经验分到这里才把搭配升级为正式共现边

IMPLICIT_FOLLOWUP_BONUS = 0.15
IMPLICIT_SHIFT_DECAY = -0.05
EXPLICIT_UPVOTE_BONUS = 0.30
EXPLICIT_DOWNVOTE_PENALTY = -0.45

MAX_ENTITIES = 3
_MAX_ENTITY_LEN = 8
_BAD_CHARS = re.compile(r"[\s|,，。;；:：\"'`\\/<>{}\[\]]")


def sanitize_entity(word: str) -> Optional[str]:
    """只收 2–8 字、无分隔符/换行的名词；其余当作注入直接丢弃。"""
    w = (word or "").strip()
    if not (2 <= len(w) <= _MAX_ENTITY_LEN) or w.isdigit():
        return None
    if not any(ch.isalnum() for ch in w):
        return None
    if _BAD_CHARS.search(w):
        return None
    return w


def _validate(entities: List[str], action: str, allowed_actions: Optional[set]) -> List[str]:
    act = (action or "").strip()
    if not act:
        return []
    if allowed_actions is not None and act not in allowed_actions:
        return []
    cleaned: List[str] = []
    for ent in entities[:MAX_ENTITIES]:
        w = sanitize_entity(ent)
        if w and w not in cleaned:
            cleaned.append(w)
    return cleaned


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


MAX_EDGE_ENTRIES = 500


def _gc(ext: dict) -> None:
    """超过容量时只留分数最高的边；正式共现边同步收缩到留下来的键。"""
    scores: Dict[str, float] = ext.get("edge_scores") or {}
    limit = ext.get("max_edge_scores") or MAX_EDGE_ENTRIES
    if len(scores) <= limit:
        return
    kept = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit])
    ext["edge_scores"] = kept
    valid = {tuple(k.split("|", 1)) for k in kept if "|" in k}
    ext["cooccurrence_edges"] = [list(e) for e in sorted(
        {tuple(e) for e in ext.get("cooccurrence_edges") or []} & valid)]


def apply_evolution_update(entities: List[str], action: str, skeleton: str,
                           delta: float, reason: str = "feedback",
                           allowed_actions: Optional[set] = None,
                           vault_id: Optional[str] = None) -> str:
    """后台任务入口：更新经验乘子/骨架偏好，做钳位与 GC，原子落盘并返回新 ext_hash。

    vault_id 只做溯源标记（字库 bank_id），不参与打分：换字库后 /stats 能看出
    这些经验乘子是在哪个字库上学的；旧扩展库没有该字段时按"未标记"处理。
    """
    from plugins.domain_expander import load_domain_extensions_with_hash, update_extensions

    ents = _validate(entities, action, allowed_actions)
    if not ents:
        _, h = load_domain_extensions_with_hash()
        return h
    delta = _clamp(float(delta), -1.0, 1.0)

    def mutate(ext: dict) -> None:
        edges = ext.setdefault("edge_scores", {})
        cooc = {tuple(e) for e in ext.get("cooccurrence_edges") or []}
        for ent in ents:
            key = f"{ent}|{action}"
            new = round(_clamp(edges.get(key, 1.0) + delta, MIN_EDGE_SCORE, MAX_EDGE_SCORE), 3)
            if new <= MIN_EDGE_SCORE:
                edges.pop(key, None)
                cooc.discard((ent, action))
            else:
                edges[key] = new
                if new >= PROMOTE_EDGE_SCORE:
                    cooc.add((ent, action))
                elif new < PROMOTE_EDGE_SCORE:
                    cooc.discard((ent, action))
        if skeleton:
            sk = ext.setdefault("skeleton_scores", {})
            sk[skeleton] = round(
                _clamp(sk.get(skeleton, 1.0) + delta * 0.5, 0.5, MAX_EDGE_SCORE), 3)
        ext["cooccurrence_edges"] = [list(e) for e in sorted(cooc)]
        ext["last_evolved_at"] = int(time.time())
        ext["last_evolved_reason"] = reason
        if vault_id:
            ext["vault_id"] = vault_id
        _gc(ext)

    return update_extensions(mutate)


def learned_multiplier(entity: str, action: str, edge_scores: Optional[Dict[str, float]]) -> float:
    if not edge_scores:
        return 1.0
    return float(edge_scores.get(f"{entity}|{action}", 1.0))


class LastTurn:
    __slots__ = ("entities", "action", "skeleton", "style_deg")

    def __init__(self, entities: List[str], action: str, skeleton: str, style_deg: float):
        self.entities = [e for e in entities if e]
        self.action = action
        self.skeleton = skeleton
        self.style_deg = style_deg


class ImplicitSignalTracker:
    """同一 session 的上一轮快照；只为判断"用户在顺着追问还是换话题"。"""

    def __init__(self, max_sessions: int = MAX_TRACKED_SESSIONS):
        self._turns: "OrderedDict[str, LastTurn]" = OrderedDict()
        self._max = max_sessions

    def forget(self, session_id: Optional[str]) -> bool:
        """清掉某个 session 的上一轮快照；/reset 后下一轮不再被当成"顺着追问"。
        返回是否真的删掉了记录。"""
        if not session_id or session_id not in self._turns:
            return False
        del self._turns[session_id]
        return True

    def observe(self, session_id: Optional[str], entities: List[str], action: str,
                skeleton: str, style_deg: float) -> Optional[dict]:
        if not session_id:
            return None
        prev = self._turns.get(session_id)
        signal = None
        if prev is not None and prev.entities:
            overlap = set(prev.entities) & set(entities)
            coherent = bool(overlap) or abs(style_deg - prev.style_deg) <= 8.0
            drifted = not overlap and abs(style_deg - prev.style_deg) > 20.0
            if coherent:
                signal = {"entities": prev.entities, "action": prev.action,
                          "skeleton": prev.skeleton, "delta": IMPLICIT_FOLLOWUP_BONUS,
                          "reason": "implicit_followup"}
            elif drifted:
                signal = {"entities": prev.entities, "action": prev.action,
                          "skeleton": prev.skeleton, "delta": IMPLICIT_SHIFT_DECAY,
                          "reason": "implicit_topic_shift"}
        self._turns[session_id] = LastTurn(entities, action, skeleton, style_deg)
        self._turns.move_to_end(session_id)
        while len(self._turns) > self._max:
            self._turns.popitem(last=False)
        return signal

    def clear(self, session_id: str) -> None:
        self._turns.pop(session_id, None)


implicit_tracker = ImplicitSignalTracker()
