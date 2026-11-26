"""语料自动量子化切片器（Corpus → 量子字粒库）。

输入一段中文语料，输出三槽位（主语意象 / 量子动作 / 时空归宿）的 QGlyph：
每个字粒携带复数振幅 — 模长来自搭配强度与稀有度，相位来自语义锚点，
套话字粒取**负振幅**（与正项在同一路径里相消，而不是靠额外乘小系数硬压）。

提取优先走在线模型（DeepSeek JSON 模式）；不可用或解析失败时回落到
零依赖的搭配统计引擎：按标点切句 → 2~4 字候选块 → 过滤虚词边界 →
按搭配凝聚度与词长排序 → 用动词位置把名词分流进主语槽/归宿槽。

设计上刻意不叫"量子分词"：这里的相位与振幅是可解释的启发式量，
不是对真实量子态的近似；所有魔数都写在本文件顶部，便于一起调。
"""

import hashlib
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from shared.deepseek_client import chat_completion, client_available

# ---------------------------------------------------------------------------
# 可调参数（集中在这里，避免散落魔数）
# ---------------------------------------------------------------------------
CLICHE_COUNT_THRESHOLD = 4          # 同一词块在语料里出现这么多次即视为套话
MAX_PHRASE_LEN = 4
MIN_PHRASE_LEN = 2
DEFAULT_MAX_PER_SLOT = 3
SLOT_CAPS = {"SUBJECT": 8, "VERB": 8, "OBJECT": 8}

# 虚词：出现在词块首/尾即丢弃；同时作为离线切段的边界
FUNCTION_CHARS = set(
    "的了着过是在和与或把被将从向于就都而及其之为以对由并把吗呢吧啊呀嘛么这那我你他她它们个些"
    "很太更最又还只才再刚正不没别从沿向往朝跟给让使到却竟已曾要会能想该应须可每各"
    "当用条张片颗种次遍场座辆架艘层"
)

# 允许跨字拼接的连接词：「孤独 + 的 + 粒子」→「孤独的粒子」
JOIN_CHARS = ("的", "之", "与", "和")

# 套话标记（命中即打 cliche 标签、取负振幅）
CLICHE_MARKERS = (
    "我们", "你们", "他们", "非常", "进行", "一个", "可以", "真的", "不错",
    "今天", "东西", "问题", "时候", "大家", "觉得", "然后", "所以", "但是",
    "这个", "那个", "什么", "怎么", "知道", "喜欢", "其实", "已经", "还是",
    "就是", "不是", "一起", "开始", "结束", "事情",
)

# 相位锚点：标签 → (基准相位°, 关键词)
ANCHORS: Dict[str, Tuple[float, Tuple[str, ...]]] = {
    "logic": (10.0, (
        "公式", "定理", "证明", "逻辑", "边界", "守恒", "系统", "结构", "实验",
        "数据", "因果", "模型", "计算", "物理", "定律", "方程", "测量", "单位",
    )),
    "nature": (45.0, (
        "春风", "大海", "山川", "落叶", "飞鸟", "细雨", "森林", "月光", "雪",
        "河流", "云端", "微风", "花", "雨", "山", "海", "风", "云", "树",
    )),
    "quantum": (90.0, (
        "量子", "波函数", "光", "电子", "时空", "维度", "隧穿", "坍缩", "宇宙",
        "星辰", "引力", "原子", "概率", "观测", "路径", "纠缠", "叠加", "相干",
        "粒子", "能量", "振荡", "震荡",
    )),
    "time": (165.0, (
        "时间", "钟表", "影子", "岁月", "记忆", "昨天", "黄昏", "尘埃", "废墟",
        "遗忘", "尽头", "深夜", "孤独", "梦境", "宿命", "沉默", "寂静", "沉睡",
    )),
}

# 动词种子：离线模式判槽 + 定位主语/宾语分界
VERB_SEEDS = (
    "穿过", "穿越", "行走", "震荡", "振荡", "坍缩", "唤醒", "沉睡", "折射",
    "折叠", "吞噬", "凝视", "跨越", "编织", "颠覆", "倾听", "点燃", "解构",
    "重塑", "拥抱", "坠落", "漂移", "漂浮", "燃烧", "照亮", "淹没", "遗忘",
    "等待", "升起", "裂开", "缝合", "煮沸", "冷却", "旋转", "滑动", "奔跑",
    "追赶", "呼吸", "眨眼", "抵达", "逃离", "抚平", "撕开", "刻下", "写下",
)

# 槽位为空时的兜底字粒（避免极短语料直接空盘）
FALLBACK = {
    "SUBJECT": [
        ("孤独的粒子", 0.88, 95.0, "quantum"),
        ("生锈的钟表", 0.80, 165.0, "time"),
        ("沉默的公式", 0.82, 12.0, "logic"),
    ],
    "VERB": [
        ("悄悄隧穿了", 0.88, 90.0, "quantum"),
        ("折叠了", 0.84, 160.0, "time"),
        ("折射出", 0.82, 45.0, "nature"),
    ],
    "OBJECT": [
        ("绝对零度的围墙", 0.90, 92.0, "quantum"),
        ("昨天的影子", 0.84, 168.0, "time"),
        ("守恒的边界", 0.80, 10.0, "logic"),
    ],
}


class QGlyph(BaseModel):
    """一颗量子字粒。amp 为有符号振幅：负值表示套话反相字粒。"""

    word: str
    slot: str = Field(..., description="SUBJECT / VERB / OBJECT")
    amp: float = Field(..., ge=-1.0, le=1.0)
    phase_deg: float = Field(..., ge=0.0, le=180.0)
    tag: str
    count: int = 0
    cliche: bool = False


class FontBank(BaseModel):
    """由原始语料提炼出的三槽位量子活字盘。"""

    bank_id: str
    mode: str
    source_summary: str
    subjects: List[QGlyph]
    verbs: List[QGlyph]
    objects: List[QGlyph]

    def slot_items(self) -> Dict[str, List[QGlyph]]:
        return {"SUBJECT": self.subjects, "VERB": self.verbs, "OBJECT": self.objects}


def corpus_digest(text: str, max_per_slot: int, mode: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(
        f"{mode}|{max_per_slot}|{normalized}".encode("utf-8")
    ).hexdigest()[:16]


def _stable_jitter(word: str) -> float:
    """确定性相位扰动：同一词在任何进程里都得到同一角度。"""
    h = sum((i + 1) * ord(ch) for i, ch in enumerate(word))
    return float(h % 17 - 8)


def _anchor_for(word: str) -> Tuple[str, float]:
    for tag, (base_deg, keywords) in ANCHORS.items():
        if any(k in word for k in keywords):
            return tag, base_deg
    return "nature", ANCHORS["nature"][0]


def _is_cliche(word: str, count: int) -> bool:
    return count >= CLICHE_COUNT_THRESHOLD or any(m in word for m in CLICHE_MARKERS)


def _clause_is_cliche(clause: str) -> bool:
    """套话句判定：一句里出现两个以上套话标记，且句内没有语义锚点。"""
    hits = sum(1 for m in CLICHE_MARKERS if m in clause)
    return hits >= 2 and _anchor_bonus(clause) == 0.0


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in re.split(r"[，。！？；、,.!?;\n\r…—\s\"“”'‘’（）()]+", text) if c.strip()]


def _verb_positions(clause: str) -> List[int]:
    hits = []
    for seed in VERB_SEEDS:
        start = 0
        while True:
            idx = clause.find(seed, start)
            if idx < 0:
                break
            hits.append(idx)
            start = idx + len(seed)
    return sorted(hits)


# ---------------------------------------------------------------------------
# 离线启发式切片
# ---------------------------------------------------------------------------
def _verb_spans(clause: str) -> List[Tuple[int, int]]:
    """在短句里定位所有动词种子的跨度（长词优先，互不重叠），并吞掉时态/趋向后缀。"""
    spans: List[Tuple[int, int]] = []
    for seed in sorted(VERB_SEEDS, key=len, reverse=True):
        start = 0
        while True:
            i = clause.find(seed, start)
            if i < 0:
                break
            if not any(i < b and i + len(seed) > a for a, b in spans):
                end = i + len(seed)
                if end < len(clause) and clause[end] in "了着过出":
                    end += 1
                spans.append((i, end))
            start = i + 1
    return sorted(spans)


def _segments(clause: str) -> List[str]:
    """按虚词与动词把短句切成实词段，动词本身不进入意象候选。"""
    spans = dict(_verb_spans(clause))
    parts: List[str] = []
    cur = ""
    i = 0
    while i < len(clause):
        if i in spans:
            if cur:
                parts.append(cur)
                cur = ""
            i = spans[i]
            continue
        ch = clause[i]
        if ch in FUNCTION_CHARS:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(ch)
        else:
            cur += ch
        i += 1
    if cur:
        parts.append(cur)
    return parts


def _clause_candidates(clause: str) -> List[str]:
    """一段话里的候选意象：实词段本身的 2~4 字窗口 + 跨「的/与/和」的拼接短语。"""
    parts = _segments(clause)
    cands: List[str] = []
    for part in parts:
        if len(part) == 1 and part in FUNCTION_CHARS:
            continue
        if len(part) <= MAX_PHRASE_LEN:
            if MIN_PHRASE_LEN <= len(part) <= MAX_PHRASE_LEN:
                cands.append(part)
        else:
            for i in range(len(part) - MIN_PHRASE_LEN + 1):
                for n in range(MIN_PHRASE_LEN, MAX_PHRASE_LEN + 1):
                    gram = part[i:i + n]
                    if len(gram) == n and gram[0] not in FUNCTION_CHARS and gram[-1] not in FUNCTION_CHARS:
                        cands.append(gram)

    for i in range(len(parts) - 2):
        left, joiner, right = parts[i], parts[i + 1], parts[i + 2]
        if len(joiner) == 1 and joiner in JOIN_CHARS and len(left) >= 2 and len(right) >= 2:
            joined = left + joiner + right
            if MIN_PHRASE_LEN <= len(joined) <= 6:
                cands.append(joined)
    return cands


def _verbs_in(clause: str) -> List[str]:
    """动词槽直接由种子词定位，避免把碎块当成动作；时态/趋向后缀已并入跨度。"""
    return [clause[start:end] for start, end in _verb_spans(clause)]


def _candidate_counts(clauses: List[str]) -> Tuple[Counter, Dict[str, str]]:
    """统计候选词频与首次出现位置（用于主语/宾语分流）。"""
    counts: Counter = Counter()
    first_pos: Dict[str, Tuple[int, int]] = {}
    for ci, clause in enumerate(clauses):
        for cand in _clause_candidates(clause):
            counts[cand] += 1
            first_pos.setdefault(cand, (ci, clause.find(cand)))
    for verb in (v for clause in clauses for v in _verbs_in(clause)):
        counts[verb] += 1
    return counts, first_pos


def _overlaps(a: str, b: str) -> bool:
    """两个意象共享任意二字片段即视为重复，避免「沉睡的公式 / 公式与星辰」并列。"""
    grams = {b[i:i + 2] for i in range(len(b) - 1)}
    return any(a[i:i + 2] in grams for i in range(len(a) - 1))


def _anchor_bonus(word: str) -> float:
    return 0.9 if any(k in word for _tag, (deg, kws) in ANCHORS.items() for k in kws) else 0.0


def _slot_for_noun(word: str, clause: str) -> str:
    """分句里的"头号意象"作主语：位置最先、同起点取更长；其余作归宿。"""
    cands = _clause_candidates(clause)
    if not cands:
        return "OBJECT"
    head = min(cands, key=lambda w: (clause.find(w), -len(w)))
    return "SUBJECT" if word == head else "OBJECT"


def _make_glyph(word: str, slot: str, count: int, clause: str = "") -> QGlyph:
    tag, base_phase = _anchor_for(word)
    phase = max(0.0, min(180.0, base_phase + _stable_jitter(word)))
    # 套话判定：命中套话标记/高频，或整句是套话且本词无任何语义锚点
    cliche = _is_cliche(word, count) or (
        bool(clause) and _clause_is_cliche(clause) and _anchor_bonus(word) == 0.0
    )
    if cliche:
        return QGlyph(
            word=word, slot=slot, amp=-round(min(0.80, 0.52 + 0.05 * len(word)), 3),
            phase_deg=round(phase, 1), tag="cliche", count=count, cliche=True,
        )
    rarity = 1.0 / (1.0 + count)
    amp = 0.60 + 0.06 * min(len(word), 6) + 0.10 * rarity + 0.06 * _anchor_bonus(word)
    return QGlyph(
        word=word, slot=slot, amp=round(min(0.96, amp), 3),
        phase_deg=round(phase, 1), tag=tag, count=count, cliche=False,
    )


def _heuristic_quantize(text: str, max_per_slot: int, mode: str) -> FontBank:
    clauses = _clauses(text)
    counts, _ = _candidate_counts(clauses)
    verbs = {v for clause in clauses for v in _verbs_in(clause)}

    def score(word: str) -> float:
        return 0.20 * len(word) + _anchor_bonus(word) + 0.35 * min(counts[word], 3) + (
            0.25 if any(j in word for j in JOIN_CHARS) else 0.0
        )

    ranked = sorted(counts, key=lambda w: (-score(w), -len(w), w))

    slots: Dict[str, List[QGlyph]] = {"SUBJECT": [], "VERB": [], "OBJECT": []}
    chosen: Dict[str, List[str]] = {"SUBJECT": [], "VERB": [], "OBJECT": []}
    used: set = set()

    for word in ranked:
        if len(slots["VERB"]) >= max_per_slot and len(slots["SUBJECT"]) >= max_per_slot \
                and len(slots["OBJECT"]) >= max_per_slot:
            break
        if word in used:
            continue
        if word in verbs:
            slot = "VERB"
        else:
            slot = next(
                (_slot_for_noun(word, c) for c in clauses if word in c), "OBJECT"
            )
        if len(slots[slot]) >= max_per_slot:
            continue
        # 同槽内不保留互为子串或共享二字片段的重复意象
        if any(word in w or _overlaps(word, w) for w in chosen[slot]):
            continue
        clause = next((c for c in clauses if word in c), "")
        slots[slot].append(_make_glyph(word, slot, counts[word], clause))
        chosen[slot].append(word)
        used.add(word)

    # 只给"动作槽"兜底：动词是句法胶水；名词槽宁缺毋滥（空槽会让印刷如实拒绝）
    filled = 0
    if not slots["VERB"]:
        for word, amp, phase, tag in FALLBACK["VERB"][:max_per_slot]:
            slots["VERB"].append(QGlyph(word=word, slot="VERB", amp=amp, phase_deg=phase, tag=tag))
            filled += 1

    total = sum(len(v) for v in slots.values())
    summary = f"语料 {len(text)} 字 → {total} 颗字粒（离线搭配引擎）"
    if filled:
        summary += f"，其中 {filled} 颗取自内置量子字盘兜底"

    return FontBank(
        bank_id=corpus_digest(text, max_per_slot, mode),
        mode=mode,
        source_summary=summary,
        subjects=slots["SUBJECT"],
        verbs=slots["VERB"],
        objects=slots["OBJECT"],
    )


# ---------------------------------------------------------------------------
# 在线切片（DeepSeek JSON 模式），失败即回落
# ---------------------------------------------------------------------------
_LLM_PROMPT = """你是"量子活字印刷铸字师"。从下面的中文语料中挑选最具画面感、思想张力的字、词或短语，
分别装入 subjects（主语意象）、verbs（动作）、objects（归宿意象）三个槽位，每槽至多 {cap} 个。

铁律：
1. word 必须是语料里原样出现的连续字串，不得改写；2~6 字。
2. amp 取值 -1.0~1.0：套话/空洞词（我们、非常、进行、真的……）取负值；隐喻张力越强取正值越大。
3. phase_deg 取 0~180：0=严谨理性，45=自然写意，90=量子奇幻，165=深沉时空/宿命。
4. tag 只能取 logic / nature / quantum / time / cliche 之一，同源意象用同一标签。
5. 只输出 JSON：{{"subjects":[{{"word":..,"amp":..,"phase_deg":..,"tag":..}}],"verbs":[...],"objects":[...]}}

语料：
{corpus}
"""


def _llm_quantize(text: str, max_per_slot: int, mode: str) -> Optional[FontBank]:
    if not client_available():
        return None
    model = os.getenv("WORKER_MODEL", "deepseek-chat")
    prompt = _LLM_PROMPT.format(cap=max_per_slot, corpus=text[:3000])
    try:
        raw = chat_completion(
            model, prompt, temperature=0.2, max_tokens=1200,
            response_format={"type": "json_object"}, timeout=30.0,
        )
        payload = json.loads(raw)
    except Exception:
        return None

    slots: Dict[str, List[QGlyph]] = {"SUBJECT": [], "VERB": [], "OBJECT": []}
    for key, slot in (("subjects", "SUBJECT"), ("verbs", "VERB"), ("objects", "OBJECT")):
        for item in (payload.get(key) or [])[:max_per_slot]:
            word = str(item.get("word", "")).strip()
            if not (MIN_PHRASE_LEN <= len(word) <= 6) or word not in text:
                continue
            try:
                amp = max(-1.0, min(1.0, float(item.get("amp", 0.8))))
                phase = max(0.0, min(180.0, float(item.get("phase_deg", 90.0))))
            except (TypeError, ValueError):
                continue
            tag = str(item.get("tag", "nature"))
            if tag not in (*ANCHORS.keys(), "cliche"):
                tag = "nature"
            slots[slot].append(QGlyph(
                word=word, slot=slot, amp=round(amp, 3), phase_deg=round(phase, 1),
                tag=tag, cliche=(tag == "cliche" or amp < 0),
            ))

    if not all(slots.values()):
        return None  # 任一槽为空则视为提取失败，回落离线

    total = sum(len(v) for v in slots.values())
    return FontBank(
        bank_id=corpus_digest(text, max_per_slot, mode),
        mode=mode,
        source_summary=f"语料 {len(text)} 字 → {total} 颗字粒（在线语义铸字）",
        subjects=slots["SUBJECT"],
        verbs=slots["VERB"],
        objects=slots["OBJECT"],
    )


async def quantize_corpus_to_glyphs(
    corpus_text: str, max_per_slot: int = DEFAULT_MAX_PER_SLOT
) -> FontBank:
    text = corpus_text.strip()
    if not text:
        raise ValueError("语料为空")
    max_per_slot = max(2, min(SLOT_CAPS["SUBJECT"], int(max_per_slot)))
    mode = "online" if client_available() else "offline"

    bank = _llm_quantize(text, max_per_slot, mode) if mode == "online" else None
    return bank or _heuristic_quantize(text, max_per_slot, mode)


if __name__ == "__main__":
    import asyncio

    sample = (
        "在微观的量子世界里，孤独的粒子从不沿着单一的轨迹行走。它像一阵春风穿过时间的废墟，"
        "在无数条可能的路径上同时震荡。当观测者的目光折射出好奇的光芒，所有的概率云瞬间坍缩，"
        "在寂静的宇宙尽头唤醒了沉睡的公式与星辰。"
    )
    bank = asyncio.run(quantize_corpus_to_glyphs(sample, max_per_slot=3))
    print(bank.source_summary)
    for name, items in (("主语意象槽", bank.subjects), ("量子动作槽", bank.verbs), ("时空归宿槽", bank.objects)):
        print(f"\n【{name}】")
        for g in items:
            print(f"  {g.word:<8} amp={g.amp:+.2f} phase={g.phase_deg:5.1f}° tag=#{g.tag} count={g.count}")
