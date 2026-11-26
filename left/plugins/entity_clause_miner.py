"""实体专属后果挖掘器（离线扩展管线的取证级）。

从语料里只摘"实体在主语位置、后接谓词"的分句，作为后果槽候选。与
P10 的 E→C 共现检索不同，这里不认"同句出现"，只认句法位置，专门用来
替换共现池里"行星→电子绕原子核旋转"这类噪声。

全部规则确定性、无模型调用；候选必须逐字节可在原句中找到（溯源）。
产物只写候选文件，不碰 vault；入库前另行评审。

用法见 scripts/mine_entity_consequences.py。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

import jieba.posseg as pseg

# 分句：这些标点都断开，只保留同一小句
SPLIT = re.compile(r"[。；！？\n，、：]")

# 句首可剥掉的连接词（可叠加）
CONNECTORS = (
    "所以", "因此", "但是", "可是", "然而", "而且", "并且", "然后", "于是",
    "那么", "如果", "假如", "要是", "当", "因为", "由于", "其实", "事实上",
    "首先", "其次", "例如", "比如", "换句话说", "也就是说", "总之", "同时",
    "现在", "这样", "那样", "当然", "显然", "可见", "这就是", "这说明",
)

# 句首可剥掉的量词/限定词（只剥一层）
QUANTIFIERS = (
    "所有的", "所有", "每个", "任何", "两个", "三个", "一些", "许多", "多数",
    "少数", "整个", "部分", "一个", "一种", "这个", "那个", "这些", "那些",
    "这种", "那种", "这样的", "那样的", "同样的", "相同的",
)

PRONOUNS = ("我们", "你们", "他们", "她们", "它们", "我", "你", "他", "她", "它")
QUESTION_WORDS = ("为什么", "怎样", "怎么", "如何", "是否", "吗", "呢", "？", "?")
BAD_CHARS = set("$\\{}_^<>|*#`~")

MIN_LEN = 4
MAX_LEN = 24
VERB_WINDOW = 4          # 实体之后最多看几个词找谓词


def _strip_connectors(clause: str) -> tuple[str, int]:
    """剥掉句首连接词，返回剩余文本及其偏移。量词保留（属于主语的一部分）。"""
    offset = 0
    changed = True
    while changed:
        changed = False
        for c in CONNECTORS:
            if clause.startswith(c, offset):
                offset += len(c)
                changed = True
                break
    return clause[offset:], offset


def _strip_quantifier(text: str) -> str:
    for q in QUANTIFIERS:
        if text.startswith(q):
            return text[len(q):]
    return text


def _starts_with_entity(remainder: str, entity: str) -> bool:
    """remainder 的第一个词是否恰好是 entity（容忍 jieba 把长实体切碎）。"""
    if not remainder.startswith(entity):
        return False
    acc = ""
    for word, _flag in pseg.cut(remainder):
        acc += word
        if len(acc) < len(entity):
            continue
        return acc == entity
    return acc == entity


def _has_predicate(remainder: str, entity: str) -> bool:
    """实体之后 VERB_WINDOW 个词内出现动词。"""
    rest = remainder[len(entity):]
    if not rest:
        return False
    for i, (word, flag) in enumerate(pseg.cut(rest)):
        if i >= VERB_WINDOW:
            break
        if flag.startswith("v"):
            return True
    return False


def _gate(clause: str, subject: str, entity: str) -> bool:
    if not (MIN_LEN <= len(clause) <= MAX_LEN):
        return False
    if any(ch in BAD_CHARS for ch in clause):
        return False
    if any(p in clause for p in PRONOUNS):
        return False
    if any(q in clause for q in QUESTION_WORDS):
        return False
    if subject[len(entity):].startswith("的"):
        return False
    if not _starts_with_entity(subject, entity):
        return False
    return _has_predicate(subject, entity)


def mine_clauses(text: str, entities: set[str]) -> list[dict]:
    """返回该句中每个分句至多一个候选：第一个实体作主语、后接谓词。"""
    out = []
    for raw in SPLIT.split(text or ""):
        clause = raw.strip()
        if not clause:
            continue
        # 最早出现的实体；同位置取更长的（引力优先于力）
        hits = []
        for e in entities:
            pos = clause.find(e)
            if pos >= 0:
                hits.append((pos, -len(e), e))
        if not hits:
            continue
        hits.sort()
        entity = hits[0][2]
        after, _offset = _strip_connectors(clause)
        subject = _strip_quantifier(after)
        if not subject.startswith(entity):
            continue
        if not _gate(after, subject, entity):
            continue
        out.append({"entity": entity, "clause": after})
    return out


def aggregate(sentences: Iterable[dict], entities: set[str],
              min_count: int = 3, min_sections: int = 2) -> list[dict]:
    """按 (实体, 分句) 聚合出现次数、来源小节与首次原句。

    sentences 逐条含 ``text`` 与 ``section``；同一句里重复出现只算一次。
    """
    count: Counter = Counter()
    sections: dict[tuple, set] = defaultdict(set)
    example: dict[tuple, str] = {}
    for row in sentences:
        text = row.get("text", "")
        section = row.get("section", "")
        seen = set()
        for c in mine_clauses(text, entities):
            key = (c["entity"], c["clause"])
            if key in seen:
                continue
            seen.add(key)
            count[key] += 1
            sections[key].add(section)
            example.setdefault(key, text)

    rows = []
    for (entity, clause), n in count.items():
        secs = sorted(sections[(entity, clause)])
        if n < min_count or len(secs) < min_sections:
            continue
        rows.append({
            "entity": entity,
            "clause": clause,
            "count": n,
            "sections": secs,
            "example": example[(entity, clause)],
        })
    rows.sort(key=lambda r: (r["entity"], -r["count"], r["clause"]))
    return rows


# ---- 二次质量门禁：聚合后仍不合格的候选在这里挡掉 ----

# 叙事实体：句法上当主语也抽不出物理后果，整类不进后果槽
NARRATIVE_ENTITIES = frozenset({
    "学生", "家伙", "朋友", "父亲", "物理学家", "科学家", "费曼",
})

# 以报告/认知动词收尾 = 截断从句（"量子力学表明" / "科学家们发现"）
REPORT_ENDINGS = (
    "表明", "说明", "表示", "发现", "认为", "指出", "提出", "相信", "觉得",
    "意识到", "同意", "写道", "回答", "问道", "强调", "记得", "知道", "看到",
    "说过", "说道", "看到", "看出", "证明", "解释",
)

# 状语残句（"…时" / "…中" / "…里"），后面原本还有主句
ADVERBIAL_ENDINGS = ("时", "的时候", "当中", "中", "里", "上", "下")

# 指代前文的元话语（脱离上下文不可读）
DEIXIS = (
    "上面", "下面", "前述", "上述", "所谓", "刚才", "如前", "前文", "后文",
    "上一", "下一", "本章", "这一章", "这一节", "这节课", "图中", "见图",
    "见表", "如图", "如表", "上式", "下式", "前面提到",
)

BRACKET_PAIRS = (("（", "）"), ("(", ")"), ("“", "”"), ("「", "」"),
                 ("《", "》"), ("〈", "〉"))


def _brackets_balanced(clause: str) -> bool:
    for left, right in BRACKET_PAIRS:
        if clause.count(left) != clause.count(right):
            return False
    return True


def _reject_reason(clause: str) -> str | None:
    if clause.endswith(REPORT_ENDINGS):
        return "report_fragment"
    if clause.endswith(ADVERBIAL_ENDINGS):
        return "adverbial_fragment"
    if not _brackets_balanced(clause):
        return "unbalanced_brackets"
    if any(d in clause for d in DEIXIS):
        return "discourse_deixis"
    return None


def filter_candidates(rows: list[dict],
                      exclude_entities: frozenset[str] = NARRATIVE_ENTITIES
                      ) -> list[dict]:
    """去掉叙事实体、残句、元指代，以及同一实体的子串重复。"""
    kept = [r for r in rows
            if r["entity"] not in exclude_entities
            and _reject_reason(r["clause"]) is None]
    by_entity: dict[str, list[str]] = defaultdict(list)
    for r in kept:
        by_entity[r["entity"]].append(r["clause"])
    out = []
    for r in kept:
        longer = any(r["clause"] != other and r["clause"] in other
                     for other in by_entity[r["entity"]])
        if not longer:
            out.append(r)
    return out
