"""P15 空档实体·跨书从句挖掘器。

entity_clause_miner 要求实体作分句主语；对坐标系、直线、常数、函数
这类词几乎零产出——中文物理书里它们多作状语或宾语（沿直线、在平面内、
测得常数、波函数）。本模块改为"从句包含实体"，其余门禁与既有管线一致：

- 长度 4–24 字、无坏字符 / 代词 / 疑问词 / 元指代 / 图号引用
- 不以转述动词、状语残尾收束
- 必须通过 feynman_vault._p8_consequence_ok（完整从句 + 变化类动词）

只产出候选，不写 vault；由 build_p15_inventory 逐字节核对来源后才入库。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from plugins.entity_clause_miner import (
    ADVERBIAL_ENDINGS,
    BAD_CHARS,
    DEIXIS,
    MAX_LEN,
    MIN_LEN,
    PRONOUNS,
    QUESTION_WORDS,
    REPORT_ENDINGS,
    SPLIT,
    _strip_connectors,
)
import plugins.feynman_vault as fv

# 图号/表号/公式号引用：候选一旦指向“图9-8”就不是可独立成句的后果
FIGURE_REF = re.compile(r"[图表式]\s*\d|\d+\s*[-—]\s*\d+")
# 一个段落至少多少汉字才当成中文正文（英文章节整段跳过）
HAN_RATIO_MIN = 0.20


def _han_ratio(text: str) -> float:
    if not text:
        return 0.0
    han = sum(1 for ch in text if "一" <= ch <= "鿿")
    return han / len(text)


def iter_book_sentences(path: Path):
    """按段落读取一本书的 markdown/纯文本，产出与 vault 同口径的候选句。"""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    for para in text.split("\n"):
        if _han_ratio(para) < HAN_RATIO_MIN:
            continue
        yield from fv.sentences(para)


def _clause_ok(clause: str) -> bool:
    if not (MIN_LEN <= len(clause) <= MAX_LEN):
        return False
    if any(ch in BAD_CHARS for ch in clause):
        return False
    if any(p in clause for p in PRONOUNS):
        return False
    if any(q in clause for q in QUESTION_WORDS):
        return False
    if any(d in clause for d in DEIXIS):
        return False
    if FIGURE_REF.search(clause):
        return False
    if clause.endswith(REPORT_ENDINGS):
        return False
    if clause.endswith(ADVERBIAL_ENDINGS):
        return False
    if not fv._p8_consequence_ok(clause):
        return False
    return True


# "不变性陈述"专用门：坐标系这类词在物理中文里从不作变化的主语，
# 它的合法后果是"物理规律与坐标系无关"这类表述。只对显式标了
# gate=frame_invariance 的策展条目开放，自动挖掘仍走 _clause_ok。
INVARIANCE_MARKERS = ("无关", "相同", "等效", "不变")


def _frame_invariance_ok(clause: str) -> bool:
    if not (MIN_LEN <= len(clause) <= 15):
        return False
    if "坐标系" not in clause:
        return False
    if any(ch in BAD_CHARS for ch in clause):
        return False
    if any(p in clause for p in PRONOUNS):
        return False
    if any(q in clause for q in QUESTION_WORDS):
        return False
    if any(d in clause for d in DEIXIS):
        return False
    if FIGURE_REF.search(clause):
        return False
    if clause.endswith(REPORT_ENDINGS):
        return False
    if clause.endswith(ADVERBIAL_ENDINGS):
        return False
    if any(c in clause for c in fv.P8_CONDITIONAL_INNER):
        return False
    if not any(m in clause for m in INVARIANCE_MARKERS):
        return False
    return True


def mine_entity_clauses(text: str, entities: set[str]) -> list[dict]:
    """返回该句中每个分句至多一个候选：包含实体的整分句（实体不必作主语）。"""
    out = []
    for raw in SPLIT.split(text or ""):
        raw = raw.strip()
        if not raw:
            continue
        hits = [e for e in entities if e in raw]
        if not hits:
            continue
        entity = max(hits, key=len)
        clause, _offset = _strip_connectors(raw)
        clause = clause.strip()
        if not _clause_ok(clause):
            continue
        out.append({"entity": entity, "clause": clause})
    return out


def aggregate_books(paths, entities: set[str], min_books: int = 1) -> list[dict]:
    """跨书聚合候选：同一 (实体, 从句) 计出现次数与不同书数。"""
    counts: Counter = Counter()
    books: dict[tuple[str, str], set[str]] = {}
    examples: dict[tuple[str, str], str] = {}
    for p in paths:
        p = Path(p)
        for sent in iter_book_sentences(p):
            for hit in mine_entity_clauses(sent, entities):
                key = (hit["entity"], hit["clause"])
                counts[key] += 1
                books.setdefault(key, set()).add(p.stem)
                examples.setdefault(key, sent)
    rows = []
    for (entity, clause), n in counts.items():
        if len(books[(entity, clause)]) < min_books:
            continue
        rows.append({"entity": entity, "clause": clause, "count": n,
                     "books": len(books[(entity, clause)]),
                     "example": examples[(entity, clause)][:120],
                     "sources": sorted(books[(entity, clause)])[:5]})
    rows.sort(key=lambda r: (r["entity"], -r["count"], r["clause"]))
    return rows
