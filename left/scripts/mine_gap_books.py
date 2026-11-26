"""P15 空档实体·跨书候选挖掘：扫桌面中文物理书 + p8 语料，落候选清单。

只写候选文件；不重建 vault、不改既有产物。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/mine_gap_books.py \
        --books ~/Desktop/费曼物理_文字版/raw_md \
        --corpus data/p8_corpus_v2.jsonl \
        --out data/gap_entity_book_candidates.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from plugins.gap_clause_miner import aggregate_books, mine_entity_clauses
import plugins.feynman_vault as fv

# P10 普查后仍缺专属从句的实体（人物名词另案处理，见 NARRATIVE_ENTITIES）
DEFAULT_ENTITIES = [
    "坐标系", "量子力学", "直线", "常数", "平面", "强度", "颜色",
    "函数", "速率", "长度", "电路", "线圈", "曲线",
]


def _corpus_rows(corpus: Path, entities: set[str]):
    counts: Counter = Counter()
    sources: dict[tuple[str, str], list[str]] = {}
    examples: dict[tuple[str, str], str] = {}
    for src, section, text in fv.iter_sections(corpus):
        tag = f"{src}｜{section[:24]}"
        for sent in fv.sentences(text):
            for hit in mine_entity_clauses(sent, entities):
                key = (hit["entity"], hit["clause"])
                counts[key] += 1
                examples.setdefault(key, sent)
                lst = sources.setdefault(key, [])
                if tag not in lst and len(lst) < 5:
                    lst.append(tag)
    return counts, sources, examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=str(Path.home() / "Desktop/费曼物理_文字版/raw_md"))
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--out", default="data/gap_entity_book_candidates.json")
    ap.add_argument("--min-books", type=int, default=1)
    args = ap.parse_args()

    entities = set(DEFAULT_ENTITIES)
    book_paths = sorted(Path(args.books).glob("*.md"))
    rows = aggregate_books(book_paths, entities, min_books=args.min_books)

    counts: Counter = Counter((r["entity"], r["clause"]) for r in rows
                              for _ in range(r["count"]))
    provenance: dict[tuple[str, str], list[str]] = {
        (r["entity"], r["clause"]): [f"书｜{b}" for b in r["sources"]] for r in rows}
    examples: dict[tuple[str, str], str] = {
        (r["entity"], r["clause"]): r["example"] for r in rows}

    c2, s2, e2 = _corpus_rows(Path(args.corpus), entities)
    for key, n in c2.items():
        counts[key] += n
        examples.setdefault(key, e2[key])
        lst = provenance.setdefault(key, [])
        for tag in s2[key]:
            if tag not in lst and len(lst) < 5:
                lst.append(tag)

    out_rows = []
    for (entity, clause), n in counts.items():
        out_rows.append({
            "entity": entity,
            "clause": clause,
            "count": n,
            "sources": provenance.get((entity, clause), []),
            "example": examples.get((entity, clause), "")[:120],
        })
    out_rows.sort(key=lambda r: (r["entity"], -r["count"], r["clause"]))

    # 溯源自检：每条候选必须逐字节出现在 example 里
    bad = [r for r in out_rows if r["clause"] not in r["example"]]
    if bad:
        raise SystemExit(f"溯源校验失败 {len(bad)} 条，例：{bad[0]}")

    per_entity = Counter(r["entity"] for r in out_rows)
    payload = {
        "entities": DEFAULT_ENTITIES,
        "books_scanned": len(book_paths),
        "candidates": len(out_rows),
        "per_entity_counts": dict(sorted(per_entity.items(),
                                         key=lambda kv: (-kv[1], kv[0]))),
        "rows": out_rows,
        "checksum": hashlib.sha256(
            json.dumps(out_rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"扫描书 {len(book_paths)} 本；候选 {len(out_rows)} 条 → {args.out}")
    print("每实体：" + "、".join(f"{e}={n}" for e, n in payload["per_entity_counts"].items()))


if __name__ == "__main__":
    main()
