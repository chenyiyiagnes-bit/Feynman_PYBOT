"""P15 后果库存：P10 库存（391）+ 人工策展的非费曼物理书从句。

与 P10 的区别只在数据来源：P10 的专属从句挖自费曼语料本身，
P15 补的是费曼语料里结构性缺失的空档实体（强度/速率/颜色/…），
候选来自本机中文物理书（桌面 raw_md）与 p8 语料。

每一步都可溯源：
- 每条从句必须逐字节出现在某本书或语料句子里，否则拒收；
- 计数、示例、来源标签都指向真实句子；
- 边补充只包含"非语料书"里的共现，语料内共现仍由 build_vault 自己统计，
  避免重复计数。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_p15_inventory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import plugins.feynman_vault as fv
from plugins.gap_clause_miner import (
    _clause_ok,
    _frame_invariance_ok,
    iter_book_sentences,
)


def _load_corpus_sources(corpus: Path) -> set[str]:
    return {Path(src).stem for src, _sec, _txt in fv.iter_sections(corpus)}


def _book_sources(books_dir: Path, corpus_stems: set[str]):
    """返回 [(stem, path, in_corpus)]，按文件名排序保证确定性。"""
    out = []
    for p in sorted(Path(books_dir).glob("*.md")):
        out.append((p.stem, p, p.stem in corpus_stems))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", default=str(Path.home() / "Desktop/费曼物理_文字版/raw_md"))
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--p10", default="data/p10_consequence_inventory.json")
    ap.add_argument("--vault", default="data/feynman_style_vault_p10.json")
    ap.add_argument("--allowlist", default="data/p15_gap_allowlist.json")
    ap.add_argument("--out", default="data/p15_consequence_inventory.json")
    ap.add_argument("--edges", default="data/p15_consequence_edges.json")
    ap.add_argument("--report", default="data/p15_consequence_inventory_report.json")
    args = ap.parse_args()

    p10 = json.loads(Path(args.p10).read_text(encoding="utf-8"))
    p10_words = {it["word"] for it in p10}
    vault = json.loads(Path(args.vault).read_text(encoding="utf-8"))
    entity_words = {g["word"] for g in vault["entities"]}

    allow = json.loads(Path(args.allowlist).read_text(encoding="utf-8"))
    rows = allow["rows"]

    corpus_path = Path(args.corpus)
    corpus_stems = _load_corpus_sources(corpus_path)

    # 语料句子先读一遍；书按需读，但为确定性先排好序
    corpus_sents: list[tuple[str, str]] = []   # (tag, sentence)
    for src, section, text in fv.iter_sections(corpus_path):
        tag = f"{src}｜{section[:24]}"
        for sent in fv.sentences(text):
            corpus_sents.append((tag, sent))

    books = _book_sources(Path(args.books), corpus_stems)
    book_sents: dict[str, list[tuple[str, str]]] = {}
    for stem, path, in_corpus in books:
        if in_corpus:
            continue
        sents = []
        for sent in iter_book_sentences(path):
            sents.append((f"书｜{stem}", sent))
        book_sents[stem] = sents

    added, rejected = [], []
    items_new: list[dict] = []
    edges: Counter = Counter()
    per_entity = Counter()

    for r in rows:
        w = r["clause"]
        e = r["entity"]
        if e not in entity_words:
            rejected.append({"word": w, "entity": e, "why": "entity_not_in_vault"})
            continue
        if w in p10_words:
            rejected.append({"word": w, "entity": e, "why": "duplicate"})
            continue
        gate = r.get("gate", "consequence")
        if gate == "frame_invariance":
            ok = _frame_invariance_ok(w)
        elif gate == "consequence":
            ok = _clause_ok(w)
        else:
            rejected.append({"word": w, "entity": e, "why": f"unknown_gate:{gate}"})
            continue
        if not ok:
            rejected.append({"word": w, "entity": e, "why": f"{gate}_gate"})
            continue

        hits_corpus = [(tag, s) for tag, s in corpus_sents if w in s]
        hits_books = []
        for stem, _path, in_corpus in books:
            if in_corpus:
                continue
            hits_books.extend((tag, s) for tag, s in book_sents[stem] if w in s)
        if not hits_corpus and not hits_books:
            rejected.append({"word": w, "entity": e, "why": "not_verbatim_in_sources"})
            continue

        all_hits = hits_corpus + hits_books
        sample_tag, sample = all_hits[0]
        sources = []
        for tag, _s in all_hits:
            if tag not in sources and len(sources) < 3:
                sources.append(tag)

        # 边补充：只算非语料书里的 E×C 同句共现；语料内由 build_vault 自己数
        pair_books = sum(1 for tag, s in hits_books if e in s)
        if pair_books:
            edges[f"{e}|{w}"] += pair_books

        style, phase = fv._style_of(sample)
        items_new.append({
            "word": w,
            "slot": "consequence",
            "count": len(all_hits),
            "amp": 0.0,                       # 合并后统一按 max_count 重算
            "domain": "physics",
            "style": style,
            "phase_deg": phase,
            "sources": sources,
            "example": fv._clean_consequence(fv._trim_tail(sample))[:60],
            "tier": 2,
            "curated": True,                  # 人工策展：tier2 内优先于自动挖出的从句
            "source_kind": "book" if hits_books else "corpus",
            "entity": e,
        })
        added.append({"word": w, "entity": e, "count": len(all_hits),
                      "corpus_hits": len(hits_corpus), "book_hits": len(hits_books),
                      "sample": sample[:80]})
        per_entity[e] += 1

    merged = [dict(it) for it in p10] + items_new
    max_count = max(it["count"] for it in merged) if merged else 1
    for it in merged:
        it["amp"] = round(0.94 + 0.06 * math.log1p(it["count"]) / math.log1p(max_count), 3)
    merged.sort(key=lambda it: (-it["count"], it["word"]))

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    Path(args.edges).write_text(json.dumps(dict(sorted(edges.items())),
                                           ensure_ascii=False, indent=1),
                                encoding="utf-8")

    report = {
        "p10_words": len(p10),
        "allowlist_rows": len(rows),
        "added": len(added),
        "rejected": len(rejected),
        "rejected_detail": rejected,
        "inventory_words": len(merged),
        "tier1": sum(1 for it in merged if it["tier"] == 1),
        "tier2": sum(1 for it in merged if it["tier"] == 2),
        "per_entity_added": dict(sorted(per_entity.items())),
        "deferred": allow.get("deferred", []),
        "edges": len(edges),
        "checksum": hashlib.sha256(
            json.dumps(merged, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    print(json.dumps({k: report[k] for k in
                      ("p10_words", "allowlist_rows", "added", "rejected",
                       "inventory_words", "tier1", "tier2", "edges")},
                     ensure_ascii=False, indent=1))
    print("每实体新增：" + "、".join(f"{e}={n}" for e, n in report["per_entity_added"].items()))
    if rejected:
        print(f"⚠️ 拒收 {len(rejected)} 条：" + "；".join(
            f"{r['word']}（{r['why']}）" for r in rejected[:8]))


if __name__ == "__main__":
    main()
