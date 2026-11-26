#!/usr/bin/env python3
"""把旁路爬虫的共现实体转成右脑候选对——只进候选区，永不直接成边。

输入：`left/data/domain_extensions.json`（爬虫只写这一处）。
输出：`right/scripts/data/assoc_candidates_web.json`，字段与 `mine_assoc.py`
的候选一致：`kind="same_domain"`、`origin="cooccur"`、`mechanism=""`。
这些对不会说话，必须经人工/代审后才能写进 `assoc_edges.jsonl`。

匹配规则（保守）：
- 实体出现在卡片标题里 → 强度 2；
- 只出现在答案或标签里 → 强度 1；
- 标题/答案都没有、但实体的所有二字片段都在卡片词表里 → 强度 0.5，
  且只在该实体没有任何强匹配时才采用，避免“量子”+“退火”拼出来的假重叠。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_EXT = ROOT.parent / "left" / "data" / "domain_extensions.json"
DEFAULT_OUT = ROOT / "scripts" / "data" / "assoc_candidates_web.json"
MAX_CARDS_PER_ENTITY = 6
MAX_PAIRS = 200


def crawl_entities(ext_data: Dict) -> List[str]:
    """扩展库里所有抓取成功的实体，按首次出现顺序去重。"""
    ordered: List[str] = []
    seen = set()
    for record in ext_data.get("crawl_runs") or []:
        if (record.get("status") or "") != "ok":
            continue
        entity = (record.get("entity") or "").strip()
        if entity and entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    if ordered:
        return ordered
    # 老库没有 crawl_runs，只看 crawl_log 的键
    for entity in (ext_data.get("crawl_log") or {}):
        entity = (entity or "").strip()
        if entity and entity not in seen:
            seen.add(entity)
            ordered.append(entity)
    return ordered


def _bigrams(text: str) -> List[str]:
    return [text[i:i + 2] for i in range(len(text) - 1)]


def doc_haystack(doc) -> Tuple[str, str]:
    title = (getattr(doc, "title", "") or "")
    answer = (getattr(doc, "answer", "") or "")
    return title, answer


def match_strength(doc, entity: str) -> float:
    title, answer = doc_haystack(doc)
    if entity and entity in title:
        return 2.0
    if entity and entity in answer:
        return 1.0
    weights = getattr(doc, "weights", None) or {}
    pieces = _bigrams(entity)
    if len(pieces) >= 2 and all(piece in weights for piece in pieces):
        return 0.5
    return 0.0


def match_docs(docs: Iterable, entity: str,
               limit: int = MAX_CARDS_PER_ENTITY) -> List[Tuple[float, object]]:
    """一个实体落到哪些卡上；强匹配存在时丢掉词表拼接级的弱匹配。"""
    scored = []
    for doc in docs:
        if (getattr(doc, "language", "zh") or "zh") != "zh":
            continue
        strength = match_strength(doc, entity)
        if strength > 0:
            scored.append((strength, doc))
    strong = [pair for pair in scored if pair[0] >= 1.0]
    if strong:
        scored = strong
    scored.sort(key=lambda pair: (-pair[0], getattr(pair[1], "id", "")))
    return scored[:limit]


def build_entity_hits(docs: Sequence, entities: Sequence[str]) -> Dict[str, List[dict]]:
    hits: Dict[str, List[dict]] = {}
    for entity in entities:
        matched = match_docs(docs, entity)
        if matched:
            hits[entity] = [
                {"id": getattr(doc, "id", ""), "title": getattr(doc, "title", ""),
                 "strength": strength}
                for strength, doc in matched
            ]
    return hits


def build_pairs(entity_hits: Dict[str, List[dict]]) -> List[dict]:
    """同一实体落到两张以上卡片时，两两成候选对；共享实体越多分越高。"""
    shared: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    strength: Dict[Tuple[str, str], float] = defaultdict(float)
    for entity, hits in entity_hits.items():
        by_id = {hit["id"]: float(hit.get("strength") or 0.0)
                 for hit in hits if hit.get("id")}
        ids = sorted(by_id)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                key = (a, b)
                shared[key].append(entity)
                strength[key] += (by_id[a] + by_id[b]) / 2.0

    pairs = []
    for (a, b), entities in shared.items():
        pairs.append({
            "a": a,
            "b": b,
            "kind": "same_domain",
            "origin": "cooccur",
            "mechanism": "",
            "shared": sorted(entities),
            "score": round(strength[(a, b)], 2),
        })
    pairs.sort(key=lambda p: (-p["score"], p["a"], p["b"]))
    return pairs[:MAX_PAIRS]


def build_report(docs: Sequence, ext_data: Dict) -> Dict:
    entities = crawl_entities(ext_data)
    hits = build_entity_hits(docs, entities)
    pairs = build_pairs(hits)
    sources = {}
    for record in ext_data.get("crawl_runs") or []:
        entity = (record.get("entity") or "").strip()
        if entity and record.get("source_url"):
            sources[entity] = record["source_url"]
    return {
        "note": ("爬虫共现候选：kind=same_domain、origin=cooccur，"
                 "mechanism 为空，永不直接说话；须代审后才可写边。"),
        "entities": entities,
        "unmatched": [e for e in entities if e not in hits],
        "hits": hits,
        "sources": sources,
        "pairs": pairs,
    }


def load_docs():
    from omega_mind.feynman_kb import FeynmanKB

    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise SystemExit("找不到 feynman-open 知识库；先在能读到语料的机器上运行。")
    return kb.docs


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ext", default=str(DEFAULT_EXT),
                        help="domain_extensions.json 路径")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="候选输出路径")
    args = parser.parse_args(argv)

    ext_path = Path(args.ext)
    if not ext_path.is_file():
        raise SystemExit(f"扩展库不存在：{ext_path}")
    ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
    report = build_report(load_docs(), ext_data)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"实体 {len(report['entities'])}｜命中 {len(report['hits'])}｜"
          f"未匹配 {len(report['unmatched'])}｜候选对 {len(report['pairs'])}")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
