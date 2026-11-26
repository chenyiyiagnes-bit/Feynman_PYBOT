"""离线知识库扩展管线·第一段：实体专属后果取矿并落候选。

扫描 p8 语料，按实体主语位置抽取分句，聚合频次/来源，输出候选清单。
只写候选文件；不重建 vault、不改任何既有产物。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/mine_entity_consequences.py \
        --corpus data/p8_corpus_v2.jsonl \
        --vault data/feynman_style_vault_p9.json \
        --out data/entity_consequence_candidates.json \
        --min-count 3 --min-sections 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from plugins.entity_clause_miner import aggregate, filter_candidates
import plugins.feynman_vault as fv


def _sentences(path: Path):
    """与 vault/共现表同一份句视图：同样的来源过滤、噪声/去世词过滤与句长窗。

    之前按原始段落分句，挖出的从句可能落在 sentences() 丢掉的超长块里，
    重建后 E→C 边为 0；这里统一口径，保证每条候选都在合法句内。
    """
    for src, section, text in fv.iter_sections(path):
        tag = f"{src}｜{section[:24]}"
        for sent in fv.sentences(text):
            yield {"text": sent, "section": tag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/p8_corpus_v2.jsonl")
    ap.add_argument("--vault", default="data/feynman_style_vault_p9.json")
    ap.add_argument("--out", default="data/entity_consequence_candidates.json")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--min-sections", type=int, default=2)
    args = ap.parse_args()

    vault = json.loads(Path(args.vault).read_text(encoding="utf-8"))
    entities = {g["word"] for g in vault["entities"]}

    rows_raw = aggregate(_sentences(Path(args.corpus)), entities,
                         min_count=args.min_count, min_sections=args.min_sections)
    rows = filter_candidates(rows_raw)

    # 溯源自检：每条候选的 clause 必须逐字节出现在 example 原句里
    bad = [r for r in rows if r["clause"] not in r["example"]]
    if bad:  # 不可能发生；发生即管线 bug，宁可失败
        raise SystemExit(f"溯源校验失败 {len(bad)} 条，例：{bad[0]}")

    per_entity: dict[str, list] = {}
    for r in rows:
        per_entity.setdefault(r["entity"], []).append(r)

    covered = len(per_entity)
    out = {
        "corpus": args.corpus,
        "vault": args.vault,
        "min_count": args.min_count,
        "min_sections": args.min_sections,
        "entities": len(entities),
        "covered_entities": covered,
        "candidates": len(rows),
        "per_entity_counts": {e: len(v) for e, v in
                              sorted(per_entity.items(), key=lambda kv: (-len(kv[1]), kv[0]))},
        "rows": rows,
        "checksum": hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"实体 {len(entities)}；有候选 {covered}；候选 {len(rows)} 条；"
          f"溯源校验 {len(rows)}/{len(rows)} 通过 → {args.out}")
    thin = [e for e in sorted(entities) if e not in per_entity]
    print(f"仍无候选的实体（{len(thin)}）：{'、'.join(thin)}")


if __name__ == "__main__":
    main()
