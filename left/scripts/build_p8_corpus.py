"""把 su01 集成语料库（英文+中文对齐）转成铸字器吃的语料格式。

输入：base_library_integrated.jsonl（每条 messages 含中英两道；这里只取中文
assistant 正文，meta 里的来源/体裁写进 source/section 便于追溯）。
输出：src/section/text 的 JSONL，供 `feynman_vault.py --p8 --trusted --corpus` 使用。

用法：
    python scripts/build_p8_corpus.py \
        --library ~/path/to/base_library_integrated.jsonl \
        --out data/p8_corpus_integrated_zh.jsonl
"""

import argparse
import json
from pathlib import Path


def convert(library: Path, out: Path) -> dict:
    kept = skipped = 0
    seen = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with library.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as w:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                skipped += 1
                continue
            meta = rec.get("meta") or {}
            if meta.get("lang") != "zh":
                continue
            answer = ""
            for msg in rec.get("messages") or []:
                if msg.get("role") == "assistant" and msg.get("content"):
                    answer = str(msg["content"]).strip()
                    break
            if len(answer) < 30:
                skipped += 1
                continue
            text = " ".join(answer.split())
            if text in seen:
                skipped += 1
                continue
            seen.add(text)
            ident = str(meta.get("id") or rec.get("uid") or "")
            origin = ident.split("#")[0] or "集成库"
            row = {
                "source": f"费曼集成库｜{meta.get('genre', 'unknown')}｜{origin}",
                "section": ident,
                "text": text,
            }
            w.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
    return {"kept": kept, "skipped": skipped, "out": str(out)}


def append_authored(extra: Path, out: Path) -> dict:
    """把旧书库语料并进来，但只留通过署名白名单的段（防止"10000个科学难题"这类混入）。"""
    from plugins.feynman_vault import _clean_source

    kept = dropped = 0
    with extra.open(encoding="utf-8") as fh, out.open("a", encoding="utf-8") as w:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not _clean_source(str(rec.get("source", ""))):
                dropped += 1
                continue
            w.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    return {"kept": kept, "dropped": dropped}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/p8_corpus_integrated_zh.jsonl"))
    ap.add_argument("--extra", type=Path, default=None,
                    help="额外语料（source/section/text JSONL），只并入署名白名单内的段")
    args = ap.parse_args()
    stats = convert(args.library, args.out)
    print(f"✅ 转换完成：{stats['kept']} 条，跳过 {stats['skipped']} 条 -> {stats['out']}")
    if args.extra:
        more = append_authored(args.extra, args.out)
        print(f"✅ 并入旧书库：{more['kept']} 条，白名单外丢弃 {more['dropped']} 条")


if __name__ == "__main__":
    main()
