"""为现有字库重算「可打分的」纠缠对：只取 实体×后果 的语料共现边。

历史字库里的 entangled_pairs 来自分句候选名词，和打分键 (entity, consequence)
对不上，从未生效。本脚本不改字库文件、不改 bank_id，只产出配对表，
供 `QBOT_ENTANGLED_PAIRS` 挂载做消融实验。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_entangled_pairs.py
    ... --vault data/feynman_style_vault_p15.json --out data/entangled_pairs_p15.json
    ... --top 80 --min-count 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plugins.feynman_vault import _entangled_from_cooc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="data/feynman_style_vault_p15.json")
    ap.add_argument("--out", default="data/entangled_pairs_p15.json")
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()

    raw = json.loads(Path(args.vault).read_text(encoding="utf-8"))
    vocab = {
        "entity": raw.get("entities") or [],
        "consequence": raw.get("consequences") or [],
    }
    pairs = _entangled_from_cooc(raw.get("cooccurrence") or {}, vocab,
                                 top=args.top, min_count=args.min_count)
    Path(args.out).write_text(
        json.dumps(pairs, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"✅ {args.out}：{len(pairs)} 对（实体×后果，count≥{args.min_count}）")
    if pairs:
        print("例：", ", ".join(f"{p['everyday']}×{p['physics']}({p['count']})"
                               for p in pairs[:5]))


if __name__ == "__main__":
    main()
