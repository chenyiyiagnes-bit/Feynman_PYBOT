"""拓扑消融盲评表：只用 ablate_topology_p15.py 的 60 种问法（每种两句）。

输出两件东西：
- data/rating_worksheet_topology_blind.jsonl  只有 id/query/text，打乱、不标配置
- data/rating_worksheet_topology_blind.map.json  id → off/on（评分时不可看）

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/make_topology_rating_worksheet.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OUT = Path("data/rating_worksheet_topology_blind.jsonl")
MAP = Path("data/rating_worksheet_topology_blind.map.json")
SRC = Path("data/eval_topology_ablation_p15.json")


def _rid(*parts: str) -> str:
    return "t" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:9]


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if r["off"] and r["on"]]
    items, mapping = [], {}
    for r in rows:
        q = r["query"]
        for cfg, text in (("off", r["off"]), ("on", r["on"])):
            if text == r["off"] and text == r["on"]:
                continue  # 两句完全一样，比不出东西
            rid = _rid(q, cfg)
            items.append({"id": rid, "query": q, "text": text, "rating": None})
            mapping[rid] = {"query": q, "config": "topology_" + cfg, "text": text}
    random.Random(20261125).shuffle(items)
    OUT.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items) + "\n",
                   encoding="utf-8")
    MAP.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(items)} 行 → {OUT}")
    print(f"映射 → {MAP}")


if __name__ == "__main__":
    main()
