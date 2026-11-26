"""构建 P15 词表与拓扑图：P10 库存 + 非费曼物理书补库（468 条）。

不覆盖任何 p10 产物；输出：
    data/feynman_style_vault_p15.json
    data/topology_graph_p15.npz

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/build_vault_p15.py
"""

from __future__ import annotations

import json
from pathlib import Path

from plugins.feynman_vault import build_vault
from plugins.topology_builder import build_topology

CORPUS = Path("data/p8_corpus_v2.jsonl")
INVENTORY = Path("data/p15_consequence_inventory.json")
EDGES = Path("data/p15_consequence_edges.json")
VAULT_OUT = Path("data/feynman_style_vault_p15.json")
TOPO_OUT = Path("data/topology_graph_p15.npz")


def main() -> None:
    for p in (VAULT_OUT, TOPO_OUT):
        if p.exists():
            raise SystemExit(f"{p} 已存在，拒绝覆盖（旧产物保持不动）")

    vault = build_vault(CORPUS, p8=True, consequence_inventory=INVENTORY,
                        edge_supplement=EDGES)
    VAULT_OUT.write_text(json.dumps(vault, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    print(f"vault -> {VAULT_OUT}：hook {len(vault['hooks'])}、entity {len(vault['entities'])}、"
          f"action {len(vault['actions'])}、consequence {len(vault['consequences'])}、"
          f"cooccurrence {len(vault['cooccurrence'])}")

    meta = build_topology(VAULT_OUT, CORPUS, TOPO_OUT)
    keys = ("topo_hash", "words", "triangles", "window2_triangles",
            "window3_triangles", "section_triangles", "passage_triangles",
            "edges", "edge_support_ge2", "analogy_pairs")
    print(json.dumps({k: meta.get(k) for k in dict.fromkeys(keys)},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
