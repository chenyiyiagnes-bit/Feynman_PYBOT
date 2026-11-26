"""P21 段落盲评材料：40 组"基线 vs 开启同构"段落，A/B 随机摆位 + 机械指标。

两类问句各一半：
- 语料外实体（股票/咖啡/台风…）：测"用户实体 × 动作相容"乘子；
- 语料内实体（电子/光/磁场…）：测全路径全局机制项。

同一 nonce 出两段，摆位由问句哈希决定，可复现。脚本只生成材料与机械指标，
不替人/模型打分；盲评表里不写谁是谁，钥匙单独存。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ab_p21_blind.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plugins.feynman_typesetter import Vault
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph

VAULT_PATH = Path("data/feynman_style_vault_p15.json")
PAIRS_MD = Path("data/ab_p21_pairs.md")
KEY_JSON = Path("data/ab_p21_key.json")
PROXY_JSON = Path("data/ab_p21_proxy.json")
PROXY_MD = Path("data/ab_p21_proxy.md")
STYLE_DEG = 60.0
BEAM = 6

OOV_QUERIES = [
    "股票为什么会下跌", "猫为什么喜欢纸箱", "咖啡为什么会变凉", "合同为什么要有条款",
    "台风为什么会转向", "公司为什么会长大", "算法为什么会收敛", "音乐为什么好听",
    "城市为什么变热", "货币为什么会贬值", "钥匙为什么能开锁", "沙子为什么烫脚",
    "影子为什么跟着人走", "面包为什么会发霉", "电梯为什么失重", "镜子为什么会反光",
]


def build_vault_queries(vault: Vault, count: int) -> list[str]:
    from scripts.ab_tension_blind import build_queries
    return build_queries(vault, count)


def side_order(query: str) -> tuple[str, str]:
    h = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
    return ("baseline", "p21") if h % 2 == 0 else ("p21", "baseline")


def mechanical(res: dict) -> dict:
    sents = res.get("sentences") or []
    ents = res.get("user_entities") or []
    lines = [s.get("line") for s in sents]
    glyph_ents = [(s.get("glyphs") or {}).get("entity") for s in sents]
    target = ents[0] if ents else None
    anchored = sum(1 for e in glyph_ents if e and target and e == target)
    return {
        "complete": bool(res.get("complete")),
        "blank": not (res.get("paragraph") or "").strip(),
        "sentences": len(sents),
        "target_entity": target,
        "entity_anchored": anchored,
        "entity_anchor_rate": round(anchored / len(glyph_ents), 3) if glyph_ents else 0.0,
        "duplicate_lines": len(lines) - len(set(lines)),
    }


def main() -> None:
    vault = Vault(VAULT_PATH)
    queries = list(OOV_QUERIES) + build_vault_queries(vault, 24)
    key: dict[str, dict] = {}
    proxy_rows = []
    md = ["# P21 段落盲评表（40 组）", "",
          "每组两段，A/B 已按问句哈希随机摆位。只按阅读感受评分：",
          "1 答非所问 / 2 勉强 / 3 说得通 / 4 顺畅且相关 / 5 好读且贴题；"
          "备注写更想读哪段（A/B/持平）。", ""]
    for i, q in enumerate(queries, 1):
        req = ParagraphRequest(user_query=q, style_deg=STYLE_DEG, nonce=100 + i,
                               beam_width=BEAM)
        off = weave_paragraph(vault, ParagraphRequest(**req.model_dump()))
        on = weave_paragraph(vault, ParagraphRequest(**{**req.model_dump(),
                                                        "use_isomorphism": True}))
        first, second = side_order(q)
        blocks = {"baseline": off, "p21": on}
        md.append(f"## {i:02d}. {q}")
        for side, variant in (("A", first), ("B", second)):
            md.append(f"**{side}**")
            md.append("")
            md.append(blocks[variant].get("paragraph") or "（留白）")
            md.append("")
        md.append("评分：A ___ / 5    B ___ / 5    更想读：＿＿    备注：＿＿")
        md.append("")
        key[f"pair_{i:02d}"] = {"query": q, "A": first, "B": second}
        proxy_rows.append({"pair": f"pair_{i:02d}", "query": q,
                           "baseline": mechanical(off), "p21": mechanical(on)})

    md.append("---")
    md.append("揭盲钥匙在 `data/ab_p21_key.json`；回收评分后再揭盲，"
              "现在不下任何胜负结论。")
    PAIRS_MD.write_text("\n".join(md), encoding="utf-8")
    KEY_JSON.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")

    def agg(side: str) -> dict:
        rows = [r[side] for r in proxy_rows]
        return {
            "blank": sum(1 for r in rows if r["blank"]),
            "incomplete": sum(1 for r in rows if not r["complete"]),
            "entity_anchor_rate": round(
                sum(r["entity_anchored"] for r in rows)
                / max(1, sum(r["sentences"] for r in rows)), 3),
            "duplicate_lines": sum(r["duplicate_lines"] for r in rows),
        }

    summary = {"pairs": len(proxy_rows), "baseline": agg("baseline"),
               "p21": agg("p21")}
    PROXY_JSON.write_text(json.dumps({"summary": summary, "rows": proxy_rows},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# P21 机械指标（盲评前，客观计数）", "",
             f"{summary['pairs']} 组；两版同 nonce。", "",
             "| 指标 | 基线 | 开启 P21 |", "|---|---|---|",
             f"| 留白 | {summary['baseline']['blank']} | {summary['p21']['blank']} |",
             f"| 不足三句 | {summary['baseline']['incomplete']} | {summary['p21']['incomplete']} |",
             f"| 用户实体锁定率 | {summary['baseline']['entity_anchor_rate']} | {summary['p21']['entity_anchor_rate']} |",
             f"| 句内重复 | {summary['baseline']['duplicate_lines']} | {summary['p21']['duplicate_lines']} |",
             "", "机械指标只看形状，不判内容好坏；内容好坏走盲评。", ""]
    PROXY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"-> {PAIRS_MD}, {KEY_JSON}, {PROXY_JSON}, {PROXY_MD}")


if __name__ == "__main__":
    main()
