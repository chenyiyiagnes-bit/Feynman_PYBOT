"""P16 收尾：把 40 组"基线 vs 张力弧"段落做成盲评材料，等人打分。

只做三件事：
1. 用字库里的实体/动作确定性拼 40 个问句（不联网、不造新语料）；
2. 每个问句同一 nonce 出两段：关闭张力弧的基线、开启张力弧的重排版；
3. 用问句哈希决定 A/B 摆位，输出盲评表 `data/ab_tension_pairs.md` 与
   揭盲钥匙 `data/ab_tension_key.json`；报告里不写谁是谁。

注意：脚本只生成材料，不替人打分；结果未回收前，任何"哪版更好"的结论都不成立。

用法：
    PYTHONPATH=. .venv/bin/python -B scripts/ab_tension_blind.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from plugins.feynman_typesetter import Vault
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph

VAULT_PATH = Path("data/feynman_style_vault_p15.json")
PAIRS_MD = Path("data/ab_tension_pairs.md")
KEY_JSON = Path("data/ab_tension_key.json")
PAIR_COUNT = 40
STYLE_DEG = 60.0
BEAM = 6

ACTION_HINTS = ("振动", "发光", "冷却", "吸引", "穿过", "碰撞", "流动", "弯曲",
                "旋转", "膨胀", "收缩", "传导", "反射", "蒸发", "凝结", "扩散",
                "加速", "减速", "上升", "下沉")


def build_queries(vault: Vault, count: int = PAIR_COUNT) -> list[str]:
    ents = [str(g["word"]) for g in vault.by_slot["entity"]]
    acts = [a for a in ACTION_HINTS if any(g["word"] == a for g in vault.by_slot["action"])]
    acts = acts or [str(g["word"]) for g in vault.by_slot["action"]]
    queries: list[str] = []
    seen: set[str] = set()
    for i in range(count * 4):
        if len(queries) >= count:
            break
        e = ents[i % len(ents)]
        a = acts[(i // max(1, len(ents) // len(acts) or 1)) % len(acts)]
        q = f"{e}为什么{('会' if not a.endswith('射') else '会')}{a}"
        if q in seen:
            q = f"{e}怎么会{a}，这件事奇怪吗"
        if q in seen:
            continue
        seen.add(q)
        queries.append(q)
    return queries


def side_order(query: str) -> tuple[str, str]:
    """按问句哈希把基线放 A 或 B；同一问句永远同一摆位，可复现。"""
    h = int(hashlib.sha256(query.encode("utf-8")).hexdigest()[:8], 16)
    return ("baseline", "tension") if h % 2 == 0 else ("tension", "baseline")


def render_pair(req: ParagraphRequest, vault) -> tuple[str, str]:
    off = weave_paragraph(vault, ParagraphRequest(**{**req.model_dump(),
                                                     "dissonance_resolution": False}))
    on = weave_paragraph(vault, ParagraphRequest(**{**req.model_dump(),
                                                    "dissonance_resolution": True}))
    return off, on


def main() -> None:
    vault = Vault(VAULT_PATH)
    queries = build_queries(vault)
    key: dict[str, dict] = {}
    md = ["# P16 张力弧盲评表（40 组）", "",
          "每组两段，A/B 已随机摆位。请只按阅读感受打分：1–5 分，"
          "并在备注写更想读哪一段（A/B/持平）。打完再把评分表交给脚本揭盲。", ""]
    for i, q in enumerate(queries, 1):
        req = ParagraphRequest(user_query=q, style_deg=STYLE_DEG, nonce=100 + i, beam_width=BEAM)
        off, on = render_pair(req, vault)
        first, second = side_order(q)
        blocks = {"baseline": off, "tension": on}
        md.append(f"## {i:02d}. {q}")
        for side, variant in (("A", first), ("B", second)):
            res = blocks[variant]
            md.append(f"**{side}**（{len(res.get('sentences') or [])} 句）")
            md.append("")
            md.append(res.get("paragraph") or "（留白）")
            arc = res.get("tension_arc") or {}
            if variant == "tension" and arc.get("T1") is not None:
                md.append("")
                md.append(f"> 张力弧：T1={arc['T1']:.2f} → T3={arc['T3']:.2f}")
            md.append("")
        md.append("评分：A ___ / 5    B ___ / 5    更想读：＿＿    备注：＿＿")
        md.append("")
        key[f"pair_{i:02d}"] = {"query": q, "A": first, "B": second}
    md.append("---")
    md.append("揭盲钥匙在 `data/ab_tension_key.json`；回收评分后再看，"
              "现在不下任何胜负结论。")
    PAIRS_MD.write_text("\n".join(md), encoding="utf-8")
    KEY_JSON.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已生成 {PAIRS_MD}（{len(queries)} 组）与 {KEY_JSON}")


if __name__ == "__main__":
    main()
