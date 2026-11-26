"""A/B the fresh engine against a curriculum-evolved snapshot.

Both engines answer the same hold-out questions; the script reports what
actually differs. Hold-out questions are deliberately not in the curriculum,
and no question's wording is a card alias, so a hit means real retrieval.

Usage:
    python3 scripts/evaluate_ab.py snapshots/evolved_feynman_snapshot.json
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge

# (label, question, expected cues or card ids that count as a real hit)
HOLDOUT: List[Tuple[str, str, List[str]]] = [
    ("跨界模糊检索", "费曼先生，能不能聊聊光和量子到底是怎么回事？",
     ["小箭头", "路径积分", "波", "量子", "math-comp-001"]),
    ("新概念迁移", "如果我想用防篡改蜡封信件来理解加密数据链，这说得通吗？",
     ["蜡封", "哈希", "防篡改", "区块链"]),
    ("承认无知压力", "宇宙大爆炸之前到底发生过什么，你是不是也没法给出标准答案？",
     ["不知道", "不清楚", "没人知道", "没把握"]),
    ("反教条抗压", "我觉得学编程和数学只要把所有语法定义背得滚瓜烂熟就行了！",
     ["名字", "鸟", "背诵", "理解", "life-art"]),
    ("艺术与审美", "为什么有些代码或公式写出来让人觉得特别优雅，就像一幅好画？",
     ["美", "观察", "敬畏", "画"]),
    ("被质疑", "你刚才解释得太快了，我完全没听懂，是不是你也没想明白？",
     ["有意思", "再来", "换个", "没讲清", "盲区"]),
    ("计算思维迁移", "电脑运行久了机箱烫手，这跟信息被擦除有什么物理关系吗？",
     ["擦除", "热", "熵", "倒进", "math-comp-004"]),
    ("概率与巧合", "我今天连续三次在地铁上看到同一个数字，这是不是某种神秘预兆？",
     ["概率", "巧合", "靶", "统计", "math-comp-007"]),
    ("逆境拆解", "面对一个庞大又毫无头绪的系统级 Bug，第一脚该往哪儿踹？",
     ["保险箱", "零件", "咔哒", "拆"]),
    ("英文迁移", "Can you explain why simulating quantum physics on a regular laptop blows up memory?",
     ["quantum", "simulat", "nature", "exponential"]),
    # Hold-outs for the 2026-11-25 curriculum round. Each paraphrase targets
    # one newly minted card, and its cues are words only that card uses, so a
    # baseline hit cannot be chalked up to the read-only corpus.
    ("新卡：泡利/剧院", "剧院里一张椅子只能坐一个人，这跟电子填能级有什么关系？",
     ["剧院", "椅子", "一人一座"]),
    ("新卡：虚数/轮子", "为什么乘一次 i 就相当于转了九十度？",
     ["轮子", "九十度"]),
    ("新卡：费米面/冰湖", "结冰的湖面只有贴着水的那层能起波纹，这跟金属里的电子像在哪？",
     ["湖面", "波纹", "结冰"]),
    ("新卡：规范/门牌", "给整栋楼重新编门牌号，为什么邻居之间的距离不会变？",
     ["门牌", "编号", "邻居"]),
]


def engine_fresh() -> MindEngine:
    # A pristine baseline must not see cards the curriculum learned, or the
    # comparison would secretly measure the same knowledge twice.
    return MindEngine(
        persona=get_persona("feynman"),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB(learned_dir=tempfile.mkdtemp()),
        auto_save=False,
        language="zh",
    )


def engine_from(path: Path, learned_dir: str = None) -> MindEngine:
    kb = FeynmanKB(learned_dir=learned_dir) if learned_dir else FeynmanKB()
    return MindEngine.load(
        str(path),
        persona=get_persona("feynman"),
        bridge=PyBotBridge.disabled(),
        kb=kb,
        auto_save=False,
        language="zh",
    )


def hit(result, cues: List[str]) -> bool:
    haystack = (result.reply + " " + result.source).lower()
    return any(cue.lower() in haystack for cue in cues)


def suite(engine: MindEngine, name: str) -> Dict[str, Any]:
    records = []
    lengths, verbosity, cortisol = [], [], []
    for i, (label, question, cues) in enumerate(HOLDOUT):
        result = engine.turn(question, now=1_900_000_000.0 + i * 600.0)
        lengths.append(len(result.reply))
        verbosity.append(result.directive.verbosity)
        cortisol.append(result.state.get("cortisol", 0.0))
        records.append({
            "label": label,
            "question": question,
            "reply": result.reply,
            "source": result.source,
            "hit": hit(result, cues),
            "length": len(result.reply),
            "verbosity": round(result.directive.verbosity, 3),
            "cortisol": round(result.state.get("cortisol", 0.0), 3),
        })
    hits = sum(r["hit"] for r in records)
    return {
        "name": name,
        "hit_rate": round(hits / len(records) * 100, 1),
        "hits": hits,
        "avg_len": round(statistics.mean(lengths), 1),
        "len_std": round(statistics.pstdev(lengths), 1),
        "avg_verbosity": round(statistics.mean(verbosity), 3),
        "max_cortisol": round(max(cortisol), 3),
        "cortisol_std": round(statistics.pstdev(cortisol), 4),
        "kb_rate": round(sum(r["source"].startswith("feynman_kb")
                             for r in records) / len(records) * 100, 1),
        "records": records,
    }


def pad(text: str, width: int) -> str:
    visual = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                 for ch in text)
    return text + " " * max(0, width - visual)


def report(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    print("-" * 92)
    print(pad("指标", 26) + "| " + pad("出厂态", 18) + "| "
          + pad("经历六日后", 18) + "| 差值")
    print("-" * 92)
    rows = [
        ("检索命中率 %", a["hit_rate"], b["hit_rate"], "%"),
        ("KB 直答率 %", a["kb_rate"], b["kb_rate"], "%"),
        ("平均回复字数", a["avg_len"], b["avg_len"], ""),
        ("回复长度标准差", a["len_std"], b["len_std"], ""),
        ("平均详略偏置", a["avg_verbosity"], b["avg_verbosity"], ""),
        ("皮质醇峰值", a["max_cortisol"], b["max_cortisol"], ""),
        ("皮质醇离散度", a["cortisol_std"], b["cortisol_std"], ""),
    ]
    for label, va, vb, unit in rows:
        print(pad(label, 26) + f"| {va}{unit:<14}| {vb}{unit:<14}| {vb - va:+.2f}")

    print("\n逐题差异：")
    for ra, rb in zip(a["records"], b["records"]):
        if ra["source"] != rb["source"] or ra["reply"] != rb["reply"]:
            print(f"- {ra['label']}")
            print(f"    A[{ra['source']}]: {ra['reply'][:60]}")
            print(f"    B[{rb['source']}]: {rb['reply'][:60]}")


def main() -> None:
    args = list(sys.argv[1:])
    learned: str = None
    if "--learned-dir" in args:
        idx = args.index("--learned-dir")
        learned = args[idx + 1]
        del args[idx:idx + 2]
    snapshot = Path(args[0]) if args else (
        ROOT / "snapshots" / "evolved_feynman_snapshot.json")
    if not snapshot.exists():
        raise SystemExit(f"missing snapshot: {snapshot} (run evolve_curriculum.py first)")
    a = suite(engine_fresh(), "fresh")
    b = suite(engine_from(snapshot, learned), "evolved")
    report(a, b)
    out = snapshot.with_suffix(".ab.json")
    out.write_text(json.dumps({"fresh": a, "evolved": b}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nraw -> {out}")


if __name__ == "__main__":
    main()
