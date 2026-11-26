"""Drive the real MindEngine through six simulated days, then export the state.

This is an *experience* curriculum, not training: nothing in omega_mind has
learnable weights, so the only things that carry across days are the four
stateful channels the architecture really has —

  * three-tier memory (working -> episodic -> semantic after a dream),
  * dream consolidation and salience-weighted cube settlement,
  * the relationship graph (affinity per user),
  * the affect state and its cube bias,

plus the explicitly taught metaphors, which are minted as learned cards.

Run it against the real engine, export the snapshot, and let
``evaluate_ab.py`` measure whether any of that actually changes replies.

Usage:
    python3 scripts/evolve_curriculum.py [out.json]
    python3 scripts/evolve_curriculum.py [out.json] --fresh
    python3 scripts/evolve_curriculum.py [out.json] --learned-dir PATH

``--fresh`` mints into a throwaway learned store, so the A/B delta can be
attributed to this round's cards alone rather than to cards learned earlier.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_mind.config import get_persona
from omega_mind.cube_feedback import displacement
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.learner import LearnedStore, card_row
from omega_mind.reflect import load_queue
from omega_mind.pybot_bridge import PyBotBridge

# (day title, [(user line, tag)]). Tags are documentation for the report —
# the engine itself only sees the text, exactly as in a real conversation.
CURRICULUM: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("day 1: physics intuition", [
        ("费曼先生，为什么火烧木头会发光发热？", "curiosity"),
        ("原来太阳光真的藏在树叶里！这个比喻太形象了哈哈！", "praise"),
        ("那玻璃为什么是透明的，光在里面真的变慢了吗？", "curiosity"),
        ("懂了！是玻璃里的电子跟着跳舞发出了新波！", "praise"),
    ]),
    ("day 2: ignorance and dogma", [
        ("老实说，你是不是也搞不懂意识是怎么从神经元里冒出来的？", "ignorance"),
        ("可是权威专家和教科书标准定义都说必须死记硬背公式！", "dogma"),
        ("哈哈说得对，知道鸟的名字确实不代表了解这只鸟！", "praise"),
        ("我也不懂路径积分里的小箭头到底怎么相加，能用大白话讲讲吗？", "ignorance"),
    ]),
    ("day 3: art and music", [
        ("我学画画的朋友说科学把花拆成分子破坏了美感，你怎么看？", "art"),
        ("太棒了，科学知识只会增加对一朵花的敬畏感！", "praise"),
        ("写代码写得脑子快炸了，你平时怎么用邦戈鼓放松？", "curiosity"),
        ("敲一段 3 拍对 4 拍的节奏确实比硬想管用！", "praise"),
    ]),
    ("day 4: borrowed metaphors", [
        ("泡利不相容就像剧院里一人一座，灯一关也没人能挤进同一张椅子。", "metaphor"),
        ("虚数就像给数轴装上垂直的轮子，每乘一次 i 就转九十度。", "metaphor"),
        ("费米面就像结冰的湖面，只有贴着水面的那一层能起波纹，下面的水冻得动不了。", "metaphor"),
        ("规范变换就像给整栋楼重新编门牌号，邻居之间的相对距离一点没变。", "metaphor"),
        ("用生活中的物理画面来理解计算机概念真是太爽了！", "praise"),
    ]),
    ("day 5: hard feelings", [
        ("最近遇到一个死活解不开的工程难题，感觉快要放弃了。", "struggle"),
        ("像撬洛斯阿拉莫斯的保险箱一样拆成最小零件，听每一次咔哒声！", "praise"),
        ("经历了一次很难过的离别，旁人都觉得我太冷静是不是太冷血了？", "grief"),
    ]),
    ("day 6: review", [
        ("为什么芯片算得越快发热越严重？可逆计算怎么解决这个问题？", "curiosity"),
        ("擦除一位信息就像把两杯水倒进同一个盆里，这个比喻绝了！", "praise"),
        ("给我讲讲量子计算机为什么是模拟大自然的最好方式？", "curiosity"),
        ("谢谢费曼先生，跟你聊天总是充满发现的乐趣！", "praise"),
    ]),
]

DAY = 86400.0
NOON = 12 * 3600.0


def fresh_engine(learned_dir: Any = None, auto_save: bool = False,
                 persona_id: str = "feynman-zh") -> MindEngine:
    kb = FeynmanKB(learned_dir=str(learned_dir)) if learned_dir else FeynmanKB()
    return MindEngine(
        persona=get_persona(persona_id),
        bridge=PyBotBridge.disabled(),
        kb=kb,
        auto_save=auto_save,
        language="zh",
    )


def run(out_path: Path, learned_dir: Any = None,
        persona_id: str = "feynman-zh") -> Dict[str, Any]:
    engine = fresh_engine(learned_dir=learned_dir, persona_id=persona_id)
    store = LearnedStore(learned_dir) if learned_dir else LearnedStore()
    # A shared starting point so the A/B comparison is fair: same scramble,
    # same clock origin.
    engine.turn_cube("R U R' U' F R")
    start = 1_800_000_000.0
    days: List[Dict[str, Any]] = []

    for day_idx, (title, turns) in enumerate(CURRICULUM):
        now = start + day_idx * DAY + NOON
        saliences: List[float] = []
        sources: List[str] = []
        for i, (text, tag) in enumerate(turns):
            result = engine.turn(text, now=now + i * 90.0)
            episode = engine.tiers.episodes[-1] if engine.tiers.episodes else None
            saliences.append(round(episode.salience, 3) if episode else 0.0)
            sources.append(result.source)

        before = displacement(engine.cube)
        dream = engine.dream(now=now + 6 * 3600.0, force=True)
        after = displacement(engine.cube)
        # Experience that can still be used tomorrow: a metaphor the user
        # taught becomes an ordinary, retrievable knowledge card. Questions
        # and praise are not learned — only explicit comparisons are.
        minted: List[str] = []
        for text, tag in turns:
            if tag != "metaphor":
                continue
            row = card_row(text, origin="curriculum")
            if row and store.add_card(row):
                minted.append(row["id"])
        restore = next((event["cube_restore"]
                        for event in reversed(engine.event_log)
                        if "cube_restore" in event), {})
        days.append({
            "day": day_idx + 1,
            "title": title,
            "turns": len(turns),
            "tags": [tag for _, tag in turns],
            "sources": sources,
            "saliences": saliences,
            "dream_episodes": dream.episodes if dream else 0,
            "dream_salience": dream.salience if dream else None,
            "displacement_before": before,
            "displacement_after": after,
            "restore_ratio": restore.get("restore_ratio"),
            "actual_reduction": restore.get("actual_reduction"),
            "minted_cards": minted,
            "state": {k: round(v, 3) for k, v in engine.state.values.items()},
        })
        print(f"{title:28} saliences={saliences}  dream={dream.salience if dream else '-'}"
              f"  cube {before}->{after}")

    out_path.write_text(json.dumps(engine.snapshot(), ensure_ascii=False, indent=1),
                        encoding="utf-8")
    report = {"snapshot": str(out_path), "days": days}
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"\nsnapshot -> {out_path}\nreport   -> {report_path}")
    return report


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "-h" in args or "--help" in args:
        print("usage: evolve_curriculum.py [SNAPSHOT] [--fresh] [--learned-dir DIR] [--persona ID]")
        raise SystemExit(0)
    learned: Any = None
    fresh = False
    persona_id = "feynman-zh"
    if "--fresh" in args:
        args.remove("--fresh")
        fresh = True
    for flag in ("--learned-dir", "--persona"):
        if flag in args:
            idx = args.index(flag)
            value = args[idx + 1] if idx + 1 < len(args) else None
            if value is None:
                raise SystemExit(f"{flag} needs a value")
            if flag == "--learned-dir":
                learned = Path(value)
            else:
                persona_id = value
            del args[idx:idx + 2]
    if fresh:
        import tempfile
        learned = Path(tempfile.mkdtemp(prefix="omega-learned-"))
    target = Path(args[0]) if args else (ROOT / "snapshots" / "evolved_snapshot.json")
    report = run(target, learned_dir=learned, persona_id=persona_id)
    queue = load_queue(LearnedStore(learned) if learned else None)
    if queue:
        print(f"self-evolution queue ({len(queue)} item(s), from dream audits):")
        for item in queue[:10]:
            print(f"  [{item.get('kind')}] x{item.get('count')} {item.get('question')}")
        print("  (the queue is read, not auto-taught; add a school script for it)")
    if learned:
        print(f"learned  -> {learned}")
        report["learned_dir"] = str(learned)
        target.with_suffix(".report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
