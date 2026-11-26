"""Measure the learning layer on its own, not only the 10-question A/B.

Three sets, each aimed at a failure the big A/B cannot see:

  * hard negatives — praise and address that use 「像」 but are not
    comparisons; the correct behaviour is to mint nothing;
  * paraphrase — one taught metaphor, several ways of asking about it later;
    the correct behaviour is to reach the same card every time;
  * teach-then-test — a batch of metaphors taught in one session, then
    queried the way a forgetful user would.

Every run uses a fresh temporary learned directory; the shipped one is never
touched. Exit code is non-zero when any set falls below its floor.

Usage:
    python3 scripts/evaluate_learning.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge

HARD_NEGATIVES = [
    "费曼先生，像你这么聪明的人真是太少见了！",
    "像我们这种普通人哪里听得懂量子电动力学啊。",
    "像你这么大年纪还这么有好奇心，真让人佩服。",
    "我觉得像他那样的老师才叫好老师。",
    "原来太阳光藏在树叶里，这个比喻太形象了哈哈！",
    "这张图像的分辨率好像有点低。",
]

PARAPHRASES = [
    "哈希指针为什么能防篡改？",
    "蜡封信件那个比喻再讲讲。",
    "改一封就得重盖章，这说的是什么？",
    "首尾相扣的蜡封信件怎么理解数据链？",
    "为什么说改一封后面全得重盖章？",
]

METAPHOR = ("哈希指针可以想成一串首尾相扣的蜡封信件，"
            "改一封后面全得重盖章。")

TEACH_THEN_TEST = [
    ("电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。",
     "走廊里推一下，为什么整条队伍会一起动？"),
    ("摩擦力就像地上铺了一层细沙，推着箱子走总要先碾过沙粒。",
     "地面上的细沙和推箱子有什么关系？"),
    ("光合作用就像一个把阳光存进糖里的存钱罐，叶绿素是罐口。",
     "叶绿素为什么被叫作罐口？"),
    ("引力就像一张绷紧的床单，太阳把中间压出一个坑。",
     "床单上的坑和行星绕太阳有什么关系？"),
    ("记忆就像往抽屉里塞纸条，塞得越乱下一次越难抽出来。",
     "抽屉塞乱了为什么就找不到纸条？"),
]


def engine(directory: str, persona_id: str = "feynman-zh") -> MindEngine:
    return MindEngine(
        persona=get_persona(persona_id),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB(learned_dir=directory),
        auto_save=False,
        language="zh",
    )


def run_hard_negatives(persona_id: str = "feynman-zh") -> float:
    mind = engine(tempfile.mkdtemp(), persona_id)
    for i, text in enumerate(HARD_NEGATIVES):
        mind.turn(text, now=1_000.0 + i * 60)
    minted = len(mind.kb.learned.cards)
    print(f"[硬负例] 说出 {len(HARD_NEGATIVES)} 句赞美/指代，铸卡 {minted} 张")
    for card in mind.kb.learned.cards:
        print("   误铸：", card.get("body"))
    return 1.0 if minted == 0 else 0.0


def run_paraphrase(persona_id: str = "feynman-zh") -> float:
    mind = engine(tempfile.mkdtemp(), persona_id)
    mind.turn(METAPHOR, now=1_000.0)
    hits, card_id = 0, None
    for i, question in enumerate(PARAPHRASES):
        result = mind.turn(question, now=1_100.0 + i * 60)
        ok = (result.source.startswith(mind.kb.source_prefix + ":")
              and "learned-" in result.source)
        if ok:
            hits += 1
            card_id = card_id or result.source.split(":", 1)[-1]
        print(f"    {'✓' if ok else '✗'} {question}  [{result.source}]")
    rate = hits / len(PARAPHRASES)
    print(f"[同卡换问法] {hits}/{len(PARAPHRASES)} 命中"
          f"（同一张卡：{card_id or '无'}）")
    return rate


def run_teach_then_test(persona_id: str = "feynman-zh") -> float:
    mind = engine(tempfile.mkdtemp(), persona_id)
    hits = 0
    for i, (metaphor, question) in enumerate(TEACH_THEN_TEST):
        mind.turn(metaphor, now=1_000.0 + i * 120)
        result = mind.turn(question, now=1_060.0 + i * 120)
        ok = "learned-" in result.source
        hits += ok
        print(f"    {'✓' if ok else '✗'} {question}  [{result.source}]")
    rate = hits / len(TEACH_THEN_TEST)
    print(f"[教会再问] {hits}/{len(TEACH_THEN_TEST)} 命中，"
          f"库中共 {len(mind.kb.learned.cards)} 张卡")
    return rate


USAGE = """usage: evaluate_learning.py [--persona ID]

Measure the learning layer on its own (hard negatives, paraphrase,
teach-then-test). The question sets stay Chinese/Feynman's; --persona only
selects which mind is measured.
"""


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print(USAGE)
        return 0
    persona_id = "feynman-zh"
    if "--persona" in args:
        idx = args.index("--persona")
        if idx + 1 >= len(args):
            raise SystemExit(f"--persona needs a value\n{USAGE}")
        persona_id = args[idx + 1]
    print("学习层评测（每次使用全新临时学习目录）\n")
    negative = run_hard_negatives(persona_id)
    print()
    paraphrase = run_paraphrase(persona_id)
    print()
    teach = run_teach_then_test(persona_id)
    print()
    ok = negative == 1.0 and paraphrase >= 0.8 and teach >= 0.8
    print(f"硬负例不铸卡={negative == 1.0}  同卡换问法={paraphrase:.0%}  "
          f"教会再问={teach:.0%}  =>  {'通过' if ok else '未达标'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
