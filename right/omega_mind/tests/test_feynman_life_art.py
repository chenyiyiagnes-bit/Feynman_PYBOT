"""The life-and-art card pack, and the persona switches it depends on.

The pack lives in the feynman-open knowledge directory (Chinese source of
truth) plus the English overlay under ``omega_mind/data/kb_en``. These tests
skip when the knowledge directory is not on this machine, like the other KB
tests do.
"""

from __future__ import annotations

import re
import unittest

from omega_mind.appraisal import AppraisalEngine
from omega_mind.config import get_persona
from omega_mind.feynman_kb import DEFAULT_DIRS, FeynmanKB

CJK = re.compile(r"[一-鿿]")

NEW_IDS = {f"life-art-00{i}" for i in range(1, 9)}

# query -> acceptable answer ids. Two of these have older cards that answer
# the same fact (Arlene's death, safe-cracking); either is a correct reply,
# so the test pins the topic, not one particular card.
ZH_QUERIES = {
    "你喜欢画画吗": {"life-art-001"},
    "科学会不会毁掉一朵花的美": {"life-art-002"},
    "你会打邦戈鼓吗": {"life-art-003"},
    "不懂装懂怎么办": {"life-art-004"},
    "发现的乐趣是什么": {"life-art-005"},
    "别人期待你成为什么样的人": {"life-art-006"},
    "你会开锁吗": {"life-art-007", "bio-012", "persona-safecrack-001"},
    "阿琳去世后你怎么过": {"life-art-008", "bio-007"},
}

EN_QUERIES = {
    "Do you like painting?": {"life-art-001"},
    "Does science destroy the beauty of a flower?": {"life-art-002"},
    "Do you play the bongos?": {"life-art-003"},
    "What do you think about pretending to know?": {"life-art-004"},
    "What is the pleasure of finding things out?": {"life-art-005"},
    "Do you care what other people expect of you?": {"life-art-006"},
    "Can you pick locks?": {"life-art-007", "bio-012", "persona-safecrack-001"},
    "How did you cope with Arlene's death?": {"life-art-008"},
}


def _kb_or_skip(language: str = "zh") -> FeynmanKB:
    if not any(path.is_dir() and any(path.glob("*.jsonl"))
               for path in DEFAULT_DIRS):
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    kb = FeynmanKB(language=language)
    if not kb.available() or kb.language != language:
        raise unittest.SkipTest(f"{language!r} knowledge package did not load")
    return kb


class TestLifeArtCardsLoad(unittest.TestCase):
    def setUp(self):
        self.kb = _kb_or_skip("zh")
        self.en = _kb_or_skip("en")

    def test_every_card_loads_in_both_languages(self):
        zh = {d.id for d in self.kb.docs}
        en = {d.id for d in self.en.docs}
        self.assertTrue(NEW_IDS <= zh, f"missing zh: {NEW_IDS - zh}")
        self.assertTrue(NEW_IDS <= en, f"missing en: {NEW_IDS - en}")

    def test_english_overlay_has_no_chinese(self):
        by_id = {d.id: d for d in self.en.docs}
        for cid in NEW_IDS:
            doc = by_id[cid]
            self.assertFalse(CJK.search(doc.answer),
                             f"{cid} leaked Chinese into English mode")
            self.assertFalse(CJK.search(doc.title),
                             f"{cid} has a Chinese title in English mode")

    def test_zh_and_en_pairs_stay_aligned(self):
        # The project rule is one-to-one pairing: no card may exist on only
        # one side, otherwise the two language modes drift apart.
        zh = {d.id for d in self.kb.docs if d.id in NEW_IDS}
        en = {d.id for d in self.en.docs if d.id in NEW_IDS}
        self.assertEqual(zh, en)


class TestLifeArtRetrieval(unittest.TestCase):
    def setUp(self):
        self.kb = _kb_or_skip("zh")
        self.en = _kb_or_skip("en")

    def test_chinese_topics_reach_their_card(self):
        for query, expected in ZH_QUERIES.items():
            with self.subTest(query=query):
                hits = self.kb.search(query, limit=3)
                self.assertTrue(hits, f"no hit for {query}")
                score, doc = hits[0]
                self.assertGreaterEqual(score, self.kb.MIN_SCORE)
                self.assertIn(doc.id, expected)

    def test_english_topics_reach_their_card(self):
        for query, expected in EN_QUERIES.items():
            with self.subTest(query=query):
                hits = self.en.search(query, limit=3)
                self.assertTrue(hits, f"no hit for {query}")
                score, doc = hits[0]
                self.assertGreaterEqual(score, self.kb.MIN_SCORE)
                self.assertIn(doc.id, expected)


class TestFeynmanPersonaSwitches(unittest.TestCase):
    def test_curiosity_baseline_and_low_stress(self):
        p = get_persona("feynman-zh")
        self.assertGreaterEqual(p.baseline("dopamine"), 0.60)
        self.assertLessEqual(p.baseline("cortisol"), 0.12)
        self.assertIn("corner_shame", p.disable_cubies)
        self.assertTrue(p.correction_as_curiosity)

    def test_admitting_ignorance_reads_as_invitation(self):
        engine = AppraisalEngine(get_persona("feynman-zh"))
        acts = engine.evaluate("我不懂这个，你能再讲一遍吗", turn=1)
        by_name = {a.name: a for a in acts}
        self.assertIn("core_intuition", by_name)
        self.assertGreater(by_name["core_intuition"].effects["dopamine"], 0)
        self.assertNotIn("corner_shame", by_name)

    def test_being_corrected_is_a_reward_not_a_threat(self):
        engine = AppraisalEngine(get_persona("feynman-zh"))
        acts = engine.evaluate("你刚才说错了，这个数是 3 不是 4", turn=1)
        by_name = {a.name: a for a in acts}
        self.assertIn("core_intuition", by_name)
        self.assertGreater(by_name["core_intuition"].effects["dopamine"], 0)
        self.assertNotIn("corner_shame", by_name)


if __name__ == "__main__":
    unittest.main()
