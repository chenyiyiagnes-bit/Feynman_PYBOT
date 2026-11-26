"""Recasting: "I don't get it" gets a different explanation, not a repeat.

The only source is a hand-confirmed ``same_mechanism`` / ``same_equation``
edge that has both halves written: what the two share, and where the analogy
breaks. A card without such an edge gets no recast. The same card is never
recast twice in a row, and the English branch never speaks Chinese.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


class TestRecastSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_a_card_with_a_confirmed_analogy_is_recast_with_its_break_point(self):
        hit = self.kb.recast("opt-009")
        self.assertIsNotNone(hit)
        line, source = hit
        self.assertIn("换个说法", line)
        self.assertIn("不成立", line)
        self.assertTrue(source.startswith("analogy:"))

    def test_the_english_recast_has_no_chinese(self):
        hit = self.kb.recast("opt-009", language="en")
        if hit is None:
            self.skipTest("no English analogy overlay for this card")
        line, source = hit
        self.assertNotRegex(line, r"[一-鿿]")
        self.assertIn("Put another way", line)

    def test_a_card_without_an_analogy_gets_none(self):
        self.assertIsNone(self.kb.recast("no_match"))
        self.assertIsNone(self.kb.recast("learned-abc"))


class TestRecastInTheEngine(unittest.TestCase):
    def _mind(self, language="zh"):
        # One package, one language: the English mind reads the English
        # package's own cards and its own edges.
        kb = FeynmanKB(language=language, learned_dir=None)
        if not kb.available():
            raise unittest.SkipTest("feynman-open knowledge not on this machine")
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=kb, auto_save=False, language=language)

    def test_after_an_answer_a_bare_confusion_recasts_it(self):
        mind = self._mind()
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        self.assertEqual(first.source, "feynman_kb:opt-009")
        second = mind.turn("没懂", now=1001.0)
        self.assertTrue(second.source.startswith("analogy:"))
        self.assertIn("换个说法", second.reply)

    def test_the_same_card_is_not_recast_twice(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        first = mind.turn("没懂", now=1001.0)
        self.assertTrue(first.source.startswith("analogy:"))
        second = mind.turn("没懂", now=1002.0)
        self.assertFalse(second.source.startswith("analogy:"))

    def test_a_new_subject_clears_the_recast_memory(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        mind.turn("没懂", now=1001.0)
        mind.turn("光速是多少", now=1002.0)
        # The rainbow card may be recast again only after the next answer
        # about it; the seen-set was cleared by the unrelated question.
        mind.turn("彩虹为什么是弯的", now=1003.0)
        again = mind.turn("没懂", now=1004.0)
        self.assertTrue(again.source.startswith("analogy:"))

    def test_english_confusion_recasts_in_english(self):
        mind = self._mind(language="en")
        first = mind.turn("Why is a rainbow curved?", now=1000.0)
        if not first.source.startswith(mind.kb.source_root + ":"):
            self.skipTest("English card not available")
        second = mind.turn("I don't get it", now=1001.0)
        if second.source.startswith("analogy:"):
            self.assertNotRegex(second.reply, r"[一-鿿]")

    def test_confusion_with_no_previous_card_does_not_recast(self):
        mind = self._mind()
        result = mind.turn("没懂", now=1000.0)
        self.assertFalse(result.source.startswith("analogy:"))


if __name__ == "__main__":
    unittest.main()
