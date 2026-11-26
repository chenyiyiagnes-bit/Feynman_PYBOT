"""Retrieval layer and teach-back challenge tests.

The corpus lives outside the repo; these tests skip themselves when it is not
on disk instead of failing on another machine.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.challenge import ChallengeSession
from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB()
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_corpus_loaded(self):
        self.assertGreaterEqual(len(self.kb.docs), 200)
        self.assertGreaterEqual(len(self.kb.teachable_units()), 4)

    def test_heat_cooling_finds_relaxation_unit(self):
        reply, source = self.kb.answer("热水为什么会自己变凉")
        self.assertTrue(reply)
        self.assertEqual(source, "feynman_kb:U0_RELAX")

    def test_atom_question_finds_lecture_card(self):
        reply, source = self.kb.answer("原子是什么")
        self.assertTrue(reply)
        self.assertEqual(source, "feynman_kb:flp-i-01")

    def test_irrelevant_question_abstains(self):
        reply, source = self.kb.answer("今天天气怎么样")
        self.assertIsNone(reply)
        self.assertEqual(source, "feynman_kb:no_match")

    def test_answer_is_composed_from_card_fields(self):
        reply, _ = self.kb.answer("哈勃半径和史瓦西半径")
        self.assertIn("哈勃", reply)
        self.assertIn("半径", reply)

    def test_disabled_kb_answers_nothing(self):
        kb = FeynmanKB.disabled()
        self.assertFalse(kb.available())
        self.assertEqual(kb.answer("原子"), (None, "feynman_kb:no_match"))


class TestEngineWithCards(unittest.TestCase):
    def test_cards_take_priority_over_bridge(self):
        kb = real_kb()
        e = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=kb,
            auto_save=False,
        )
        r = e.turn("热水为什么会自己变凉", now=1000.0)
        self.assertEqual(r.source, "feynman_kb:U0_RELAX")
        self.assertIn("温度" if "温度" in r.base_reply else "室温", r.base_reply)


class TestChallenge(unittest.TestCase):
    def setUp(self):
        self.kb = real_kb()
        self.e = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=self.kb,
            auto_save=False,
        )
        self.session = ChallengeSession(self.kb, self.e)
        self.session.next_unit()

    def test_question_is_asked(self):
        self.assertIsNotNone(self.session.unit)
        self.assertTrue(self.session.unit.unit.get("entry_question"))

    def test_good_answer_is_strong(self):
        before = self.e.state.get("dopamine")
        g = self.session.submit(
            "偏离平衡的比例按同一个时间常数缩小，所以越接近室温凉得越慢，半衰期和偏离多大无关"
        )
        self.assertEqual(g.verdict, "strong")
        self.assertGreater(self.e.state.get("dopamine"), before)

    def test_shrug_is_weak_and_asks_again(self):
        g = self.session.submit("不知道")
        self.assertEqual(g.verdict, "weak")
        self.assertIn("看到了什么", g.feedback)

    def test_second_failure_reveals_the_invariant(self):
        self.session.submit("不知道")
        g = self.session.submit("还是不知道")
        self.assertIsNotNone(g.revealed)
        self.assertIn("不变量" if "不变量" in g.revealed else "比例", g.revealed)

    def test_wrong_answer_costs_no_cortisol(self):
        before = self.e.state.get("cortisol")
        self.session.submit("不知道")
        self.assertLessEqual(self.e.state.get("cortisol"), before + 1e-9)

    def test_units_cycle(self):
        first = self.session.unit.id
        self.session.unit = None
        self.session.next_unit()
        self.assertNotEqual(first, self.session.unit.id)


if __name__ == "__main__":
    unittest.main()
