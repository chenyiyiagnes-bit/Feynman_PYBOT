"""Curiosity: the right brain asks a question of its own, from real data.

A curious mind does not only answer; it follows a thread and asks. This is
the one place the engine adds a question, and it may only name what already
exists:

* a genuine near-tie — the top two cards are within a few hundredths, so it
  asks which of the two was meant, using their real titles;
* a card with a confirmed premise edge below it — it invites the user one
  step down the why-chain that already exists.

No question is ever generated from a card that was fabricated, no card id is
spoken, and a question is only asked in the ``explore`` posture, at most once
per ``QUESTION_COOLDOWN`` turns, and never in the same answer as an aside.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import engine as engine_module
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB, Lookup, _question_for
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


def _found(**over):
    base = dict(text="answer", source="feynman_kb:opt-009", score=1.0,
                margin=0.30, alternatives=(("opt-009", 1.0), ("cm-001", 0.70)))
    base.update(over)
    return Lookup(**base)


TITLES = {"opt-009": "彩虹", "cm-001": "超导"}
EN_TITLES = {"opt-009": "the rainbow", "cm-001": "superconductivity"}


class TestQuestionChoice(unittest.TestCase):
    def test_a_near_tie_asks_which_card_was_meant(self):
        found = _found(margin=0.01)
        line = _question_for(found, TITLES, "zh", premise_ok=True)
        self.assertIn("彩虹", line)
        self.assertIn("超导", line)

    def test_a_clear_winner_with_a_premise_invites_the_why_chain(self):
        found = _found(margin=0.30)
        line = _question_for(found, TITLES, "zh", premise_ok=True)
        self.assertIn("为什么", line)
        self.assertNotIn("超导", line)

    def test_no_tie_and_no_premise_means_no_question(self):
        found = _found(margin=0.30)
        self.assertIsNone(_question_for(found, TITLES, "zh", premise_ok=False))

    def test_a_missing_title_is_never_replaced_by_a_card_id(self):
        found = _found(margin=0.01)
        self.assertIsNone(_question_for(found, {"opt-009": "彩虹"}, "zh",
                                        premise_ok=True))

    def test_the_english_question_speaks_no_chinese(self):
        found = _found(margin=0.01)
        line = _question_for(found, EN_TITLES, "en", premise_ok=True)
        self.assertIn("the rainbow", line)
        self.assertIn("superconductivity", line)
        self.assertNotRegex(line, r"[一-鿿]")
        why = _question_for(_found(margin=0.30), TITLES, "en", premise_ok=True)
        self.assertNotRegex(why, r"[一-鿿]")

    def test_an_english_question_refuses_chinese_titles(self):
        found = _found(margin=0.01)
        self.assertIsNone(_question_for(found, TITLES, "en", premise_ok=True))

    def test_an_empty_answer_gets_no_question(self):
        self.assertIsNone(_question_for(_found(text=None, margin=0.01),
                                        TITLES, "zh", premise_ok=True))


class TestQuestionInLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_lookup_is_quiet_unless_curiosity_was_asked_for(self):
        found = self.kb.lookup("彩虹为什么是弯的")
        self.assertIsNone(found.question)

    def test_a_card_with_a_premise_below_it_offers_the_step(self):
        found = self.kb.lookup("彩虹为什么是弯的", curiosity=True)
        self.assertTrue(found.question)
        self.assertIn("为什么", found.question)

    def test_the_english_branch_asks_in_english(self):
        found = self.kb.lookup("Why is a rainbow curved?", language="en",
                               curiosity=True)
        self.assertTrue(found.question)
        self.assertNotRegex(found.question, r"[一-鿿]")


class TestQuestionInTheEngine(unittest.TestCase):
    def _mind(self, language="zh"):
        kb = real_kb()
        mind = MindEngine(
            persona=__import__("omega_mind.config", fromlist=["get_persona"]
                               ).get_persona("generic"),
            store_path=None, clock=lambda: 1000.0,
            bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False, language=language)
        return mind

    def _curious(self, mind):
        # Curiosity is the only posture that may ask; set it directly so the
        # test does not depend on which praise wording raises dopamine.
        mind.state.set("dopamine", 0.90)

    def test_a_curious_turn_can_ask_one_question(self):
        mind = self._mind()
        self._curious(mind)
        result = mind.turn("彩虹为什么是弯的", now=1000.0)
        self.assertEqual(result.left_brain["posture"], "explore")
        self.assertTrue(result.retrieval.get("question"))
        # Exactly one trailing remark: the question speaks only when no
        # association or scale aside already took the slot.
        if result.retrieval.get("association"):
            self.assertNotIn(result.retrieval["question"], result.reply)
        else:
            self.assertIn(result.retrieval["question"], result.reply)

    def test_the_cooldown_keeps_the_next_turn_quiet(self):
        mind = self._mind()
        self._curious(mind)
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        self.assertTrue(first.retrieval.get("question"))
        self._curious(mind)
        second = mind.turn("彩虹为什么是弯的", now=1001.0)
        self.assertTrue(second.retrieval.get("question"))
        self.assertNotIn(second.retrieval["question"], second.reply)

    def test_an_aside_wins_the_single_trailing_slot(self):
        mind = self._mind()
        self._curious(mind)
        result = mind.turn("彩虹为什么是弯的", now=1000.0)
        # This card carries a confirmed association remark; when one is
        # spoken the question yields rather than stacking a second aside.
        if result.retrieval.get("association"):
            self.assertNotIn(result.retrieval["question"], result.reply)

    def test_a_calm_turn_asks_nothing(self):
        mind = self._mind()
        result = mind.turn("彩虹为什么是弯的", now=1000.0)
        self.assertNotEqual(result.left_brain["posture"], "explore")
        self.assertIsNone(result.retrieval.get("question"))

    def test_the_cooldown_survives_a_snapshot(self):
        mind = self._mind()
        self._curious(mind)
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        if not first.retrieval.get("question"):
            self.skipTest("question not asked this turn")
        clone = MindEngine.load_from_dict(
            mind.snapshot(), bridge=PyBotBridge.disabled(), kb=mind.kb)
        self._curious(clone)
        second = clone.turn("彩虹为什么是弯的", now=1001.0)
        self.assertNotIn(second.retrieval.get("question") or "", second.reply)


if __name__ == "__main__":
    unittest.main()
