"""Retrieval inhibition: the card just spoken is demoted, not banned.

Humans are slower to return to a stimulus they have just attended to
(inhibition of return). The mechanical form: a card answered within the last
few turns loses a small fraction of its score at ranking time, so a near-tie
resolves to the other card instead of circling. Two invariants keep it
honest: it can never turn a confident answer into a miss, and a question
whose only card is the inhibited one still gets that card.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.engine import INHIBIT_TURNS, MindEngine
from omega_mind.feynman_kb import INHIBIT_FACTOR, FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


class TestInhibitionAtRetrieval(unittest.TestCase):
    def setUp(self):
        self.kb = real_kb()

    def _near_tie(self):
        """A real query whose top two cards are a few hundredths apart."""
        for query in ("熵是什么", "温度是什么", "光是什么"):
            found = self.kb.lookup(query, limit=3)
            if len(found.alternatives) >= 2 and found.margin is not None:
                return query, found
        self.skipTest("no near-tie query on this corpus")
        return "", None

    def test_a_recently_spoken_card_loses_the_near_tie(self):
        query, found = self._near_tie()
        top = found.source.split(":", 1)[1]
        again = self.kb.lookup(query, limit=3, inhibit=[top])
        self.assertNotEqual(again.source, found.source)
        self.assertEqual(again.inhibited, (top,))

    def test_the_demotion_is_small_and_audited(self):
        query, found = self._near_tie()
        top = found.source.split(":", 1)[1]
        again = self.kb.lookup(query, limit=3, inhibit=[top])
        demoted = dict(again.alternatives)[top]
        original = dict(found.alternatives)[top]
        self.assertAlmostEqual(demoted, original * INHIBIT_FACTOR, places=4)

    def test_inhibition_never_turns_an_answer_into_a_miss(self):
        # This query has exactly one candidate; inhibiting it must not
        # produce no_match, because there is nothing else to say.
        found = self.kb.lookup("彩虹为什么是弯的", limit=3)
        self.assertEqual(found.considered, 1)
        card = found.source.split(":", 1)[1]
        again = self.kb.lookup("彩虹为什么是弯的", limit=3, inhibit=[card])
        self.assertEqual(again.source, found.source)
        self.assertEqual(again.inhibited, ())

    def test_no_inhibition_changes_nothing(self):
        query, found = self._near_tie()
        again = self.kb.lookup(query, limit=3, inhibit=[])
        self.assertEqual(again.source, found.source)
        self.assertEqual(again.score, found.score)


class TestInhibitionWindowInTheEngine(unittest.TestCase):
    def _mind(self):
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False)

    def test_a_spoken_card_is_inhibited_for_a_few_turns_then_returns(self):
        mind = self._mind()
        mind.turn("熵是什么", now=1000.0)
        card = mind._last_card
        self.assertIn(card, mind._inhibit_set("别的题目"))
        for step in range(1, INHIBIT_TURNS - 1):
            mind.turn("谢谢", now=1000.0 + step)
            self.assertIn(card, mind._inhibit_set("别的题目"))
        for step in range(INHIBIT_TURNS - 1, INHIBIT_TURNS + 1):
            mind.turn("谢谢", now=1000.0 + step)
        self.assertNotIn(card, mind._inhibit_set("别的题目"))

    def test_re_asking_the_same_question_suspends_inhibition(self):
        mind = self._mind()
        mind.turn("熵是什么", now=1000.0)
        self.assertEqual(mind._inhibit_set("熵是什么"), set())

    def test_the_retrieval_audit_names_what_was_demoted(self):
        mind = self._mind()
        mind.turn("熵是什么", now=1000.0)
        again = mind.turn("熵是什么", now=1001.0)
        # The same question still gets answered; the audit says whether the
        # previous card was actually demoted this time.
        self.assertEqual(again.source.startswith("feynman_kb"), True)
        self.assertIn("inhibited", again.retrieval)


if __name__ == "__main__":
    unittest.main()
