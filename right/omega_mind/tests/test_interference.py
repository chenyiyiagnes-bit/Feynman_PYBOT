"""Proactive interference: the last subject is still on the cube.

A new topic does not arrive into a clean mind. When the cube is still loaded
from the previous subject, the answer gets the same confidence hedge a
guarded posture would — one style sentence, no change to which card answered
or what it says. A clean cube behaves exactly as before.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.engine import hedge_applies
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


class TestHedgeRule(unittest.TestCase):
    def test_a_clean_mind_does_not_hedge(self):
        self.assertFalse(hedge_applies(0.01, "answer"))
        self.assertFalse(hedge_applies(0.01, "explore"))

    def test_a_careful_mind_hedges_inside_the_line_only(self):
        self.assertTrue(hedge_applies(0.04, "careful"))
        self.assertFalse(hedge_applies(0.08, "careful"))

    def test_interference_widens_the_line(self):
        self.assertTrue(hedge_applies(0.09, "answer", interference=True))
        self.assertTrue(hedge_applies(0.04, "answer", interference=True))
        self.assertFalse(hedge_applies(0.12, "answer", interference=True))

    def test_no_margin_never_hedges(self):
        self.assertFalse(hedge_applies(None, "careful", interference=True))


class TestInterferenceInTheEngine(unittest.TestCase):
    def _mind(self):
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False)

    def test_a_loaded_cube_hedges_a_new_subject(self):
        mind = self._mind()
        mind.cube.apply("R U F L")
        mind._last_card = "thermo-temp-heat"
        result = mind.turn("熵是什么", now=1000.0)
        self.assertTrue(result.source.startswith("feynman_kb"))
        self.assertTrue(result.retrieval.get("hedge"))

    def test_a_clean_cube_does_not_hedge(self):
        mind = self._mind()
        mind._last_card = "thermo-temp-heat"
        result = mind.turn("熵是什么", now=1000.0)
        self.assertFalse(result.retrieval.get("hedge", False))

    def test_the_same_subject_again_is_not_a_shift(self):
        mind = self._mind()
        mind.cube.apply("R U F L")
        mind.turn("熵是什么", now=1000.0)
        again = mind.turn("熵是什么", now=1001.0)
        self.assertEqual(again.source.split(":", 1)[1],
                         "thermo-entropy-disorder")

    def test_the_hedge_respects_its_cooldown(self):
        mind = self._mind()
        mind.cube.apply("R U F L")
        mind._last_card = "thermo-temp-heat"
        first = mind.turn("熵是什么", now=1000.0)
        self.assertTrue(first.retrieval.get("hedge"))
        mind._last_card = "opt-012"
        second = mind.turn("温度是什么", now=1001.0)
        self.assertFalse(second.retrieval.get("hedge", False))


if __name__ == "__main__":
    unittest.main()
