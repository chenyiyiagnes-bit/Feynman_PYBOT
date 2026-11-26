"""Right-brain state must change how the left brain looks up and speaks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind import left_brain
from omega_mind.left_brain import plan_for
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.state import AffectState


def real_kb(language: str = "zh") -> FeynmanKB:
    kb = FeynmanKB(language=language)
    if not kb.available() or kb.language != language:
        raise unittest.SkipTest(f"{language!r} knowledge package not installed")
    return kb


class TestLeftBrainPolicy(unittest.TestCase):
    def test_default_state_keeps_the_normal_gate(self):
        p = get_persona("generic")
        state = AffectState.from_persona(p, 0.0)
        policy = plan_for(state, p, [])
        self.assertAlmostEqual(policy.min_score, 0.30, places=4)
        self.assertEqual(policy.detail, "normal")
        self.assertEqual(policy.posture, "answer")

    def test_stress_raises_the_gate_and_briefs_the_answer(self):
        p = get_persona("generic")
        state = AffectState.from_persona(p, 0.0)
        state.set("cortisol", 0.85)
        policy = plan_for(state, p, [])
        self.assertGreater(policy.min_score, 0.30)
        self.assertEqual(policy.detail, "brief")
        self.assertEqual(policy.posture, "careful")
        self.assertIn("physics_edge", policy.prefer_kinds)

    def test_curiosity_prefers_teachable_units_and_rich_answers(self):
        p = get_persona("generic")
        state = AffectState.from_persona(p, 0.0)
        state.set("dopamine", 0.90)
        policy = plan_for(state, p, [])
        self.assertLess(policy.min_score, 0.30)
        self.assertEqual(policy.detail, "rich")
        self.assertEqual(policy.posture, "explore")
        self.assertEqual(policy.prefer_kinds, ("teachable_unit", "method_card"))

    def test_fatigue_limits_candidates(self):
        p = get_persona("generic")
        state = AffectState.from_persona(p, 0.0)
        state.set("energy", 0.20)
        policy = plan_for(state, p, [])
        self.assertEqual(policy.limit, 1)
        self.assertEqual(policy.detail, "brief")


class TestLeftBrainRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_policy_gate_can_withhold_a_card(self):
        reply, source = self.kb.answer("热水为什么会自己变凉")
        self.assertEqual(source, "feynman_kb:U0_RELAX")
        withheld, withheld_source = self.kb.answer(
            "热水为什么会自己变凉", min_score=1.0)
        self.assertIsNone(withheld)
        self.assertEqual(withheld_source, "feynman_kb:no_match")

    def test_detail_changes_how_much_of_the_card_is_spoken(self):
        brief, _ = self.kb.answer("热水为什么会自己变凉", detail="brief")
        rich, _ = self.kb.answer("热水为什么会自己变凉", detail="rich")
        self.assertLess(len(brief), len(rich))
        self.assertIn("边界", rich)

    def test_engine_exposes_the_policy_and_shrinks_under_stress(self):
        normal = MindEngine(
            persona=get_persona("feynman"),
            bridge=PyBotBridge.disabled(),
            kb=self.kb,
            auto_save=False,
        )
        stressed = MindEngine(
            persona=get_persona("feynman"),
            bridge=PyBotBridge.disabled(),
            kb=self.kb,
            auto_save=False,
        )
        stressed.state.set("cortisol", 0.62)
        now = stressed.state.last_ts
        stressed.state.last_ts = now
        normal.state.last_ts = now

        normal_result = normal.turn("热水为什么会自己变凉", now=now)
        stressed_result = stressed.turn("热水为什么会自己变凉", now=now)

        self.assertEqual(normal_result.left_brain["detail"], "normal")
        self.assertIn("left_brain", normal.snapshot())
        self.assertEqual(stressed_result.left_brain["posture"], "careful")
        self.assertEqual(stressed_result.left_brain["detail"], "brief")
        self.assertLess(len(stressed_result.base_reply),
                        len(normal_result.base_reply))
        self.assertEqual(stressed_result.source, "feynman_kb:U0_RELAX")


class TestPostureIsBehavioral(unittest.TestCase):
    def test_careful_drops_prefixes_and_playful_endings(self):
        from omega_mind.policy import StyleDirective
        from omega_mind.renderer import Renderer

        directive = StyleDirective(
            tone="warm", warmth=0.7, directness=0.6, verbosity=0.5,
            humor=0.5, tempo=1.0, expression="smile",
            prefixes=["嗯，"], endings=["～"],
        )
        relaxed = Renderer().render("熵是无序的度量", directive, 1,
                                    "zh", posture="answer")
        careful = Renderer().render("熵是无序的度量", directive, 1,
                                    "zh", posture="careful")
        self.assertTrue(relaxed.startswith("嗯，"))
        self.assertFalse(careful.startswith("嗯，"))
        self.assertIn("熵是无序的度量", careful)
        self.assertNotIn("～", careful)

    def test_near_tie_hedges_only_under_careful_posture(self):
        # An English turn is answered from the English package; a mind built
        # on one package cannot switch languages at turn time.
        kb = real_kb("en")
        engine = MindEngine(
            persona=get_persona("feynman"),
            bridge=PyBotBridge.disabled(),
            kb=kb,
            auto_save=False,
            language="en",
        )
        careful = left_brain.LeftBrainPolicy(posture="careful", limit=3)
        tentative, source = engine._base_reply("What is entropy?", careful)
        self.assertTrue(source.startswith("feynman_kb"))
        self.assertIsNotNone(engine.last_retrieval["margin"])
        self.assertIn("not fully certain", tentative)

        # A short answer still has the same runner-up and uncertainty.
        terse = left_brain.LeftBrainPolicy(posture="careful", limit=1)
        still_tentative, terse_source = engine._base_reply("What is entropy?", terse)
        self.assertIsNotNone(engine.last_retrieval["margin"])
        self.assertEqual(terse_source, source)
        self.assertEqual(still_tentative, tentative)


class TestSharpActivationAlert(unittest.TestCase):
    def test_sharp_activation_tightens_before_slow_state_moves(self):
        from omega_mind.appraisal import Activation

        p = get_persona("generic")
        state = AffectState.from_persona(p, 0.0)
        quiet = plan_for(state, p, [])
        jolted = plan_for(state, p, [Activation(
            name="edge_suspicion", strength=1.0, effects={})])
        self.assertEqual(quiet.posture, "answer")
        self.assertEqual(jolted.posture, "careful")
        self.assertGreater(jolted.stress, 0.0)
        self.assertGreaterEqual(jolted.min_score, quiet.min_score)
