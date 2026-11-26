"""Regressions for mechanisms that were reachable on paper but dead in practice.

Each test names the failure it locks out: habituation that never expires, a
style layer no consumer reads, thresholds no real state reaches, a cube bias
that sticks at its maximum after one turn, memory facts that are stored but
never spoken, and a dream that quotes English episodes with Chinese brackets.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.appraisal import AppraisalEngine
from omega_mind.config import get_persona
from omega_mind.cube import CubeState
from omega_mind.cube_feedback import CubeFeedback, CubeBias, displacement
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.left_brain import plan_for
from omega_mind.memory import EPISODIC_CAP, Episode, MemoryTiers
from omega_mind.policy import Policy, StyleDirective
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.renderer import Renderer
from omega_mind.state import AffectState


def offline_engine(persona: str = "generic", language: str = "zh") -> MindEngine:
    return MindEngine(
        persona=get_persona(persona),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
        language=language,
    )


def strength_of(engine: AppraisalEngine, text: str, turn: int, name: str) -> float:
    for act in engine.evaluate(text, turn):
        if act.name == name:
            return act.strength
    return 0.0


class TestHabituationWindow(unittest.TestCase):
    def test_first_trigger_keeps_full_strength(self):
        engine = AppraisalEngine(get_persona("generic"))
        first = strength_of(engine, "听说公司要裁员", 0, "edge_trauma")
        self.assertGreater(first, 0.5)

    def test_repeat_inside_the_window_is_damped(self):
        engine = AppraisalEngine(get_persona("generic"))
        first = strength_of(engine, "听说公司要裁员", 0, "edge_trauma")
        second = strength_of(engine, "听说公司要裁员", 1, "edge_trauma")
        self.assertAlmostEqual(second / first, 0.6, places=2)

    def test_habituation_expires_after_the_window(self):
        engine = AppraisalEngine(get_persona("generic"))
        for turn in range(3):
            engine.evaluate("听说公司要裁员", turn)
        later = strength_of(engine, "听说公司要裁员", 10, "edge_trauma")
        fresh = strength_of(AppraisalEngine(get_persona("generic")),
                            "听说公司要裁员", 0, "edge_trauma")
        self.assertAlmostEqual(later, fresh, places=4)

    def test_sustained_repetition_bottoms_out_instead_of_vanishing(self):
        engine = AppraisalEngine(get_persona("generic"))
        first = strength_of(engine, "听说公司要裁员", 0, "edge_trauma")
        values = [strength_of(engine, "听说公司要裁员", turn, "edge_trauma")
                  for turn in range(1, 8)]
        self.assertGreater(values[-1], 0.0)
        self.assertGreaterEqual(values[-1], first * 0.2)


class TestStyleIsConsumed(unittest.TestCase):
    def setUp(self):
        self.persona = get_persona("generic")

    def _directive(self, **vals) -> StyleDirective:
        state = AffectState.from_persona(self.persona, 0.0)
        for key, value in vals.items():
            state.set(key, value)
        return Policy(self.persona).compute(state, [])

    def test_warm_and_cold_render_different_shells(self):
        warm = self._directive(dopamine=0.9, valence=0.9, affinity=0.9,
                               cortisol=0.1, defensiveness=0.0)
        cold = self._directive(dopamine=0.1, valence=-0.9, affinity=0.0,
                               cortisol=0.8, defensiveness=0.7)
        self.assertGreater(warm.warmth, cold.warmth)
        r_warm = Renderer().render("答案是四十二", warm, 0, "zh")
        r_cold = Renderer().render("答案是四十二", cold, 0, "zh")
        self.assertNotEqual(r_warm, r_cold)

    def test_low_verbosity_drops_the_prefix(self):
        directive = StyleDirective(
            tone="neutral", warmth=0.5, directness=0.5, verbosity=0.2,
            humor=0.2, tempo=1.0, expression="neutral_calm",
            prefixes=["嗯，"], endings=["。"],
        )
        reply = Renderer().render("熵是无序的度量", directive, 0, "zh")
        self.assertFalse(reply.startswith("嗯，"))

    def test_expression_chooses_its_own_tags(self):
        renderer = Renderer()
        smile = renderer._tags_for("soft_smile", "zh")
        pout = renderer._tags_for("angry_pout", "zh")
        self.assertNotEqual(smile, pout)
        self.assertTrue(all(tag.startswith("（") for tag in smile))

    def test_exhaustion_reaches_terse_verbosity(self):
        # Without stress/defensiveness in the formula, verbosity could never
        # fall below 0.40 and the renderer's terse branch was dead code.
        directive = self._directive(energy=0.0, arousal=0.0,
                                    cortisol=1.0, defensiveness=0.8)
        self.assertLessEqual(directive.verbosity, 0.30)


class TestLeftBrainThresholdsAreReachable(unittest.TestCase):
    def test_high_but_reachable_dopamine_explores(self):
        p = get_persona("feynman")
        state = AffectState.from_persona(p, 0.0)
        state.set("dopamine", 0.72)  # observed peak after a strong compliment
        policy = plan_for(state, p, [])
        self.assertEqual(policy.detail, "rich")
        self.assertEqual(policy.posture, "explore")
        self.assertIn("teachable_unit", policy.prefer_kinds)

    def test_sustained_effort_drains_energy_and_briefs_the_answer(self):
        engine = offline_engine()
        lines = ["你今天真的让我很失望", "闭嘴，废物", "这都什么乱七八糟的",
                 "你到底懂不懂", "烦死了", "你根本不是认真的"]
        now = 1000.0
        for i in range(30):
            now += 10.0
            engine.turn(lines[i % len(lines)], now=now)
        self.assertLess(engine.state.get("energy"), 0.62)
        self.assertEqual(engine.last_left_brain["detail"], "brief")
        self.assertGreater(engine.last_left_brain["fatigue"], 0.0)

    def test_a_long_rest_restores_energy(self):
        engine = offline_engine()
        now = 1000.0
        for i in range(30):
            now += 10.0
            engine.turn("闭嘴，废物", now=now)
        self.assertLess(engine.state.get("energy"), 0.62)
        engine.turn("嗯", now=now + 7200.0)
        self.assertGreater(engine.state.get("energy"), 0.70)


class TestCubeBiasIsGraduated(unittest.TestCase):
    def test_one_turn_does_not_close_the_aperture(self):
        cube = CubeState()
        cube.apply("R")
        bias = CubeFeedback().bias(cube)
        self.assertLessEqual(abs(bias.depth), 1)
        self.assertLessEqual(abs(bias.verbosity), 0.2)
        p = get_persona("generic")
        policy = plan_for(AffectState.from_persona(p, 0.0), p, [],
                          cube_bias=bias)
        self.assertEqual(policy.detail, "normal")

    def test_a_dream_that_solves_the_cube_returns_a_neutral_bias(self):
        cube = CubeState()
        cube.apply("R U F'")
        feedback = CubeFeedback()
        report = feedback.partial_restore(cube, ratio=1.0, max_moves=50)
        self.assertEqual(displacement(cube), 0)
        self.assertEqual((feedback.bias(cube).depth, feedback.bias(cube).verbosity),
                         (0, 0.0))

    def test_extreme_bias_is_still_bounded(self):
        bias = CubeFeedback().bias(CubeState())
        self.assertEqual((bias.depth, bias.verbosity), (0, 0.0))
        for depth in range(-4, 5):
            policy = plan_for(AffectState.from_persona(get_persona("generic"), 0.0),
                              get_persona("generic"), [],
                              cube_bias=CubeBias(depth=depth, verbosity=0.0))
            self.assertIn(policy.limit, (1, 2, 3, 4))


class TestSemanticFactsAreSpoken(unittest.TestCase):
    def test_name_comes_back_when_asked(self):
        engine = offline_engine()
        engine.turn("我叫王小明", now=1000.0)
        result = engine.turn("我叫什么？", now=1010.0)
        self.assertIn("王小明", result.reply)
        self.assertTrue(result.source.startswith("memory"))

    def test_english_name_comes_back_when_asked(self):
        engine = offline_engine(language="en")
        engine.turn("My name is Alex", now=1000.0)
        result = engine.turn("What's my name?", now=1010.0)
        self.assertIn("Alex", result.reply)
        self.assertTrue(result.source.startswith("memory"))
        self.assertFalse(any("一" <= ch <= "鿿" for ch in result.reply))

    def test_role_comes_back_from_the_relationship_graph(self):
        engine = offline_engine()
        engine.turn("我妈叫王芳", now=1000.0)
        result = engine.turn("我妈叫什么？", now=1010.0)
        self.assertIn("王芳", result.reply)
        self.assertTrue(result.source.startswith("memory"))

    def test_unknown_question_falls_through(self):
        engine = offline_engine()
        engine.turn("我叫王小明", now=1000.0)
        result = engine.turn("我养的花叫什么？", now=1010.0)
        self.assertFalse(result.source.startswith("memory"))


class TestQuestionsDoNotBecomeFacts(unittest.TestCase):
    """A question is not a statement: its interrogative must never be stored."""

    def test_asking_what_i_like_stores_nothing(self):
        engine = offline_engine()
        engine.turn("我喜欢什么？", now=1000.0)
        self.assertNotIn("likes:default", engine.tiers.semantic)

    def test_asking_where_i_live_stores_nothing(self):
        engine = offline_engine()
        engine.turn("我住在哪儿？", now=1000.0)
        self.assertNotIn("lives_in:default", engine.tiers.semantic)

    def test_asking_a_relative_name_creates_no_node(self):
        engine = offline_engine()
        result = engine.turn("我妈叫什么？", now=1000.0)
        self.assertNotIn("什么", engine.relations.nodes)
        self.assertFalse(result.source.startswith("memory"))

    def test_a_question_before_the_introduction_does_not_poison_it(self):
        engine = offline_engine()
        engine.turn("我妈叫什么？", now=1000.0)
        engine.turn("我妈叫王芳", now=1010.0)
        result = engine.turn("我妈叫什么？", now=1020.0)
        self.assertIn("王芳", result.reply)

    def test_a_statement_about_not_knowing_is_not_the_work_question(self):
        engine = offline_engine()
        engine.turn("我在学校工作", now=1000.0)
        result = engine.turn("我不知道我干什么", now=1010.0)
        self.assertNotIn("学校", result.reply)

    def test_the_work_question_still_works(self):
        engine = offline_engine()
        engine.turn("我在学校工作", now=1000.0)
        result = engine.turn("我做什么工作？", now=1010.0)
        self.assertIn("学校", result.reply)


class TestEnglishRelatives(unittest.TestCase):
    def test_english_relative_is_remembered_and_asked_in_english(self):
        engine = offline_engine(language="en")
        engine.turn("My mom is Wang Fang", now=1000.0)
        result = engine.turn("What's my mom's name?", now=1010.0)
        self.assertIn("Wang Fang", result.reply)
        self.assertTrue(result.source.startswith("memory"))

    def test_relative_introduced_in_chinese_answers_an_english_question(self):
        engine = offline_engine()
        engine.turn("我妈叫王芳", now=1000.0)
        engine.set_language("en")
        result = engine.turn("What's my mom's name?", now=1010.0)
        self.assertIn("王芳", result.reply)

    def test_english_question_does_not_create_a_bogus_node(self):
        engine = offline_engine(language="en")
        engine.turn("What's my mom's name?", now=1000.0)
        self.assertNotIn("name", engine.relations.nodes)

    def test_english_question_does_not_store_a_fact(self):
        engine = offline_engine(language="en")
        engine.turn("What do I like?", now=1000.0)
        engine.turn("Where do I live?", now=1010.0)
        self.assertFalse(engine.tiers.semantic)

    def test_an_english_occupation_is_not_a_name(self):
        engine = offline_engine(language="en")
        engine.turn("My mom is a teacher", now=1000.0)
        self.assertNotIn("a teacher", engine.relations.nodes)

    def test_an_english_trailing_clause_is_cut(self):
        engine = offline_engine(language="en")
        engine.turn("My mom is Wang Fang and she is a teacher", now=1000.0)
        self.assertIn("Wang Fang", engine.relations.nodes)
        self.assertNotIn("Wang Fang and she is a teacher", engine.relations.nodes)


class TestEpisodesStayBounded(unittest.TestCase):
    def test_more_episodes_than_the_cap_do_not_grow_without_dreams(self):
        # A batch caller that never dreams must not grow memory forever.
        tiers = MemoryTiers()
        for i in range(EPISODIC_CAP + 300):
            tiers.record(Episode(turn=i, now=float(i), user="u", text="嗯",
                                 reply="r", source="s", salience=0.0))
        self.assertLessEqual(len(tiers.episodes), EPISODIC_CAP)


class TestTrustAndProactive(unittest.TestCase):
    def test_trust_tracks_closeness(self):
        from omega_mind.relations import RelationshipGraph
        graph = RelationshipGraph()
        node = graph.node("u")
        node.affinity = 1.0
        for i in range(20):
            graph.touch("u", float(i))
        self.assertGreater(node.trust, 0.6)
        node.affinity = 0.0
        for i in range(20):
            graph.touch("u", float(i + 20))
        self.assertLess(node.trust, 0.6)

    def test_a_moderately_happy_mind_still_reaches_out(self):
        engine = offline_engine()
        engine.state.set("affinity", 0.8)
        engine.state.set("dopamine", 0.50)
        engine.last_interaction = 0.0
        self.assertIsNotNone(engine.tick(now=1000.0))

    def test_a_very_happy_mind_does_not_interrupt(self):
        engine = offline_engine()
        engine.state.set("affinity", 0.8)
        engine.state.set("dopamine", 0.75)
        engine.last_interaction = 0.0
        self.assertIsNone(engine.tick(now=1000.0))


class TestDreamLanguage(unittest.TestCase):
    def test_english_dream_uses_english_quotes(self):
        engine = offline_engine(language="en")
        for i in range(8):
            engine.tiers.record(Episode(turn=i, now=float(i), user="u",
                                        text=f"event {i}", reply="r",
                                        source="s", salience=1.0))
        dream = engine.dream(force=True)
        self.assertIsNotNone(dream)
        self.assertIn('"', dream.narrative)
        self.assertNotIn("「", dream.narrative)

    def test_tied_episodes_are_ordered_by_affect_intensity(self):
        engine = offline_engine()
        for i in range(8):
            engine.tiers.record(Episode(turn=i, now=float(i), user="u",
                                        text=f"事件{i}", reply="r", source="s",
                                        salience=1.0,
                                        affect={"cortisol": 0.0}))
        engine.tiers.episodes[3].affect = {"cortisol": 0.9}
        dream = engine.dream(force=True)
        self.assertIsNotNone(dream)
        self.assertIn("事件3", dream.narrative)


class TestSnapshotRestoresTurnContext(unittest.TestCase):
    def test_last_left_brain_survives_a_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mind.json")
            engine = MindEngine(
                persona=get_persona("generic"), store_path=path,
                bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled(),
                auto_save=False,
            )
            engine.turn("听说公司要裁员", now=1000.0)
            engine.save()
            loaded = MindEngine.load(
                path, persona=get_persona("generic"),
                bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled(),
            )
            self.assertEqual(loaded.last_left_brain, engine.last_left_brain)
            self.assertEqual(loaded.last_retrieval, engine.last_retrieval)


if __name__ == "__main__":
    unittest.main()
