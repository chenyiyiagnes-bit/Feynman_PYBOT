"""The cube's continuous state, read as a signal instead of a decoration.

Three quantities, all derived from the arrangement and nothing else:

* ``displacement`` — which fraction of the 26 cubies is away from home;
* ``delta`` — how many displaced cubies were gained/lost since the last
  observed turn (the mind's momentum: settling vs churn);
* ``front_load`` — which fraction of the front face is displaced (how much
  of the current focus is a moved stance rather than the default one).

The affect state already relaxes toward a baseline shifted by displacement;
that is a slow channel. ``delta`` is the fast one, and the only one the left
brain consumes: a cube that just closed part of its displacement lets a calm
mind say the fuller answer, and a cube that just took a fresh stir must not
open up further on the same turn. Zero signal changes nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import left_brain
from omega_mind.config import get_persona
from omega_mind.cube import CubeState
from omega_mind.cube_feedback import CubeFeedback, CubeSignal, displacement, front_names
from omega_mind.cube import HOME_OF
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.state import AffectState


def offline_engine(persona="generic") -> MindEngine:
    return MindEngine(
        persona=get_persona(persona),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
    )


class TestCubeSignalValues(unittest.TestCase):
    def test_a_solved_cube_reads_as_zero(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        signal = feedback.signal
        self.assertEqual(signal.displacement, 0.0)
        self.assertEqual(signal.delta, 0)
        self.assertEqual(signal.front_load, 0.0)

    def test_one_move_shows_up_as_positive_momentum(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        cube.apply("R")
        feedback.observe(cube, 1)
        signal = feedback.signal
        self.assertGreater(signal.delta, 0)
        self.assertEqual(signal.delta, displacement(cube))
        self.assertAlmostEqual(signal.displacement, displacement(cube) / 26.0)
        front = {name for pos, name in cube.occupancy.items() if pos[2] == 1}
        loaded = sum(1 for name in front if HOME_OF.get(name) != cube.position_of(name))
        self.assertAlmostEqual(signal.front_load, loaded / 9.0)

    def test_undoing_the_move_reads_as_settling(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        cube.apply("R")
        feedback.observe(cube, 1)
        cube.apply("R'")
        feedback.observe(cube, 2)
        self.assertLess(feedback.signal.delta, 0)
        self.assertEqual(feedback.signal.displacement, 0.0)

    def test_a_repeated_observation_on_the_same_turn_is_ignored(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        block = feedback.signal
        cube.apply("R")
        feedback.observe(cube, 0)          # same turn: no re-measure
        self.assertEqual(feedback.signal, block)

    def test_quantities_stay_inside_their_bounds(self):
        cube = CubeState()
        feedback = CubeFeedback()
        for turn, move in enumerate("R U R' U' F D L B".split()):
            feedback.observe(cube, turn)
            cube.apply(move)
        feedback.observe(cube, 99)
        signal = feedback.signal
        self.assertGreaterEqual(signal.displacement, 0.0)
        self.assertLessEqual(signal.displacement, 1.0)
        self.assertGreaterEqual(signal.front_load, 0.0)
        self.assertLessEqual(signal.front_load, 1.0)

    def test_snapshot_restore_keeps_the_momentum_baseline(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        cube.apply("R")
        feedback.observe(cube, 1)
        data = feedback.snapshot()
        restored = CubeFeedback()
        restored.restore(data)
        cube.apply("R'")
        restored.observe(cube, 2)
        self.assertLess(restored.signal.delta, 0)
        self.assertEqual(restored.signal.displacement, 0.0)

    def test_reset_clears_the_signal(self):
        cube = CubeState()
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        cube.apply("R")
        feedback.observe(cube, 1)
        feedback.reset()
        self.assertEqual(feedback.signal, CubeSignal())


class TestSignalPolicy(unittest.TestCase):
    def _calm(self):
        p = get_persona("generic")
        return p, AffectState.from_persona(p, 0.0)

    def test_zero_signal_changes_nothing(self):
        p, state = self._calm()
        base = left_brain.plan_for(state, p, [])
        zero = left_brain.plan_for(state, p, [], cube_signal=CubeSignal())
        self.assertEqual(base.detail, zero.detail)

    def test_a_settling_cube_lets_a_calm_mind_say_more(self):
        p, state = self._calm()
        base = left_brain.plan_for(state, p, [])
        settling = left_brain.plan_for(
            state, p, [], cube_signal=CubeSignal(delta=-3))
        self.assertEqual(base.detail, "normal")
        self.assertEqual(settling.detail, "rich")

    def test_a_settling_cube_does_not_open_up_under_stress(self):
        p, state = self._calm()
        state.set("cortisol", 0.85)
        settling = left_brain.plan_for(
            state, p, [], cube_signal=CubeSignal(delta=-3))
        self.assertEqual(settling.detail, "brief")

    def test_a_fresh_stir_caps_a_rich_answer_at_normal(self):
        p, state = self._calm()
        state.set("dopamine", 0.90)
        curious = left_brain.plan_for(state, p, [])
        churned = left_brain.plan_for(
            state, p, [], cube_signal=CubeSignal(delta=3))
        self.assertEqual(curious.detail, "rich")
        self.assertEqual(churned.detail, "normal")

    def test_a_fresh_stir_never_forces_brief_by_itself(self):
        p, state = self._calm()
        churned = left_brain.plan_for(
            state, p, [], cube_signal=CubeSignal(delta=3))
        self.assertEqual(churned.detail, "normal")

    def test_detail_always_stays_on_the_ladder(self):
        p, state = self._calm()
        for delta in range(-26, 27):
            policy = left_brain.plan_for(
                state, p, [], cube_signal=CubeSignal(delta=delta))
            self.assertIn(policy.detail, ("brief", "normal", "rich"))

    def test_the_signal_is_reported_in_the_policy(self):
        p, state = self._calm()
        policy = left_brain.plan_for(
            state, p, [], cube_signal=CubeSignal(
                displacement=0.5, delta=2, front_load=0.25))
        self.assertEqual(policy.to_dict()["cube"]["delta"], 2)
        self.assertAlmostEqual(policy.to_dict()["cube"]["displacement"], 0.5)


class TestSignalInTheEngine(unittest.TestCase):
    def test_a_turn_reports_the_cube_signal(self):
        engine = offline_engine()
        result = engine.turn("你好", now=1000.0)
        self.assertIn("cube", result.left_brain)
        cube = result.left_brain["cube"]
        self.assertIn("displacement", cube)
        self.assertIn("delta", cube)
        self.assertIn("front_load", cube)

    def test_the_first_turn_opens_on_a_solved_cube(self):
        engine = offline_engine()
        result = engine.turn("你好", now=1000.0)
        self.assertEqual(result.left_brain["cube"]["displacement"], 0.0)

    def test_the_next_turn_sees_what_the_last_one_moved(self):
        engine = offline_engine()
        engine.turn("你好", now=1000.0)
        engine.cube.apply("R")
        result = engine.turn("你好", now=1001.0)
        self.assertGreater(result.left_brain["cube"]["delta"], 0)

    def test_snapshot_restore_keeps_the_signal_channel(self):
        engine = offline_engine()
        engine.turn("你好", now=1000.0)
        engine.cube.apply("R")
        data = engine.snapshot()
        clone = MindEngine.load_from_dict(
            data, bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled())
        result = clone.turn("你好", now=1001.0)
        self.assertGreater(result.left_brain["cube"]["delta"], 0)


if __name__ == "__main__":
    unittest.main()
