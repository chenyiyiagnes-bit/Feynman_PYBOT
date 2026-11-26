"""Reframing: the whole cube turns in the mind's hands, nothing else moves.

The mechanical form of reappraisal. A reframe changes which face counts as
"front", so attention, gains and front-corner bias land on different cubies —
but the arrangement is untouched, and displacement is exactly invariant. If a
reframe ever changed displacement it would be a solve in disguise, and the
tests here refuse that.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.cube import HOME_OF, CubeState, render_net
from omega_mind.cube_feedback import CubeFeedback, displacement, front_names
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


class TestReframeIsAFrameChange(unittest.TestCase):
    def test_reframe_moves_no_cubie(self):
        cube = CubeState()
        cube.apply("R U F")
        before = dict(cube.occupancy)
        for _ in range(4):
            cube.reframe(1)
            self.assertEqual(cube.occupancy, before)

    def test_reframe_preserves_displacement_exactly(self):
        cube = CubeState()
        cube.apply("R U F' L2")
        reference = displacement(cube)
        self.assertGreater(reference, 0)
        for _ in range(4):
            cube.reframe(1)
            self.assertEqual(displacement(cube), reference)

    def test_four_reframes_are_the_identity(self):
        cube = CubeState()
        self.assertEqual(cube.orientation, 0)
        cube.reframe()
        cube.reframe()
        cube.reframe()
        cube.reframe()
        self.assertEqual(cube.orientation, 0)

    def test_the_view_is_a_bijection_on_all_26_positions(self):
        cube = CubeState()
        for _ in range(4):
            viewed = {cube.view(pos) for pos in cube.occupancy}
            self.assertEqual(len(viewed), 26)
            cube.reframe(1)

    def test_the_front_face_follows_the_frame(self):
        cube = CubeState()
        cube.apply("R U")
        default_front = front_names(cube)
        cube.reframe(1)
        framed = front_names(cube)
        # After one yaw quarter-turn the mind is looking at the body's L face.
        expected = {name for pos, name in cube.occupancy.items() if pos[0] == -1}
        self.assertEqual(framed, expected)
        self.assertNotEqual(framed, default_front)

    def test_front_load_is_measured_in_the_current_frame(self):
        cube = CubeState()
        cube.apply("L2")  # the L face is now heavily displaced
        feedback = CubeFeedback()
        feedback.observe(cube, 0)
        default = feedback.signal.front_load
        cube.reframe(1)  # L is now in front
        feedback2 = CubeFeedback()
        feedback2.observe(cube, 0)
        self.assertGreater(feedback2.signal.front_load, default)

    def test_corner_bias_is_read_through_the_frame(self):
        cube = CubeState()
        cube.apply("L2")
        feedback = CubeFeedback()
        before = feedback.bias(cube)
        cube.reframe(1)
        after = feedback.bias(cube)
        self.assertNotEqual(before.to_dict(), after.to_dict())

    def test_orientation_survives_a_snapshot_round_trip(self):
        cube = CubeState()
        cube.reframe(3)
        data = cube.snapshot()
        restored = CubeState()
        restored.restore(data)
        self.assertEqual(restored.orientation, 3)
        self.assertEqual(restored.occupancy, cube.occupancy)

    def test_an_old_snapshot_restores_to_the_default_frame(self):
        cube = CubeState()
        cube.restore({"occupancy": [], "history": []})
        self.assertEqual(cube.orientation, 0)

    def test_render_mentions_the_frame_only_after_a_reframe(self):
        cube = CubeState()
        self.assertNotIn("frame:", render_net(cube))
        cube.reframe(1)
        self.assertIn("frame:", render_net(cube))


class TestRecastReframesTheEngine(unittest.TestCase):
    def _mind(self, language="zh"):
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False, language=language)

    def test_a_recast_turns_the_whole_cube(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        before = displacement(mind.cube)
        second = mind.turn("没懂", now=1001.0)
        self.assertTrue(second.source.startswith("analogy:"))
        self.assertEqual(mind.cube.orientation, 1)
        self.assertEqual(displacement(mind.cube), before)

    def test_the_frame_does_not_turn_again_inside_the_cooldown(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        mind.turn("没懂", now=1001.0)
        self.assertEqual(mind.cube.orientation, 1)
        # Pretend the recast machinery offers another one right away.
        mind._recast_seen.clear()
        mind.turn("没懂", now=1002.0)
        self.assertEqual(mind.cube.orientation, 1)


if __name__ == "__main__":
    unittest.main()
