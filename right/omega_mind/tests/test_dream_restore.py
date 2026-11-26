"""How far a dream walks the cube home depends on how heavy the batch was.

A dream about ordinary chatter may settle the stance; a dream about a fight
or a loss must leave most of it standing. These tests pin the curve, the
Dream record, save compatibility, and the engine actually using it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.cube import CubeState
from omega_mind.cube_feedback import (
    DREAM_RESTORE_MIN_FRACTION,
    RESTORE_RATIO,
    CubeFeedback,
    dream_restore_ratio,
    displacement,
)
from omega_mind.dream import Dream, Dreamer
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.memory import Episode, MemoryTiers
from omega_mind.pybot_bridge import PyBotBridge


def offline_engine() -> MindEngine:
    return MindEngine(
        persona=get_persona("feynman-zh"),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
        language="zh",
    )


def episode(turn: int, salience: float) -> Episode:
    return Episode(turn=turn, now=1000.0 + turn, user="嗯",
                   text=f"第{turn}件事", reply="好", source="test",
                   salience=salience)


class TestRestoreRatioCurve(unittest.TestCase):
    def test_low_salience_gets_the_full_settling_pass(self):
        self.assertEqual(dream_restore_ratio(0.0), RESTORE_RATIO)

    def test_high_salience_keeps_only_the_floor_fraction(self):
        expected = round(RESTORE_RATIO * DREAM_RESTORE_MIN_FRACTION, 4)
        self.assertEqual(dream_restore_ratio(2.5), expected)
        # Above the scale it must not go lower than the floor.
        self.assertEqual(dream_restore_ratio(50.0), expected)

    def test_scale_matches_measured_salience_not_the_hard_cap(self):
        # Real turns land well below salience_of's 0..10 cap: ordinary chat
        # near 0.1, an insult near 0.9, the heaviest line seen near 2.4.
        self.assertGreater(dream_restore_ratio(0.1), 0.42)
        self.assertLess(dream_restore_ratio(0.9), 0.36)
        self.assertLessEqual(dream_restore_ratio(2.4), 0.13)

    def test_negative_salience_is_treated_as_zero(self):
        self.assertEqual(dream_restore_ratio(-3.0), RESTORE_RATIO)

    def test_curve_never_increases(self):
        values = [dream_restore_ratio(s) for s in (0, 1, 2.5, 5, 7.5, 9, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_curve_stays_inside_the_ratio_bounds(self):
        floor = round(RESTORE_RATIO * DREAM_RESTORE_MIN_FRACTION, 4)
        for s in (-1, 0, 3.3, 10, 99):
            ratio = dream_restore_ratio(s)
            self.assertGreaterEqual(ratio, floor)
            self.assertLessEqual(ratio, RESTORE_RATIO)


class TestHeavyDreamRestoresLess(unittest.TestCase):
    MOVES = ("R U R' U' F R")

    def _cube(self) -> CubeState:
        cube = CubeState()
        cube.apply(self.MOVES)
        return cube

    def test_displacement_ordering_follows_salience(self):
        light, heavy = self._cube(), self._cube()
        start = displacement(light)
        feedback = CubeFeedback()
        light_report = feedback.partial_restore(
            light, ratio=dream_restore_ratio(0.0))
        heavy_report = feedback.partial_restore(
            heavy, ratio=dream_restore_ratio(10.0))
        self.assertEqual(displacement(heavy), heavy_report["final_displaced"])
        self.assertGreaterEqual(displacement(heavy), displacement(light))
        self.assertGreater(heavy_report["final_displaced"],
                          light_report["final_displaced"])
        self.assertLessEqual(displacement(heavy), start)

    def test_no_ratio_can_scramble_the_cube(self):
        for salience in (0.0, 5.0, 10.0):
            with self.subTest(salience=salience):
                cube = self._cube()
                before = displacement(cube)
                CubeFeedback().partial_restore(
                    cube, ratio=dream_restore_ratio(salience))
                self.assertLessEqual(displacement(cube), before)


class TestNoGrossOvershoot(unittest.TestCase):
    """One inverse move can erase more of the stance than a light dream asked."""

    def test_light_request_on_a_deep_walk_restores_nothing(self):
        # Ratio 0.09 on a 10-facelet walk asks to keep ~90%; the walk jumps
        # 10 -> 7 at best on the gentler side, so only that small step is
        # taken, never the 10 -> 0 crossing.
        cube = CubeState()
        cube.apply("R U R' U' F R")
        report = CubeFeedback().partial_restore(
            cube, ratio=dream_restore_ratio(2.4))
        self.assertTrue(report["moves"])
        self.assertEqual(displacement(cube), 8)
        self.assertLessEqual(report["actual_reduction"], 0.2)

    def test_gentler_prefix_wins_over_the_crossing_one(self):
        cube = CubeState()
        cube.apply("R U R' U' F R")
        report = CubeFeedback().partial_restore(cube, ratio=RESTORE_RATIO)
        self.assertLess(displacement(cube), 10)
        self.assertGreater(report["final_displaced"], report["target_remaining"])
        self.assertLessEqual(report["actual_reduction"], RESTORE_RATIO + 0.15)


class TestRealAppraisalCalibration(unittest.TestCase):
    """The scale must fit salience as the real appraisal path produces it."""

    def _dream_after(self, lines) -> tuple[float, int]:
        engine = offline_engine()
        engine.turn_cube("R U R' U' F R")
        for i, text in enumerate(lines):
            engine.turn(text, now=1000.0 + 20 * i)
        dream = engine.dream(now=5000.0, force=True)
        self.assertIsNotNone(dream)
        from omega_mind.cube_feedback import dream_restore_ratio
        return dream.salience, dream_restore_ratio(dream.salience)

    def test_ordinary_chat_still_gets_a_full_settling_pass(self):
        salience, ratio = self._dream_after(["今天天气不错", "随便聊聊"] * 4)
        self.assertLess(salience, 0.3)
        self.assertGreater(ratio, 0.42)

    def test_an_insult_batch_keeps_most_of_the_stance(self):
        salience, ratio = self._dream_after(
            ["你真是个蠢货", "滚开，别再出现在我面前"] * 4)
        self.assertGreater(salience, 0.5)
        self.assertLess(ratio, 0.36)

    def test_heavy_leaves_more_displacement_than_ordinary(self):
        engine_light = offline_engine()
        engine_heavy = offline_engine()
        for e in (engine_light, engine_heavy):
            e.turn_cube("R U R' U' F R")
        for i in range(8):
            engine_light.turn("今天天气不错", now=1000.0 + 20 * i)
            engine_heavy.turn("你真是个蠢货", now=1000.0 + 20 * i)
        engine_light.dream(now=5000.0, force=True)
        engine_heavy.dream(now=5000.0, force=True)
        self.assertGreater(displacement(engine_heavy.cube),
                           displacement(engine_light.cube))


class TestDreamRecordsSalience(unittest.TestCase):
    def test_batch_peak_is_recorded(self):
        tiers = MemoryTiers()
        for i in range(8):
            tiers.record(episode(i, salience=2.0 if i != 4 else 9.5))
        engine = offline_engine()
        dream = Dreamer(threshold=8).consolidate(
            tiers, engine.state, engine.persona, 2000.0)
        self.assertIsNotNone(dream)
        self.assertEqual(dream.salience, 9.5)

    def test_snapshot_round_trip_keeps_salience(self):
        dream = Dream(index=1, now=1.0, episodes=8, salience=7.25)
        data = json.loads(json.dumps(dream.to_dict()))
        self.assertEqual(Dream(**data).salience, 7.25)

    def test_old_saves_without_the_field_still_load(self):
        old = {"index": 1, "now": 1.0, "episodes": 8}
        self.assertEqual(Dream(**old).salience, 0.0)


class TestEngineDreamUsesSalience(unittest.TestCase):
    def _dreamed_displacement(self, salience: float) -> tuple[int, int, dict]:
        engine = offline_engine()
        engine.turn_cube("R U R' U' F R")
        for i in range(8):
            engine.tiers.record(episode(i, salience=salience))
        before = displacement(engine.cube)
        dream = engine.dream(now=3000.0, force=True)
        self.assertIsNotNone(dream)
        self.assertEqual(dream.salience, salience)
        after = displacement(engine.cube)
        report = next(event["cube_restore"]
                      for event in reversed(engine.event_log)
                      if "cube_restore" in event)
        return before, after, report

    def test_heavy_batch_leaves_more_displacement(self):
        light_before, light_after, light_report = self._dreamed_displacement(0.0)
        heavy_before, heavy_after, heavy_report = self._dreamed_displacement(10.0)
        self.assertEqual(light_before, heavy_before)
        self.assertGreater(heavy_after, light_after)
        self.assertGreater(heavy_report["restore_ratio"],
                           0)
        self.assertLess(heavy_report["restore_ratio"],
                        light_report["restore_ratio"])
        self.assertEqual(heavy_report["salience"], 10.0)


if __name__ == "__main__":
    unittest.main()
