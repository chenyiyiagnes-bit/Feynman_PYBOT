"""A save from another version must never stop the app from starting.

Extra keys are dropped, entries missing required fields are skipped, and the
rest of the save still loads. Also pins that a zero restore ratio is a true
no-op.
"""

from __future__ import annotations

import copy
import io
import contextlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.cube import CubeState
from omega_mind.cube_feedback import CubeFeedback, displacement
from omega_mind.dream import Dream, Dreamer
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.keyframe_report import main as keyframe_main
from omega_mind.memory import Episode, MemoryTiers
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.relations import RelationshipGraph


def offline_engine() -> MindEngine:
    return MindEngine(
        persona=get_persona("feynman-zh"),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
        language="zh",
    )


def episode(turn: int) -> Episode:
    return Episode(turn=turn, now=1000.0 + turn, user="嗯",
                   text=f"第{turn}件事", reply="好", source="test",
                   salience=3.0)


class TestZeroRatioIsANoOp(unittest.TestCase):
    def test_ratio_zero_leaves_the_cube_alone(self):
        cube = CubeState()
        cube.apply("R U R' U' F R")
        before = cube.snapshot()["occupancy"]
        report = CubeFeedback().partial_restore(cube, ratio=0.0)
        self.assertEqual(report["moves"], [])
        self.assertEqual(cube.snapshot()["occupancy"], before)
        self.assertEqual(report["actual_reduction"], 0.0)

    def test_negative_ratio_is_treated_as_zero(self):
        cube = CubeState()
        cube.apply("R U R' U' F R")
        before = displacement(cube)
        report = CubeFeedback().partial_restore(cube, ratio=-0.5)
        self.assertEqual(report["moves"], [])
        self.assertEqual(displacement(cube), before)


class TestDreamRestoreRobustness(unittest.TestCase):
    def test_extra_key_is_dropped(self):
        dreamer = Dreamer()
        dreamer.restore({"dreams": [
            {"index": 1, "now": 5.0, "episodes": 2, "future_field": 9},
        ]})
        self.assertEqual(len(dreamer.dreams), 1)
        self.assertEqual(dreamer.dreams[0].episodes, 2)

    def test_malformed_entries_are_skipped(self):
        dreamer = Dreamer()
        dreamer.restore({"dreams": [
            {"index": 1}, "garbage", None, {"index": 2, "now": 1.0, "episodes": 3},
        ]})
        self.assertEqual([d.index for d in dreamer.dreams], [2])

    def test_round_trip_still_matches(self):
        dreamer = Dreamer()
        dreamer.dreams = [Dream(index=1, now=1.0, episodes=8, salience=4.5)]
        clone = Dreamer()
        clone.restore(dreamer.snapshot())
        self.assertEqual(clone.snapshot(), dreamer.snapshot())


class TestMemoryRestoreRobustness(unittest.TestCase):
    def test_extra_key_is_dropped_and_order_kept(self):
        tiers = MemoryTiers()
        for i in range(3):
            tiers.record(episode(i))
        data = json_copy(tiers.snapshot())
        data["episodes"][1]["future_field"] = 1
        clone = MemoryTiers()
        clone.restore(data)
        self.assertEqual([e.turn for e in clone.episodes], [0, 1, 2])

    def test_malformed_entries_are_skipped(self):
        tiers = MemoryTiers()
        tiers.restore({"episodes": [
            {"turn": 1}, "junk", episode(7).to_dict(),
        ], "working": [None, episode(8).to_dict()]})
        self.assertEqual([e.turn for e in tiers.episodes], [7])
        self.assertEqual([e.turn for e in tiers.working], [8])


class TestRelationRestoreRobustness(unittest.TestCase):
    def test_extra_keys_and_bad_edges(self):
        graph = RelationshipGraph()
        graph.node("阿琳").affinity = 0.9
        data = json_copy(graph.snapshot())
        data["nodes"]["阿琳"]["future_field"] = 1
        data["edges"].append({"src": "x"})          # missing dst/kind
        data["edges"].append({"src": "a", "dst": "b", "kind": "friend",
                              "future_field": 2})
        clone = RelationshipGraph()
        clone.restore(data)
        self.assertEqual(clone.node("阿琳").affinity, 0.9)
        self.assertEqual(set(clone.edges), {("a", "b", "friend")})

    def test_round_trip_still_matches(self):
        graph = RelationshipGraph()
        graph.node("阿琳")
        clone = RelationshipGraph()
        clone.restore(graph.snapshot())
        self.assertEqual(clone.snapshot(), graph.snapshot())


class TestEngineLoadsAForeignSave(unittest.TestCase):
    def test_extra_keys_do_not_stop_startup(self):
        engine = offline_engine()
        for i in range(8):
            engine.tiers.record(episode(i))
        engine.dream(now=2000.0, force=True)
        data = json_copy(engine.snapshot())
        data["dreams"]["dreams"][0]["future_field"] = 1
        data["tiers"]["episodes"][0]["future_field"] = 1
        data["relations"]["nodes"]["self"]["future_field"] = 1
        loaded = MindEngine.load_from_dict(
            data, bridge=PyBotBridge.disabled(),
            kb=FeynmanKB.disabled(), auto_save=False)
        self.assertEqual(loaded.dreamer.last().episodes, 8)


class TestLoadFromDiskSurvivesBadShape(unittest.TestCase):
    def _write(self, payload) -> str:
        import json
        import tempfile
        path = Path(tempfile.mkdtemp()) / "state.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_wrong_container_types_start_fresh_with_a_warning(self):
        for key, value in (("tiers", [1, 2]), ("dreams", [1, 2]),
                           ("relations", {"nodes": [], "edges": []})):
            with self.subTest(key=key):
                path = self._write({
                    "persona": "feynman-zh", "state": {}, "memory": {},
                    "tiers": {}, "relations": {}, "cube": {},
                    "cube_feedback": {}, "dreams": {}, "renderer": {},
                    key: value,
                })
                engine = MindEngine.load(
                    path, bridge=PyBotBridge.disabled(),
                    kb=FeynmanKB.disabled(), auto_save=False)
                self.assertIsNotNone(engine.load_warning)
                self.assertTrue(Path(path + ".corrupt").exists())

    def test_bad_values_start_fresh_with_a_warning(self):
        path = self._write({"state": {"dopamine": "high"}})
        engine = MindEngine.load(path, bridge=PyBotBridge.disabled(),
                                 kb=FeynmanKB.disabled(), auto_save=False)
        self.assertIsNotNone(engine.load_warning)
        self.assertTrue(Path(path + ".corrupt").exists())

    def test_a_valid_save_still_loads(self):
        engine = offline_engine()
        engine.turn("你好", now=1000.0)
        path = self._write(engine.snapshot())
        loaded = MindEngine.load(path, bridge=PyBotBridge.disabled(),
                                 kb=FeynmanKB.disabled(), auto_save=False)
        self.assertIsNone(loaded.load_warning)
        self.assertEqual(loaded.turn_index, engine.turn_index)


class TestKeyframeCliErrors(unittest.TestCase):
    def test_zero_turns_is_a_usage_error_not_a_traceback(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as ctx:
                keyframe_main(["--turns", "0"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("turns must be at least 1", stderr.getvalue())


def json_copy(data):
    import json
    return json.loads(json.dumps(data))


if __name__ == "__main__":
    unittest.main()
