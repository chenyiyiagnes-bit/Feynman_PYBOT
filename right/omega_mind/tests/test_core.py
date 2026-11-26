"""Core unit tests: state bounds/decay, renderer fact safety, PyBot bridge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.cubies import CUBIES
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge
from omega_mind.state import AffectState


class TestState(unittest.TestCase):
    def test_clamps_and_decays(self):
        p = get_persona("generic")
        s = AffectState.from_persona(p, now=0.0)
        s.set("cortisol", 5.0)          # clamped to 1
        s.set("valence", -9.0)          # clamped to -1
        self.assertEqual(s.get("cortisol"), 1.0)
        self.assertEqual(s.get("valence"), -1.0)
        s.advance(10_000.0, p)
        for dim in ("cortisol", "defensiveness", "valence", "dopamine"):
            self.assertAlmostEqual(s.get(dim), p.baseline(dim), places=3)


class TestRendererFactSafety(unittest.TestCase):
    def test_numbers_and_wording_survive(self):
        e = MindEngine(persona=get_persona("generic"), bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled())
        base = "建议用 18g 咖啡粉，在 25-30 秒内萃取出 36g 咖啡液。"
        for i in range(6):
            r = e.turn("随便聊聊", now=1000.0 + i * 60, base_reply=base)
            self.assertIn("18g", r.reply)
            self.assertIn("36g", r.reply)
            self.assertIn("25-30", r.reply)

    def test_no_verbatim_repeat(self):
        e = MindEngine(persona=get_persona("generic"), bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled())
        replies = [e.turn("你好", now=1000.0 + i * 60).reply for i in range(6)]
        self.assertEqual(len(replies), len(set(replies)))

    def test_empty_and_huge_input_do_not_crash(self):
        e = MindEngine(persona=get_persona("generic"), bridge=PyBotBridge.disabled(), kb=FeynmanKB.disabled())
        r1 = e.turn("", now=1000.0)
        self.assertTrue(r1.reply)
        r2 = e.turn("啊" * 20_000, now=1060.0)
        self.assertTrue(r2.reply)
        self.assertIn(r2.base_reply.rstrip("。"), r2.reply)


class TestPyBotBridge(unittest.TestCase):
    def test_uses_verified_turn_api_and_shares_memory(self):
        calls = []

        class FakeRuntime:
            @staticmethod
            def turn(text, memory):
                memory["n"] = memory.get("n", 0) + 1
                calls.append(text)
                return {"reply": f"ok:{memory['n']}", "module": "fake"}

        bridge = PyBotBridge(runtime=FakeRuntime())
        self.assertEqual(bridge.reply("hi"), ("ok:1", "pybot:turn"))
        self.assertEqual(bridge.reply("again"), ("ok:2", "pybot:turn"))
        self.assertEqual(calls, ["hi", "again"])
        self.assertEqual(bridge.memory["n"], 2)

    def test_unavailable_is_reported(self):
        from unittest import mock

        from omega_mind import pybot_bridge

        bridge = PyBotBridge(runtime=None)
        with mock.patch.object(pybot_bridge, "_local_checkout", return_value=None):
            reply, source = bridge.reply("hi")
        self.assertIsNone(reply)
        self.assertIn("unavailable", source)

    def test_disabled_bridge_never_imports(self):
        bridge = PyBotBridge.disabled()
        self.assertFalse(bridge.available())
        self.assertEqual(bridge.reply("hi"), (None, "unavailable"))


class TestCubieInventory(unittest.TestCase):
    def test_exactly_26(self):
        self.assertEqual(len(CUBIES), 26)
        self.assertEqual(len({c.name for c in CUBIES}), 26)


if __name__ == "__main__":
    unittest.main()
