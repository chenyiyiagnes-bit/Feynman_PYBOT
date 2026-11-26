"""The 12 behavioural scenarios from spec.md."""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


def engine(persona: str = "generic", store: str | None = None) -> MindEngine:
    t = 1000.0
    return MindEngine(
        persona=get_persona(persona),
        store_path=store,
        clock=lambda: t,
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        auto_save=False,
    )


class TestScenarios(unittest.TestCase):
    # 1
    def test_criticism_defense_decays_within_three_turns(self):
        e = engine()
        r = e.turn("你刚才说错了", now=1000.0)
        peak = r.state["defensiveness"]
        self.assertGreater(peak, 0.20)
        values = []
        for i in range(1, 4):
            e.turn("嗯", now=1000.0 + 60 * i)
            values.append(e.state.get("defensiveness"))
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertLess(values[2], 0.10)

    # 2
    def test_apology_lowers_defense_faster_than_neutral(self):
        a, b = engine(), engine()
        a.turn("你刚才说错了", now=1000.0)
        b.turn("你刚才说错了", now=1000.0)
        a.turn("今天天气如何", now=1060.0)
        b.turn("对不起，是我错了", now=1060.0)
        self.assertLess(b.state.get("defensiveness"), a.state.get("defensiveness") - 0.05)

    # 3
    def test_three_compliments_diminish(self):
        e = engine()
        prev = e.state.get("dopamine")
        increments = []
        for i in range(4):
            e.turn("太厉害了，真有意思", now=1000.0 + 10 * i)
            cur = e.state.get("dopamine")
            increments.append(round(cur - prev, 4))
            prev = cur
        self.assertTrue(all(x > 0 for x in increments), increments)
        self.assertGreater(increments[0], increments[1])
        self.assertGreater(increments[1], increments[2])
        self.assertGreater(increments[2], increments[3])

    # 4
    def test_trauma_cools_next_two_turns(self):
        e = engine()
        e.turn("听说公司要裁员了", now=1000.0)
        self.assertGreater(e.state.get("cortisol"), 0.55)
        t1 = e.turn("今天天气不错", now=1060.0).directive.tone
        t2 = e.turn("今天天气不错", now=1120.0).directive.tone
        self.assertEqual(t1, "cold")
        self.assertEqual(t2, "cold")
        t4 = e.turn("今天天气不错", now=1300.0).directive.tone
        self.assertNotEqual(t4, "cold")

    # 5
    def test_friend_vs_stranger_same_provocation(self):
        friend, stranger = engine(), engine()
        friend.state.set("affinity", 0.85)
        stranger.state.set("affinity", 0.05)
        f = friend.turn("你真是个废物", now=1000.0).directive
        s = stranger.turn("你真是个废物", now=1000.0).directive
        self.assertGreater(f.warmth - s.warmth, 0.15)
        self.assertEqual(f.tone, "warm")
        self.assertNotEqual(s.tone, "warm")

    # 6
    def test_silence_triggers_proactive_once(self):
        e = engine()
        e.state.set("affinity", 0.80)
        e.state.set("dopamine", 0.20)
        e.last_interaction = 0.0
        first = e.tick(now=1000.0)
        self.assertIsNotNone(first)
        self.assertIsNone(e.tick(now=1010.0))
        later = e.tick(now=3000.0)
        self.assertIsNotNone(later)
        self.assertNotEqual(first, later)

    # 7
    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "mind.json")
            e = engine(store=path)
            e.turn("聊聊裁员这件事", now=1000.0)
            e.save()
            e2 = MindEngine.load(
                path,
                persona=get_persona("generic"),
                clock=lambda: 1000.0,
                bridge=PyBotBridge.disabled(),
                kb=FeynmanKB.disabled(),
            )
            self.assertEqual(e.state.snapshot(), e2.state.snapshot())
            self.assertEqual(e.memory, e2.memory)
            self.assertEqual(e2.turn_index, 1)
            r = e2.turn("继续", now=1060.0)
            self.assertEqual(r.turn, 1)

    # 8
    def test_catchphrase_rate_limit(self):
        e = engine()
        for i in range(10):
            e.turn("嗯", now=1000.0 + 10 * i)
        history = list(e.renderer.applied)
        for start in range(len(history) - 4):
            window = history[start:start + 5]
            counts = Counter(p for _, p in window)
            self.assertTrue(all(c <= 2 for c in counts.values()), window)

    # 9
    def test_feynman_correction_raises_dopamine(self):
        e = engine("feynman-zh")
        before = e.state.get("dopamine")
        r = e.turn("你这里说错了", now=1000.0)
        self.assertGreater(e.state.get("dopamine"), before + 0.15)
        self.assertLess(e.state.get("cortisol"), 0.20)
        self.assertLess(e.state.get("defensiveness"), 0.05)
        self.assertNotIn("corner_shame", [a.name for a in r.activations])

    # 10
    def test_feynman_jargon_raises_cortisol(self):
        e = engine("feynman-zh")
        r = e.turn("根据权威专家的高深理论，这个概念的定义是这样的", now=1000.0)
        self.assertGreater(e.state.get("cortisol"), 0.35)
        self.assertIn("corner_pedantry", [a.name for a in r.activations])
        self.assertEqual(r.directive.tone, "sharp")

    # 11
    def test_state_stays_bounded_under_sustained_stress(self):
        e = engine()
        for i in range(50):
            e.turn("你真是个废物", now=1000.0 + i)
        for dim, value in e.state.values.items():
            if dim == "valence":
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)
            else:
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        e.state.advance(1000.0 + 50 + 21600.0, e.persona)
        for dim in ("cortisol", "defensiveness", "valence", "dopamine"):
            self.assertAlmostEqual(e.state.get(dim), e.persona.baseline(dim), places=3)
        # affinity is intentionally slow (tau = 1 day): it must NOT snap back.
        self.assertLess(e.state.get("affinity"), e.persona.baseline("affinity") - 0.05)

    # 12
    def test_no_verbatim_repeat_same_input(self):
        e = engine()
        replies = [e.turn("你好", now=1000.0 + 60 * i).reply for i in range(6)]
        self.assertEqual(len(replies), len(set(replies)))

    # extra: knowledge path keeps facts
    def test_knowledge_reply_keeps_facts(self):
        e = engine()
        r = e.turn("为什么火会燃烧", now=1000.0)
        self.assertIn("氧", r.base_reply)
        self.assertIn("氧", r.reply)
        self.assertEqual(r.source, "builtin_kb")


if __name__ == "__main__":
    unittest.main()
