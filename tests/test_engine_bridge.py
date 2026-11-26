"""The bridge must be invisible when off and fail-closed when on."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind.config import get_persona  # noqa: E402
from omega_mind.engine import MindEngine  # noqa: E402
from omega_mind.feynman_kb import FeynmanKB  # noqa: E402
from omega_mind.pybot_bridge import PyBotBridge  # noqa: E402

from bridge.responder import BridgeResult  # noqa: E402

QUESTION = "光为什么沿直线走？"


def make_engine(responder=None):
    return MindEngine(
        persona=get_persona("feynman-zh"),
        bridge=PyBotBridge.disabled(),
        kb=FeynmanKB.disabled(),
        left_responder=responder,
        auto_save=False,
    )


class FakeResponder:
    def __init__(self, text="光在空旷的空间里走最短的路径，这叫费马原理。",
                 raises=False):
        self.text = text
        self.raises = raises
        self.calls = []

    def reply(self, text, state, persona, cube_depth=0, cube_verbosity=0.0,
              nonce=0, session_id=None):
        self.calls.append({
            "text": text, "state": dict(state), "persona": persona,
            "cube_depth": cube_depth, "cube_verbosity": cube_verbosity,
            "nonce": nonce, "session_id": session_id,
        })
        if self.raises:
            raise RuntimeError("typesetter exploded")
        if self.text is None:
            return None
        return BridgeResult(text=self.text, raw={}, params={})


class TestOffPathIdentical(unittest.TestCase):
    def test_none_responder_changes_nothing(self):
        plain = make_engine().turn(QUESTION, now=1000.0)
        with_none = make_engine(FakeResponder(text=None)).turn(QUESTION, now=1000.0)
        self.assertEqual(plain.reply, with_none.reply)
        self.assertEqual(plain.base_reply, with_none.base_reply)
        self.assertEqual(plain.source, with_none.source)

    def test_no_responder_is_never_consulted(self):
        r = FakeResponder()
        make_engine().turn(QUESTION, now=1000.0)
        self.assertEqual(r.calls, [])


class TestResponderPriority(unittest.TestCase):
    def test_typesetter_fills_a_card_miss(self):
        r = FakeResponder()
        result = make_engine(r).turn(QUESTION, now=1000.0)
        self.assertEqual(result.source, "left:typesetter")
        self.assertEqual(result.base_reply, r.text)

    def test_raised_exception_falls_back(self):
        result = make_engine(FakeResponder(raises=True)).turn(QUESTION, now=1000.0)
        self.assertNotEqual(result.source, "left:typesetter")
        self.assertTrue(result.base_reply)

    def test_english_output_is_rejected(self):
        r = FakeResponder(text="Light takes the shortest path.")
        result = make_engine(r).turn(QUESTION, now=1000.0)
        self.assertNotEqual(result.source, "left:typesetter")

    def test_empty_output_is_rejected(self):
        result = make_engine(FakeResponder(text="   ")).turn(QUESTION, now=1000.0)
        self.assertNotEqual(result.source, "left:typesetter")


class TestResponderContract(unittest.TestCase):
    def test_gets_affect_state_and_cube_bias(self):
        r = FakeResponder()
        engine = make_engine(r)
        engine.turn(QUESTION, now=1000.0)
        call = r.calls[0]
        self.assertEqual(set(call["state"]), {
            "dopamine", "cortisol", "valence", "arousal",
            "affinity", "defensiveness", "energy"})
        self.assertIs(call["persona"], engine.persona)
        self.assertEqual(call["session_id"], "default")
        self.assertIsInstance(call["nonce"], int)
        self.assertIsInstance(call["cube_depth"], int)
        self.assertIsInstance(call["cube_verbosity"], float)

    def test_kb_card_still_wins(self):
        engine = make_engine(FakeResponder())
        found = mock.Mock()
        found.text = "卡片上的答案。"
        found.source = "feynman_kb:test"
        found.to_dict.return_value = {}
        with mock.patch.object(engine.kb, "lookup", return_value=found):
            result = engine.turn(QUESTION, now=1000.0)
        self.assertEqual(result.base_reply, "卡片上的答案。")
        self.assertTrue(result.source.startswith("feynman_kb"))


if __name__ == "__main__":
    unittest.main()
