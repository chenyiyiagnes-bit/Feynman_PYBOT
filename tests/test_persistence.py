"""Persisted state: a second run remembers the first."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.runtime import build

PERSONAS = Path(__file__).resolve().parents[1] / "personas"


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "mind.json"

    def test_state_round_trips(self):
        first = build("feynman", personas_root=PERSONAS,
                      enable_bridge=False,
                      state_path=self.state, clock=lambda: 1000.0)
        first.turn("我叫小明", now=1000.0)
        first.turn("原子是什么？", now=1060.0)
        first.save()
        self.assertTrue(self.state.is_file())

        second = build("feynman", personas_root=PERSONAS,
                       enable_bridge=False,
                       state_path=self.state, clock=lambda: 1100.0)
        self.assertEqual(second.turn_index, first.turn_index)
        self.assertEqual(second.state.snapshot(), first.state.snapshot())
        self.assertTrue(second.turn("继续", now=1120.0).reply)

    def test_no_state_file_builds_fresh(self):
        engine = build("feynman", personas_root=PERSONAS,
                       enable_bridge=False,
                       state_path=Path(self._tmp.name) / "absent.json")
        self.assertEqual(engine.turn_index, 0)


if __name__ == "__main__":
    unittest.main()
