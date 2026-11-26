"""The scaffolder must produce a persona the loader and engine accept."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.runtime import build
from bridge import load
from tools.new_persona import scaffold

ROOT = Path(__file__).resolve().parents[1]


class TestScaffold(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_generic_base_is_valid(self):
        scaffold("demo", "演示角色", "generic", self.root)
        persona = load(self.root / "demo" / "persona.json")
        self.assertEqual(persona.id, "demo")
        self.assertEqual(persona.name, "演示角色")
        self.assertEqual(len(persona.cubies), 26)
        self.assertTrue((self.root / "demo" / "knowledge" / "README.md").is_file())

    def test_from_feynman_keeps_tuning(self):
        scaffold("mini", "", "feynman-zh", self.root)
        persona = load(self.root / "mini" / "persona.json")
        self.assertAlmostEqual(persona.baselines["dopamine"], 0.60)
        self.assertTrue(persona.switches["jargon_aversion"])
        self.assertEqual(persona.id, "mini")

    def test_overwrite_refused(self):
        scaffold("demo", "", "generic", self.root)
        with self.assertRaises(SystemExit):
            scaffold("demo", "", "generic", self.root)

    def test_new_persona_runs_without_cards(self):
        scaffold("demo", "", "generic", self.root)
        engine = build("demo", personas_root=self.root, enable_bridge=False)
        self.assertEqual(engine.kb.mode, "disabled")
        self.assertTrue(engine.turn("你好", now=1000.0).reply)

    def test_unknown_base_fails(self):
        with self.assertRaises(Exception):
            scaffold("demo", "", "nope", self.root)


if __name__ == "__main__":
    unittest.main()
