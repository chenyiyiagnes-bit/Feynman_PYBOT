"""The assembly layer: persona.json in, runnable mind out."""

from __future__ import annotations

import unittest
from pathlib import Path

from app.runtime import (
    build,
    knowledge_dir,
    resolve_persona,
    to_config,
)
from bridge import PersonaError, load

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "personas"


class TestResolve(unittest.TestCase):
    def test_by_id(self):
        path = resolve_persona("feynman-zh", PERSONAS)
        self.assertEqual(path.name, "persona.json")

    def test_by_directory(self):
        self.assertTrue(resolve_persona(PERSONAS / "feynman-zh").is_file())

    def test_by_file(self):
        target = PERSONAS / "laozhuang-zh" / "persona.json"
        self.assertEqual(resolve_persona(target), target.resolve())

    def test_unknown_id_raises(self):
        with self.assertRaises(PersonaError):
            resolve_persona("nobody", PERSONAS)


class TestConfigMapping(unittest.TestCase):
    def setUp(self):
        self.persona = load(PERSONAS / "feynman-zh" / "persona.json")
        self.cfg = to_config(self.persona)

    def test_baselines_and_taus(self):
        for dim, value in self.persona.baselines.items():
            self.assertAlmostEqual(self.cfg.baselines[dim], value)
        for dim, value in self.persona.taus.items():
            self.assertAlmostEqual(self.cfg.taus[dim], value)

    def test_voice_and_switches(self):
        self.assertEqual(self.cfg.catchphrase_prefixes,
                         self.persona.voice["prefixes_zh"])
        self.assertEqual(self.cfg.ending_particles,
                         self.persona.voice["endings_zh"])
        self.assertTrue(self.cfg.jargon_aversion)
        self.assertTrue(self.cfg.correction_as_curiosity)
        self.assertAlmostEqual(self.cfg.politeness, 0.1)

    def test_cubie_overrides(self):
        self.assertEqual(self.cfg.disable_cubies, self.persona.disable_cubies)
        self.assertEqual(self.cfg.extra_effects, self.persona.extra_effects)

    def test_laozhuang_maps_too(self):
        cfg = to_config(load(PERSONAS / "laozhuang-zh" / "persona.json"))
        self.assertLess(cfg.baselines["cortisol"], 0.12)
        self.assertEqual(cfg.name, "laozhuang-zh")


class TestBuild(unittest.TestCase):
    def test_feynman_builds_with_cards(self):
        kb_dir = knowledge_dir(load(PERSONAS / "feynman-zh" / "persona.json"))
        self.assertIsNotNone(kb_dir)
        engine = build("feynman-zh", personas_root=PERSONAS, enable_bridge=False)
        self.assertGreater(len(engine.kb.docs), 200)
        self.assertIsNone(engine.left_responder)

    def test_bridge_off_by_default(self):
        engine = build("feynman-zh", personas_root=PERSONAS, enable_bridge=False)
        self.assertIsNone(engine.left_responder)

    def test_turn_answers_from_cards(self):
        engine = build("feynman-zh", personas_root=PERSONAS, enable_bridge=False)
        result = engine.turn("原子是什么？", now=1000.0)
        self.assertTrue(result.reply)
        self.assertTrue(result.source.startswith("feynman-zh"), result.source)

    def test_no_knowledge_dir_still_builds(self):
        engine = build("laozhuang-zh", personas_root=PERSONAS, enable_bridge=False)
        self.assertEqual(engine.kb.mode, "disabled")
        self.assertTrue(engine.turn("你好", now=1000.0).reply)


if __name__ == "__main__":
    unittest.main()
