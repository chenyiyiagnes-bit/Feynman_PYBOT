"""Abstention, follow-up-gap and hedge lines belong to the persona.

They used to be module constants in ``engine.py``: every package said the
same sentence, and changing it meant editing code. They are now voice data
(``PersonaConfig``); the engine only picks the language variant.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.config import persona_from_data
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


class TestVoiceMapping(unittest.TestCase):
    def test_voice_keys_become_lines(self):
        cfg = persona_from_data({"id": "x", "voice": {
            "abstain_zh": "甲", "abstain_en": "A",
            "followup_gap_zh": "乙", "followup_gap_en": "B",
            "hedge_zh": "丙", "hedge_en": "C",
        }})
        self.assertEqual(
            (cfg.abstain, cfg.abstain_en, cfg.followup_gap,
             cfg.followup_gap_en, cfg.hedge, cfg.hedge_en),
            ("甲", "A", "乙", "B", "丙", "C"))

    def test_a_package_that_says_nothing_keeps_the_neutral_defaults(self):
        cfg = persona_from_data({"id": "x", "voice": {}})
        self.assertTrue(cfg.abstain)
        self.assertTrue(cfg.abstain_en)
        self.assertNotEqual(cfg.abstain, cfg.abstain_en)

    def test_a_line_must_be_a_string_not_a_list(self):
        cfg = persona_from_data({"id": "x", "voice": {"abstain_zh": "甲"}})
        self.assertIsInstance(cfg.abstain, str)


class TestEngineUsesThePersonaLine(unittest.TestCase):
    def build(self, voice: dict, language: str = "zh") -> MindEngine:
        cfg = persona_from_data({"id": "custom", "voice": voice})
        return MindEngine(persona=cfg, bridge=PyBotBridge.disabled(),
                          kb=FeynmanKB.disabled(), language=language)

    def test_an_unanswerable_question_says_the_persona_line(self):
        engine = self.build({"abstain_zh": "这题我真答不上来。"})
        reply = engine.turn("今天股票怎么买？", now=1000.0).reply
        self.assertIn("这题我真答不上来", reply)

    def test_the_english_mind_says_the_english_line(self):
        engine = self.build({"abstain_en": "No idea, honestly."}, language="en")
        reply = engine.turn("How do I buy stocks?", now=1000.0).reply
        self.assertIn("No idea, honestly", reply)


if __name__ == "__main__":
    unittest.main()
