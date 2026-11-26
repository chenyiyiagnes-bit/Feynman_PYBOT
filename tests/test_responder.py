"""Tests for the default-off responder contract."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bridge import persona as persona_mod
from bridge.responder import ENV_FLAG, LeftResponder

ROOT = Path(__file__).resolve().parents[1]
FEYNMAN = ROOT / "personas" / "feynman" / "persona.json"


class TestDefaultOff(unittest.TestCase):
    def test_disabled_when_env_absent(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_FLAG, None)
            r = LeftResponder()
            self.assertFalse(r.enabled)
            self.assertFalse(r.available())
            p = persona_mod.load(FEYNMAN)
            self.assertIsNone(r.reply("你好", p.baselines, p))

    def test_disabled_when_env_not_one(self):
        with mock.patch.dict(os.environ, {ENV_FLAG: "0"}):
            self.assertFalse(LeftResponder().enabled)

    def test_explicit_disable_beats_env(self):
        with mock.patch.dict(os.environ, {ENV_FLAG: "1"}):
            self.assertFalse(LeftResponder(enabled=False).enabled)


class TestFailsClosed(unittest.TestCase):
    def test_missing_vault_returns_none_not_raise(self):
        r = LeftResponder(vault_path=Path("/nonexistent/vault.json"), enabled=True)
        self.assertFalse(r.available())
        p = persona_mod.load(FEYNMAN)
        self.assertIsNone(r.reply("量子是什么", p.baselines, p))
        self.assertIn("vault missing", r.last_error or "")

    def test_params_for_uses_persona_baselines(self):
        r = LeftResponder(enabled=False)
        p = persona_mod.load(FEYNMAN)
        params = r.params_for(p.baselines, p)
        self.assertEqual(set(params.operators), {
            "micro", "cross", "boundary", "reversal",
            "anti_intuition", "jargon_strip", "least_action", "cargo_cult",
        })
        self.assertGreater(params.operators["jargon_strip"], 0.0)

    def test_cube_bias_moves_style(self):
        r = LeftResponder(enabled=False)
        p = persona_mod.load(FEYNMAN)
        low = r.params_for(p.baselines, p, cube_verbosity=-1.0)
        high = r.params_for(p.baselines, p, cube_verbosity=1.0)
        self.assertLess(low.style_deg, high.style_deg)


class TestWithRealLeftBrain(unittest.TestCase):
    """Only runs when a vault is available on this machine."""

    def setUp(self):
        self.vault = Path(os.getenv(
            "QBOT_FEYNMAN_VAULT",
            ROOT / "left" / "data" / "feynman_style_vault.json"))
        if not self.vault.is_file():
            self.skipTest(f"no vault at {self.vault}")

    def test_reply_returns_text_or_none_without_raising(self):
        r = LeftResponder(enabled=True, vault_path=self.vault)
        if not r.available():
            self.skipTest(f"left brain unavailable: {r.last_error}")
        p = persona_mod.load(FEYNMAN)
        result = r.reply("光为什么沿直线走？", p.baselines, p, nonce=7)
        if result is not None:
            self.assertTrue(result.text.strip())
            self.assertIn("style_deg", result.params)


if __name__ == "__main__":
    unittest.main()
