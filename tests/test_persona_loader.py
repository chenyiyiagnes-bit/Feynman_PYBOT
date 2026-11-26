"""Tests for loading persona.json without either brain."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bridge import persona as persona_mod

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "personas"


class TestLoadRealPersonas(unittest.TestCase):
    def test_feynman_loads(self):
        p = persona_mod.load(PERSONAS / "feynman-zh" / "persona.json")
        self.assertEqual(p.id, "feynman-zh")
        self.assertEqual(len(p.cubies), 26)
        self.assertEqual(p.baselines["dopamine"], 0.60)
        self.assertTrue(p.switches["jargon_aversion"])
        self.assertIn("corner_shame", p.disable_cubies)

    def test_laozhuang_loads(self):
        p = persona_mod.load(PERSONAS / "laozhuang-zh" / "persona.json")
        self.assertEqual(p.id, "laozhuang-zh")
        self.assertEqual(len(p.cubies), 26)
        self.assertLess(p.baselines["cortisol"], 0.12)

    def test_discover_finds_both(self):
        found = persona_mod.discover(PERSONAS)
        self.assertIn("feynman-zh", found)
        self.assertIn("laozhuang-zh", found)


class TestRejectsBadPersonas(unittest.TestCase):
    def _write(self, data):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(data, tmp, ensure_ascii=False)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def _base(self):
        return json.loads((PERSONAS / "feynman-zh" / "persona.json").read_text(encoding="utf-8"))

    def test_missing_dimension(self):
        data = self._base()
        del data["baselines"]["cortisol"]
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(self._write(data))

    def test_wrong_cubie_count(self):
        data = self._base()
        data["cubies"] = data["cubies"][:-1]
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(self._write(data))

    def test_wrong_role_distribution(self):
        data = self._base()
        for cubie in data["cubies"]:
            if cubie["role"] == "edge":
                cubie["role"] = "core"
                break
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(self._write(data))

    def test_non_positive_tau(self):
        data = self._base()
        data["taus"]["energy"] = 0
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(self._write(data))

    def test_missing_file(self):
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(PERSONAS / "nobody" / "persona.json")

    def test_bad_schema(self):
        data = self._base()
        data["schema"] = 2
        with self.assertRaises(persona_mod.PersonaError):
            persona_mod.load(self._write(data))


if __name__ == "__main__":
    unittest.main()
