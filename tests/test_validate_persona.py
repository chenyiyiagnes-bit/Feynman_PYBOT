"""tools/validate_persona.py: shipped personas pass, broken ones fail loudly."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import validate_persona

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "personas"


class TestShippedPersonas(unittest.TestCase):
    def test_every_shipped_persona_is_valid(self):
        files = sorted(PERSONAS.glob("*/persona.json"))
        self.assertTrue(files)
        for path in files:
            with self.subTest(persona=path.parent.name):
                errors, _ = validate_persona.validate(path)
                self.assertEqual(errors, [], path)


class TestBrokenPersonas(unittest.TestCase):
    def _write(self, data) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(data, tmp, ensure_ascii=False)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return Path(tmp.name)

    def test_wrong_schema(self):
        errors, _ = validate_persona.validate(
            self._write({"schema": 2}))
        self.assertTrue(any("schema" in e for e in errors))

    def test_missing_dimensions(self):
        errors, _ = validate_persona.validate(self._write({"schema": 1}))
        self.assertTrue(any("dopamine" in e for e in errors))

    def test_unreadable_json(self):
        path = Path(tempfile.mkdtemp()) / "p.json"
        path.write_text("{broken", encoding="utf-8")
        errors, _ = validate_persona.validate(path)
        self.assertTrue(errors)

    def test_non_object_toplevel(self):
        errors, _ = validate_persona.validate(self._write([1, 2]))
        self.assertTrue(any("JSON object" in e for e in errors))


class TestFieldChecks(unittest.TestCase):
    """Boundary types the audit flagged: bools are not numbers, fields required."""

    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(
            (PERSONAS / "feynman" / "persona.json").read_text(encoding="utf-8"))

    def _validate(self, mutate):
        data = json.loads(json.dumps(self.base))
        mutate(data)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
        json.dump(data, tmp, ensure_ascii=False)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return validate_persona.validate(Path(tmp.name))[0]

    def test_boolean_gain_is_rejected(self):
        errors = self._validate(
            lambda d: d["cubies"][0]["triggers"][0].__setitem__(1, True))
        self.assertTrue(any("gain" in e for e in errors))

    def test_missing_language_is_rejected(self):
        self.assertTrue(self._validate(lambda d: d.pop("language")))

    def test_bad_language_is_rejected(self):
        errors = self._validate(lambda d: d.__setitem__("language", "fr"))
        self.assertTrue(any("language" in e for e in errors))

    def test_extra_effects_checked(self):
        errors = self._validate(
            lambda d: d.__setitem__("extra_effects", {"core_ego": {"nope": 1.0}}))
        self.assertTrue(any("dimension" in e for e in errors))
        errors = self._validate(
            lambda d: d.__setitem__("extra_effects", {"core_ego": {"valence": "x"}}))
        self.assertTrue(any("number" in e for e in errors))

    def test_boolean_baseline_is_rejected(self):
        errors = self._validate(
            lambda d: d["baselines"].__setitem__("energy", True))
        self.assertTrue(any("energy" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
