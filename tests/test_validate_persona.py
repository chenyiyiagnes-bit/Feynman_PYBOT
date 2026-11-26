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


if __name__ == "__main__":
    unittest.main()
