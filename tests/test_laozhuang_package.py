"""The Lao-Zhuang packages are data, and the data has to hold up.

One package per language, no corpus in the repo: what ships is the persona and
the cards. These tests check the shippable assets only — the classical text and
its vernacular are kept locally and are deliberately not required here.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import unicodedata
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"zh": ROOT / "personas" / "laozhuang-zh",
            "en": ROOT / "personas" / "laozhuang-en"}
DIMENSIONS = {"dopamine", "cortisol", "valence", "arousal", "affinity",
              "defensiveness", "energy"}
ROLE_COUNTS = {"core": 6, "edge": 12, "corner": 8}
FACES = set("FBLRUD")


def letters_outside(text: str, lang: str) -> list:
    letters = {ch for ch in text if ch.isalpha()}
    prefixes = ("CJK ", "IDEOGRAPHIC ") if lang == "zh" else ("LATIN ",)
    return sorted(ch for ch in letters
                  if not unicodedata.name(ch, "").startswith(prefixes))


def load_persona(lang: str) -> dict:
    return json.loads((PACKAGES[lang] / "persona.json").read_text(encoding="utf-8"))


def load_cards(lang: str) -> list:
    path = PACKAGES[lang] / "knowledge" / "cards.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


class PersonaShape(unittest.TestCase):
    def test_identity_and_language_per_package(self) -> None:
        for lang, directory in PACKAGES.items():
            with self.subTest(package=lang):
                persona = load_persona(lang)
                self.assertEqual(persona["schema"], 1)
                self.assertEqual(persona["id"], directory.name)
                self.assertEqual(persona["language"], lang)
                self.assertTrue(persona["name"].strip())

    def test_twenty_six_cubies_with_roles_and_faces(self) -> None:
        for lang in PACKAGES:
            persona = load_persona(lang)
            cubies = persona["cubies"]
            self.assertEqual(len(cubies), 26, lang)
            self.assertEqual(Counter(c["role"] for c in cubies), ROLE_COUNTS, lang)
            self.assertEqual(len({c["name"] for c in cubies}), 26, lang)
            self.assertTrue(all(c["face"] in FACES for c in cubies), lang)

    def test_baselines_taus_and_triggers(self) -> None:
        for lang, persona in ((lang, load_persona(lang)) for lang in PACKAGES):
            with self.subTest(package=lang):
                self.assertEqual(set(persona["baselines"]), DIMENSIONS)
                self.assertEqual(set(persona["taus"]), DIMENSIONS)
                for dimension in DIMENSIONS:
                    self.assertTrue(math.isfinite(persona["baselines"][dimension]))
                    self.assertGreater(persona["taus"][dimension], 0)
                for cubie in persona["cubies"]:
                    for pattern, gain in cubie["triggers"]:
                        re.compile(pattern)
                        self.assertGreater(gain, 0)
                    # The runtime copies these two tables from the cubies, so a
                    # hand edit that drifts would silently change behaviour.
                    self.assertEqual(persona["override_triggers"][cubie["name"]],
                                     [p for p, _ in cubie["triggers"]])

    def test_voice_speaks_only_its_own_language(self) -> None:
        for lang, persona in ((lang, load_persona(lang)) for lang in PACKAGES):
            voice = persona["voice"]
            keys = [key for key in voice if key.endswith("_" + lang)]
            self.assertTrue(keys, lang)
            for key in keys:
                for phrase in voice[key]:
                    self.assertFalse(letters_outside(phrase, lang) and lang == "en",
                                     f"{key}: {phrase}")

    def test_no_corpus_ships_with_either_package(self) -> None:
        for lang, directory in PACKAGES.items():
            with self.subTest(package=lang):
                persona = load_persona(lang)
                self.assertIsNone(persona["knowledge"].get("corpus"))
                corpus = directory / "corpus"
                shipped = list(corpus.rglob("*.jsonl")) if corpus.is_dir() else []
                self.assertEqual(shipped, [], f"{lang}: corpus files must stay local")


class Cards(unittest.TestCase):
    def test_required_fields_and_unique_ids(self) -> None:
        for lang in PACKAGES:
            seen = set()
            for card in load_cards(lang):
                label = f"{lang}:{card.get('id')}"
                for field in ("id", "type", "title", "text", "tags", "lang", "source"):
                    self.assertIn(field, card, label)
                self.assertNotIn(card["id"], seen, f"duplicate {label}")
                seen.add(card["id"])
                self.assertEqual(card["lang"], lang, label)
                self.assertTrue(card["tags"], label)

    def test_cards_stay_in_their_language(self) -> None:
        for lang in PACKAGES:
            for card in load_cards(lang):
                label = f"{lang}:{card['id']}"
                foreign = letters_outside(card["text"], lang)
                self.assertFalse(foreign, f"{label}: letters outside {lang}: {foreign}")
                for alias in card.get("aliases", []):
                    self.assertFalse(letters_outside(alias, lang),
                                     f"{label}: alias {alias}")

    def test_english_cards_translate_chinese_ones(self) -> None:
        zh = {card["id"]: card for card in load_cards("zh")}
        en = load_cards("en")
        for card in en:
            suffix = card["id"].rsplit("-", 1)[-1]
            counterpart = f"lz-zh-{suffix}"
            self.assertIn(counterpart, zh, card["id"])
            self.assertNotEqual(card["text"], zh[counterpart]["text"], card["id"])

    def test_packages_load_through_the_runtime(self) -> None:
        from app import runtime
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {runtime.STATE_DIR_ENV: tmp}):
                for lang in PACKAGES:
                    with self.subTest(package=lang):
                        engine = runtime.build(PACKAGES[lang].name)
                        self.assertEqual(engine.language, lang)
                        self.assertEqual(engine.kb.language, lang)
                        self.assertGreater(len(engine.kb.docs), 0)


if __name__ == "__main__":
    unittest.main()
