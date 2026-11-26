"""One KB instance = one package = one language.

These tests build their own tiny packages in a temporary directory: nothing
here reads corpus material, and nothing checks a persona's contents. What
they pin is the contract the right brain owes every package:

* the language comes from the package's own ``persona.json``;
* a package is indexed once, in its language, and never merged with another;
* another language is a miss, not a fallback to a second index;
* derived state (learned cards, aliases) lives under the package that owns
  the cards, so teaching one mind cannot change another's answers;
* a mind that holds one package cannot silently switch to another.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB, package_language
from omega_mind.pybot_bridge import PyBotBridge


def make_package(root: Path, persona_id: str, language: str,
                 rows) -> Path:
    """Write the smallest package the loader accepts; return knowledge dir."""
    package = root / persona_id
    knowledge = package / "knowledge"
    knowledge.mkdir(parents=True)
    (package / "persona.json").write_text(
        json.dumps({"id": persona_id, "language": language}),
        encoding="utf-8")
    (knowledge / "cards.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    return knowledge


def zh_row(card_id: str, title: str, text: str) -> dict:
    return {"id": card_id, "type": "physics_card", "title": title,
            "text": text, "tags": [title], "aliases": []}


def en_row(card_id: str, title: str, text: str) -> dict:
    return {"id": card_id, "title": title, "text": text,
            "tags": [title], "aliases": []}


def offline_mind(kb: FeynmanKB, language: str) -> MindEngine:
    return MindEngine(
        persona=get_persona("generic"),
        store_path=None,
        clock=lambda: 1000.0,
        bridge=PyBotBridge.disabled(),
        kb=kb,
        auto_save=False,
        language=language,
    )


class TestPackageLanguage(unittest.TestCase):
    def test_reads_the_declared_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zh = make_package(root, "tiny-zh", "zh",
                              [zh_row("a", "熵", "熵是无序的度量。")])
            en = make_package(root, "tiny-en", "en",
                              [en_row("a", "Entropy", "Entropy measures disorder.")])
            self.assertEqual(package_language(zh.parent), "zh")
            self.assertEqual(package_language(en.parent), "en")
            self.assertEqual(package_language(zh), "zh")  # knowledge dir too

    def test_unreadable_package_is_unclassified(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "no-persona"
            empty.mkdir()
            self.assertIsNone(package_language(empty))
            (empty / "persona.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(package_language(empty))
            (empty / "persona.json").write_text(
                json.dumps({"id": "x", "language": "fr"}), encoding="utf-8")
            self.assertIsNone(package_language(empty))

    def test_loader_adopts_the_package_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            knowledge = make_package(
                Path(tmp), "tiny-en", "en",
                [en_row("a", "Entropy", "Entropy measures disorder.")])
            kb = FeynmanKB(knowledge_dir=str(knowledge))
            self.assertEqual(kb.language, "en")
            self.assertIn(":en:", kb.mode)
            self.assertTrue(kb.answer("What is entropy?")[1].startswith(
                kb.source_root + ":"))
            self.assertEqual(kb.source_prefix, "feynman_kb")


class TestOnePackagePerInstance(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_another_language_is_a_miss_not_a_fallback(self):
        knowledge = make_package(
            self.root, "tiny-zh", "zh",
            [zh_row("entropy", "熵", "熵是无序的度量，越多的排列越高的熵。")])
        kb = FeynmanKB(knowledge_dir=str(knowledge))
        self.assertTrue(kb.available())
        text, source = kb.answer("熵是什么")
        self.assertIsNotNone(text)
        # The same instance asked in English finds nothing: no second index
        # was ever loaded, and nothing falls through to one.
        self.assertEqual(kb.search("what is entropy", language="en"), [])
        self.assertEqual(kb.answer("what is entropy", language="en"),
                         (None, f"{kb.source_root}:no_match"))
        self.assertEqual(kb.teachable_units(language="en"), [])

    def test_language_is_part_of_the_source(self):
        knowledge = make_package(
            self.root, "tiny-en", "en",
            [en_row("entropy", "Entropy", "Entropy measures disorder.")])
        kb = FeynmanKB(knowledge_dir=str(knowledge), source_prefix="tiny-en")
        text, source = kb.answer("What is entropy?")
        self.assertIsNotNone(text)
        self.assertEqual(source, "tiny-en:entropy")
        self.assertEqual(kb.answer("熵是什么"), (None, "tiny-en:no_match"))
        self.assertEqual(kb.answer("熵是什么", language="zh"),
                         (None, "tiny-en:no_match"))

    def test_learned_state_belongs_to_its_package(self):
        first = make_package(
            self.root, "first-zh", "zh",
            [zh_row("entropy", "熵", "熵是无序的度量。")])
        second = make_package(
            self.root, "second-zh", "zh",
            [zh_row("entropy", "熵", "熵是无序的度量。")])
        a = FeynmanKB(knowledge_dir=str(first))
        b = FeynmanKB(knowledge_dir=str(second))
        self.assertEqual(a.learned.dir,
                         first.parent / "data" / "kb_learned")
        self.assertEqual(b.learned.dir,
                         second.parent / "data" / "kb_learned")
        self.assertTrue(a.learn_alias("entropy", "混乱程度"))
        # The alias is stored under the first package and the second mind
        # never sees it.
        self.assertTrue((first.parent / "data" / "kb_learned").is_dir())
        self.assertFalse((second.parent / "data" / "kb_learned").is_dir())
        self.assertEqual(list(b.learned.alias_items()), [])

    def test_a_kb_with_no_package_has_no_shared_store(self):
        kb = FeynmanKB(autoload=False)
        self.assertIsNone(kb.learned)
        self.assertFalse(kb.available())


class TestShippedPackages(unittest.TestCase):
    """The repo's own packages, when this machine has them."""

    @classmethod
    def setUpClass(cls):
        cls.zh = FeynmanKB(language="zh")
        cls.en = FeynmanKB(language="en")
        if not cls.zh.available() or not cls.en.available():
            raise unittest.SkipTest("both shipped packages are needed here")

    def test_each_language_finds_its_own_package(self):
        self.assertEqual(package_language(self.zh.dir.parent), "zh")
        self.assertEqual(package_language(self.en.dir.parent), "en")
        self.assertNotEqual(self.zh.dir, self.en.dir)

    def test_for_language_builds_a_package_local_kb(self):
        kb = FeynmanKB.for_language("en")
        self.assertIsNotNone(kb)
        self.assertEqual(kb.language, "en")
        self.assertEqual(package_language(kb.dir.parent), "en")
        data = kb.dir.parent / "data"
        self.assertEqual(kb.learned_dir, data / "kb_learned")
        self.assertEqual(kb.assoc_path, data / "assoc" / "assoc_edges.jsonl")
        self.assertEqual(kb.lexicon_dir, data / "lexicon")
        self.assertEqual(kb.source_prefix, kb.dir.parent.name)

    def test_chinese_index_holds_no_english_rows(self):
        for doc in self.zh.docs:
            if doc.language != "zh":
                self.fail(f"{doc.id} loaded into the Chinese index as "
                          f"{doc.language!r}")


class TestMindHoldsOnePackage(unittest.TestCase):
    def test_a_supplied_package_cannot_be_switched(self):
        en = FeynmanKB(language="en")
        if not en.available():
            self.skipTest("English package not installed")
        mind = offline_mind(en, "en")
        with self.assertRaises(ValueError) as caught:
            mind.set_language("zh")
        self.assertIn("package", str(caught.exception))
        self.assertEqual(mind.language, "en")
        self.assertIs(mind.kb, en)

    def test_an_implicit_kb_swaps_in_the_other_package(self):
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            auto_save=False, language="zh")
        replacement = FeynmanKB.for_language("en")
        if replacement is None:
            self.skipTest("English package not installed")
        mind.set_language("en")
        self.assertEqual(mind.kb.language, "en")
        self.assertNotEqual(mind.kb.mode, "disabled")
        self.assertIn(":en:", mind.kb.mode)

    def test_a_disabled_kb_still_allows_mode_switches(self):
        mind = offline_mind(FeynmanKB.disabled(), "zh")
        mind.set_language("en")
        self.assertEqual(mind.language, "en")


if __name__ == "__main__":
    unittest.main()
