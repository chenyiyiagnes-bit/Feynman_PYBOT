"""Magnitude anchors: schema, curated routing, and the spoken line.

A scale anchor is a number Feynman actually gave, keyed to a card. It is
spoken only when the question asks for a size / speed / order of magnitude,
and only from data that carries its lecture as a source. The curated ask
list exists because ordinary retrieval drops "原子有多大": the noun is too
common in a physics corpus to clear the IDF cut, so nothing would ever
reach the card that has the number.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omega_mind import scales
from omega_mind.feynman_kb import FeynmanKB


def _data_present() -> bool:
    return scales.DEFAULT_PATH.exists() and bool(scales.load().scales)


def _scale(**over):
    base = dict(card="c", zh="半径约 1 埃", en="about one angstrom",
                source="FLP I-1")
    base.update(over)
    return scales.Scale(**base)


class TestScaleSchema(unittest.TestCase):
    def test_a_sourced_bilingual_row_is_valid(self):
        self.assertEqual(scales.problems(_scale()), [])

    def test_a_row_without_a_source_is_rejected(self):
        self.assertTrue(scales.problems(_scale(source="")))

    def test_english_may_not_contain_chinese(self):
        self.assertTrue(scales.problems(_scale(en="半径约 1 埃")))

    def test_an_english_ask_may_not_contain_chinese(self):
        self.assertTrue(scales.problems(_scale(asks_en=("原子多大",))))

    def test_load_drops_bad_rows_and_counts_them(self):
        path = Path(tempfile.mkdtemp()) / "scales.jsonl"
        rows = [
            _scale().__dict__,
            _scale(card="d", source="").__dict__,
            _scale(card="e").__dict__,
            _scale(card="e").__dict__,  # duplicate card
        ]
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                                  for r in rows), encoding="utf-8")
        book = scales.load(path)
        self.assertEqual(book.card_ids(), ["c", "e"])
        self.assertEqual(book.dropped, 2)


class TestScaleBook(unittest.TestCase):
    def test_line_comes_back_per_language(self):
        book = scales.ScaleBook([_scale()])
        self.assertEqual(book.line("c"), "半径约 1 埃")
        self.assertEqual(book.line("c", "en"), "about one angstrom")
        self.assertIsNone(book.line("missing"))

    def test_curated_ask_matches_regardless_of_punctuation(self):
        book = scales.ScaleBook([_scale(asks=("原子有多大",))])
        self.assertEqual(book.anchor_for("原子有多大？")[0], "c")
        self.assertIsNone(book.anchor_for("原子是什么"))

    def test_english_ask_matches_without_case(self):
        book = scales.ScaleBook([_scale(asks_en=("how big is an atom",))])
        self.assertEqual(book.anchor_for("How BIG is an atom?", "en")[0], "c")


class TestScaleAnswers(unittest.TestCase):
    def setUp(self):
        if not _data_present():
            self.skipTest("data/assoc/scales.jsonl not present")
        self.kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-scale-"))
        if not self.kb.available():
            self.skipTest("feynman-open knowledge not on this machine")

    def test_atom_size_question_reaches_the_atom_card_with_a_number(self):
        found = self.kb.lookup("原子有多大", scale=True)
        self.assertEqual(found.source, "feynman_kb:flp-i-01")
        self.assertIn("埃", found.scale)

    def test_light_speed_question_reaches_the_maxwell_card(self):
        found = self.kb.lookup("光速是多少", scale=True)
        self.assertEqual(found.source, "feynman_kb:flp-ii-18")
        self.assertIn("3×10⁸", found.scale)

    def test_an_ordinary_question_gets_no_anchor(self):
        found = self.kb.lookup("原子是什么", scale=True)
        self.assertEqual(found.source, "feynman_kb:flp-i-01")
        self.assertIsNone(found.scale)

    def test_scale_is_off_unless_asked_for(self):
        found = self.kb.lookup("原子有多大")
        self.assertIsNone(found.scale)

    def test_the_english_branch_never_speaks_chinese(self):
        # One package, one language: English answers come from the English
        # package, never from a second index bolted onto the Chinese one.
        en = FeynmanKB(language="en", learned_dir=tempfile.mkdtemp(prefix="omega-scale-en-"))
        if not en.available():
            self.skipTest("English package not installed")
        found = en.lookup("How fast is light?", scale=True)
        self.assertEqual(found.source, en.source_root + ":flp-ii-18")
        self.assertNotRegex(found.scale, r"[一-鿿]")

    def test_every_anchor_card_exists_and_cites_a_lecture(self):
        book = scales.load()
        known = {doc.id for doc in self.kb.docs}
        for scale in book.scales:
            self.assertIn(scale.card, known)
            self.assertTrue(scale.source.startswith("FLP"), scale.card)


class TestScaleEngine(unittest.TestCase):
    def _mind(self, language="zh"):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        if not _data_present():
            self.skipTest("data/assoc/scales.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-scale-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False, language=language)

    def test_a_size_question_gets_the_anchor_after_the_answer(self):
        mind = self._mind()
        result = mind.turn("原子有多大", now=1000.0)
        self.assertEqual(result.source, "feynman_kb:flp-i-01")
        self.assertIn("埃", result.reply)
        self.assertIn("埃", result.retrieval.get("scale") or "")

    def test_an_ordinary_question_stays_uncluttered(self):
        mind = self._mind()
        result = mind.turn("原子是什么", now=1000.0)
        self.assertIsNone(result.retrieval.get("scale"))
        self.assertNotIn("埃", result.reply)

    def test_only_one_aside_is_spoken(self):
        mind = self._mind()
        result = mind.turn("光速是多少", now=1000.0)
        aside = (result.reply or "").split("\n", 1)
        self.assertEqual(len(aside), 2, result.reply)
        self.assertIn("3×10⁸", aside[1])


if __name__ == "__main__":
    unittest.main()
