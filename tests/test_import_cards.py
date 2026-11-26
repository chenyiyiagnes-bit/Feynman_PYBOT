"""import_cards.py: user notes in, retrievable cards out."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import import_cards


class TestParsing(unittest.TestCase):
    def test_markdown_headings_become_cards(self):
        text = "# 水\n\n标准大气压下 100 度沸腾。\n\n## 冰\n\n0 度结冰。\n"
        cards = import_cards.cards_from(text, "notes")
        self.assertEqual([c["title"] for c in cards], ["水", "冰"])
        self.assertEqual(cards[0]["text"], "标准大气压下 100 度沸腾。")

    def test_plain_paragraphs_split_on_blank_lines(self):
        text = "水的沸点\n标准大气压下 100 度。\n\n冰点是多少？\n0 度结冰。\n"
        cards = import_cards.cards_from(text, "n")
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[1]["title"], "冰点是多少？")

    def test_ids_are_stable(self):
        text = "# 水\n\n100 度沸腾。\n"
        a = import_cards.cards_from(text, "x")
        b = import_cards.cards_from(text, "x")
        self.assertEqual(a[0]["id"], b[0]["id"])

    def test_extra_tags_and_latin_title_tokens(self):
        cards = import_cards.cards_from("# QED basics\n\nstuff\n", "p", ["物理"])
        self.assertEqual(cards[0]["tags"][0], "物理")
        self.assertIn("qed", cards[0]["tags"])


class TestCLI(unittest.TestCase):
    def test_writes_jsonl_the_kb_can_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "notes.md"
            src.write_text("# 水的沸点\n\n标准大气压下，水在 100 摄氏度沸腾。\n",
                           encoding="utf-8")
            out = Path(tmp) / "cards.jsonl"
            self.assertEqual(import_cards.main([str(src), "-o", str(out)]), 0)
            card = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(card["title"], "水的沸点")

            import sys
            root = Path(__file__).resolve().parents[1]
            sys.path.insert(0, str(root / "right"))
            from omega_mind.feynman_kb import FeynmanKB
            kb = FeynmanKB(knowledge_dir=tmp)
            self.assertIn("100", kb.answer("水的沸点是什么？")[0])

    def test_missing_file(self):
        self.assertEqual(import_cards.main(["/no/such/file.md"]), 2)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "empty.md"
            src.write_text("", encoding="utf-8")
            self.assertEqual(import_cards.main([str(src), "--print"]), 1)


if __name__ == "__main__":
    unittest.main()
