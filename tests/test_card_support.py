"""card_support.py checks the optional ``support`` provenance block."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import card_support


def write_package(root: Path, persona_id: str, rows) -> Path:
    package = root / persona_id
    knowledge = package / "knowledge"
    knowledge.mkdir(parents=True)
    (package / "persona.json").write_text(
        json.dumps({"id": persona_id, "language": "zh"}), encoding="utf-8")
    (knowledge / "cards.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    return package


def card(card_id: str, support=None) -> dict:
    row = {"id": card_id, "type": "physics_card", "title": "熵",
           "text": "熵是无序的度量。"}
    if support is not None:
        row["support"] = support
    return row


class TestCardSupport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_matching_quote_in_a_present_source_is_valid(self):
        source = self.root / "reference" / "lecture.txt"
        source.parent.mkdir()
        source.write_text("先看它是什么样、怎么动。\n再看条件改掉会怎样。\n",
                          encoding="utf-8")
        quote = "先看它是什么样、怎么动。"
        write_package(self.root, "tiny-zh", [card(
            "a", [{"source": "FLP I-1-1", "quote": quote,
                   "source_file": "reference/lecture.txt",
                   "quote_sha256": hashlib.sha256(
                       quote.encode("utf-8")).hexdigest()}]),
        ])
        report = card_support.check(root=self.root)
        package = report["packages"]["tiny-zh"]
        self.assertEqual(package["with_support"], 1)
        self.assertEqual(package["valid"], 1)
        self.assertEqual(package["invalid"], [])
        self.assertEqual(card_support.main(["--root", str(self.root)]), 0)

    def test_a_hash_mismatch_is_invalid_and_fails(self):
        quote = "先看它是什么样、怎么动。"
        write_package(self.root, "tiny-zh", [card(
            "a", [{"source": "FLP I-1-1", "quote": quote,
                   "quote_sha256": "0" * 64}]),
        ])
        report = card_support.check(root=self.root)
        self.assertEqual(len(report["packages"]["tiny-zh"]["invalid"]), 1)
        self.assertEqual(card_support.main(["--root", str(self.root)]), 1)

    def test_a_quote_absent_from_the_source_is_invalid(self):
        source = self.root / "reference" / "lecture.txt"
        source.parent.mkdir()
        source.write_text("别的句子。\n", encoding="utf-8")
        write_package(self.root, "tiny-zh", [card(
            "a", [{"source": "FLP I-1-1", "quote": "先看它是什么样",
                   "source_file": "reference/lecture.txt"}]),
        ])
        report = card_support.check(root=self.root)
        invalid = report["packages"]["tiny-zh"]["invalid"]
        self.assertEqual(len(invalid), 1)
        self.assertIn("not found", invalid[0]["problem"])

    def test_a_missing_source_file_is_unverified_not_valid(self):
        write_package(self.root, "tiny-zh", [card(
            "a", [{"source": "FLP I-1-1", "quote": "先看它是什么样",
                   "source_file": "reference/absent.txt"}]),
        ])
        package = card_support.check(root=self.root)["packages"]["tiny-zh"]
        self.assertEqual(package["valid"], 0)
        self.assertEqual(package["unverified"], 1)
        self.assertEqual(package["invalid"], [])

    def test_cards_without_support_are_counted_and_listed(self):
        write_package(self.root, "tiny-zh", [card("a"), card("b")])
        report = card_support.check(root=self.root)
        self.assertEqual(report["packages"]["tiny-zh"]["missing"], 2)
        self.assertEqual(card_support.main(["--root", str(self.root)]), 0)
        self.assertEqual(card_support.main(
            ["--root", str(self.root), "--strict"]), 1)

    def test_a_malformed_support_block_is_invalid(self):
        write_package(self.root, "tiny-zh", [card("a", "not a list")])
        report = card_support.check(root=self.root)
        self.assertEqual(len(report["packages"]["tiny-zh"]["invalid"]), 1)


if __name__ == "__main__":
    unittest.main()
