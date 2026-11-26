"""card_coverage.py must account for every row, loaded or dropped with a reason."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import card_coverage


def write_package(root: Path, persona_id: str, language: str,
                  files: dict) -> Path:
    package = root / persona_id
    knowledge = package / "knowledge"
    knowledge.mkdir(parents=True)
    (package / "persona.json").write_text(
        json.dumps({"id": persona_id, "language": language}), encoding="utf-8")
    for name, rows in files.items():
        (knowledge / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8")
    return package


class TestCardCoverage(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_every_row_is_loaded_or_has_a_named_reason(self):
        write_package(self.root, "tiny-zh", "zh", {
            "cards.jsonl": [
                {"id": "a", "type": "physics_card", "title": "熵",
                 "text": "熵是无序的度量。"},
                # Structured card with no "text": the body is elsewhere.
                {"id": "b", "type": "teachable_unit", "title": "阻尼",
                 "force_invariant": "摆越摆越小。"},
                # No body at all: must be reported, not silently lost.
                {"id": "c", "type": "physics_card", "title": ""},
            ],
        })
        report = card_coverage.survey(root=self.root)
        package = report["packages"]["tiny-zh"]
        self.assertEqual(package["rows"], 3)
        self.assertEqual(package["loaded"], 2)
        self.assertEqual(package["unaccounted"], 0)
        reasons = [row["reason"] for row in package["dropped"]]
        self.assertEqual(reasons, ["no-answer-text"])
        self.assertEqual(package["dropped"][0]["id"], "c")

    def test_an_english_package_reports_rows_that_are_not_english(self):
        write_package(self.root, "tiny-en", "en", {
            "cards.jsonl": [
                {"id": "a", "title": "Entropy",
                 "text": "Entropy measures disorder."},
                {"id": "b", "title": "熵", "text": "熵是无序的度量。"},
            ],
        })
        package = card_coverage.survey(root=self.root)["packages"]["tiny-en"]
        self.assertEqual(package["loaded"], 1)
        self.assertEqual([row["reason"] for row in package["dropped"]],
                         ["not-english"])

    def test_a_broken_line_fails_the_run(self):
        package = write_package(self.root, "tiny-zh", "zh", {
            "cards.jsonl": [{"id": "a", "title": "熵",
                             "text": "熵是无序的度量。"}],
        })
        with (package / "knowledge" / "cards.jsonl").open("a",
                                                          encoding="utf-8") as fh:
            fh.write("{not json}\n")
        self.assertEqual(card_coverage.main(["--root", str(self.root)]), 1)

    def test_a_clean_tree_passes_and_a_ledger_round_trips(self):
        write_package(self.root, "tiny-zh", "zh", {
            "cards.jsonl": [{"id": "a", "title": "熵",
                             "text": "熵是无序的度量。"}],
        })
        ledger = self.root / "ledger.json"
        self.assertEqual(card_coverage.main(
            ["--root", str(self.root), "--write-ledger", str(ledger)]), 0)
        self.assertEqual(card_coverage.main(
            ["--root", str(self.root), "--check-ledger", str(ledger)]), 0)

    def test_a_changed_card_is_reported_against_the_ledger(self):
        package = write_package(self.root, "tiny-zh", "zh", {
            "cards.jsonl": [{"id": "a", "title": "熵",
                             "text": "熵是无序的度量。"}],
        })
        ledger = self.root / "ledger.json"
        card_coverage.main(
            ["--root", str(self.root), "--write-ledger", str(ledger)])
        cards = package / "knowledge" / "cards.jsonl"
        cards.write_text(json.dumps({"id": "a", "title": "熵",
                                     "text": "熵增加了。"}, ensure_ascii=False)
                         + "\n", encoding="utf-8")
        self.assertEqual(card_coverage.main(
            ["--root", str(self.root), "--check-ledger", str(ledger)]), 1)


if __name__ == "__main__":
    unittest.main()
