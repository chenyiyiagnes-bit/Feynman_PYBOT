"""The derived lexicon is reproducible, traceable, and stays out of git.

A lexicon nobody can rebuild is a rumour. These tests build one from a small
synthetic package and, when the real cards are on this machine, from
``personas/feynman-zh`` — twice — and check the two runs agree byte for byte.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_lexicon.py"
PERSONA = ROOT / "personas" / "feynman-zh"

if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import build_lexicon as bl  # noqa: E402
from omega_mind import lang  # noqa: E402


def write_persona(directory: Path, cards: list, language: str = "zh") -> Path:
    (directory / "knowledge").mkdir(parents=True)
    (directory / "persona.json").write_text(
        json.dumps({"schema": 1, "id": directory.name, "name": "t",
                    "language": language}), encoding="utf-8")
    with (directory / "knowledge" / "cards.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    return directory


class Synthetic(unittest.TestCase):
    """Small packages, so each rule is checked on its own terms."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="lexicon-"))

    def build(self, cards: list, name: str = "pkg") -> dict:
        persona = write_persona(self.tmp / name, cards)
        return bl.build(persona, built="2026-11-26")

    def test_structured_card_without_text_is_mined(self) -> None:
        # HANDOFF §6: teachable units carry the body in phenomenon, not text.
        manifest = self.build([
            {"id": "a", "type": "teachable_unit", "title": "热传导",
             "phenomenon": "热量从热的一端流到冷的一端"},
            {"id": "b", "type": "teachable_unit", "title": "热扩散",
             "force_invariant": "热量总是从高温流向低温"},
        ])
        self.assertEqual(manifest["cards"], 2)
        out = self.tmp / "pkg" / "data" / "lexicon"
        terms = json.loads((out / "terms.json").read_text(encoding="utf-8"))["terms"]
        names = {row["term"] for row in terms}
        self.assertIn("热量", names)

    def test_function_words_are_not_terms(self) -> None:
        self.build([
            {"id": "a", "type": "card", "text": "为什么温度会上升，为什么时间有方向"},
            {"id": "b", "type": "card", "text": "为什么温度总是上升，时间的方向是什么"},
        ])
        out = self.tmp / "pkg" / "data" / "lexicon"
        names = {row["term"] for row in
                 json.loads((out / "terms.json").read_text(encoding="utf-8"))["terms"]}
        self.assertIn("温度", names)
        for word in ("为什么", "什么", "不是", "还是"):
            self.assertNotIn(word, names)

    def test_provenance_points_at_real_cards(self) -> None:
        self.build([
            {"id": "a", "type": "card", "text": "超导体的电子两两配对"},
            {"id": "b", "type": "card", "text": "超导体里电子形成库珀对"},
        ])
        out = self.tmp / "pkg" / "data" / "lexicon"
        rows = json.loads((out / "terms.json").read_text(encoding="utf-8"))["terms"]
        for row in rows:
            with self.subTest(term=row["term"]):
                self.assertTrue(row["cards"])
                self.assertLessEqual(len(row["cards"]), bl.MAX_CARDS_PER_TERM)
                self.assertGreaterEqual(row["card_count"], len(row["cards"]))
                self.assertIn(row["category"], ("card", "teachable_unit"))

    def test_edges_reference_known_terms(self) -> None:
        self.build([
            {"id": "a", "type": "card", "text": "光子散射导致天空呈蓝色"},
            {"id": "b", "type": "card", "text": "散射让天空变蓝，光子偏折"},
            {"id": "c", "type": "card", "text": "温度是分子的平均动能"},
        ])
        out = self.tmp / "pkg" / "data" / "lexicon"
        names = {row["term"] for row in
                 json.loads((out / "terms.json").read_text(encoding="utf-8"))["terms"]}
        edges = [json.loads(line) for line in
                 (out / "edges.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(edges)
        for edge in edges:
            with self.subTest(edge=edge):
                self.assertIn(edge["source"], names)
                self.assertIn(edge["target"], names)
                self.assertEqual(edge["relation"], bl.RELATION)
                self.assertGreater(edge["pmi"], 0)
                self.assertLessEqual(edge["npmi"], 1.0)

    def test_empty_package_is_refused(self) -> None:
        persona = write_persona(self.tmp / "empty", [])
        with self.assertRaises(SystemExit):
            bl.build(persona)


class Reproducible(unittest.TestCase):
    def test_two_builds_are_byte_identical(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="lexicon-"))
        one, two = tmp / "one", tmp / "two"
        bl.build(PERSONA, out_dir=one, built="2026-11-26")
        bl.build(PERSONA, out_dir=two, built="2026-11-26")
        for name in ("terms.json", "edges.jsonl", "manifest.json"):
            with self.subTest(file=name):
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())

    def test_manifest_names_its_input(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="lexicon-"))
        manifest = bl.build(PERSONA, out_dir=tmp, built="2026-11-26")
        self.assertEqual(manifest["cards"], len(bl.load_cards(PERSONA / "knowledge")))
        self.assertEqual(len(manifest["source_digest"]), 16)
        self.assertEqual(manifest["language"], "zh")
        self.assertIn("never committed", manifest["note"])

    def test_default_output_is_derived_data(self) -> None:
        # The tool's default target must be the ignored directory, not the
        # cards: a rebuild can never overwrite source of truth.
        source = TOOL.read_text(encoding="utf-8")
        self.assertIn('"data" / "lexicon"', source)
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("**/data/lexicon/", ignore)


class Cli(unittest.TestCase):
    def test_runs_from_the_command_line(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="lexicon-"))
        result = subprocess.run(
            [sys.executable, str(TOOL), str(PERSONA), "--out", str(tmp),
             "--date", "2026-11-26"],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("terms", result.stdout)
        self.assertTrue((tmp / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
