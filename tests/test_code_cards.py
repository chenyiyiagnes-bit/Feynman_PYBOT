"""Every coded token can be pointed back at the bytes it came from.

The point of the coding is not to compress the cards — it is to make a token
checkable. If ``text.encode()[start:end]`` does not give the token back, slot
filling would be verifying against a coordinate system that lies.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools", ROOT / "right"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import build_lexicon as bl  # noqa: E402
import code_cards as cc  # noqa: E402

CARDS = [
    {"id": "a", "type": "card", "title": "热传导",
     "text": "热量从热的一端流到冷的一端。", "source_note": "编辑备注不该出现在词表里"},
    {"id": "b", "type": "card", "title": "热扩散",
     "text": "热量总是从高温流向低温，扩散是分子运动的结果。"},
    {"id": "c", "type": "teachable_unit", "title": "能量守恒",
     "phenomenon": "能量在热和功之间转换时总量不变。"},
]


def make_package(tmp: Path, cards=None) -> Path:
    package = tmp / "pkg"
    (package / "knowledge").mkdir(parents=True)
    (package / "persona.json").write_text(
        json.dumps({"schema": 1, "id": "pkg", "name": "t", "language": "zh"}),
        encoding="utf-8")
    with (package / "knowledge" / "cards.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards if cards is not None else CARDS:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    return package


class Traceability(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="coded-"))
        cls.package = make_package(cls.tmp)
        bl.build(cls.package, built="2026-11-26")
        cc.build(cls.package, built="2026-11-26")
        cls.dir = cls.package / "data" / "lexicon"
        cls.rows = [json.loads(line) for line in
                    (cls.dir / "cards.coded.jsonl").read_text(encoding="utf-8").splitlines()]
        cls.terms = json.loads((cls.dir / "terms.json").read_text(encoding="utf-8"))["terms"]

    def text_of(self, card_id: str, field: str, occurrence: int) -> str:
        cards = {card["id"]: card for card in bl.load_cards(self.package / "knowledge")}
        return list(bl.card_strings(cards[card_id][field], field))[occurrence]

    def test_every_token_maps_to_its_bytes(self) -> None:
        seen = 0
        for row in self.rows:
            counts: dict = {}
            for field in row["fields"]:
                name = field["field"]
                occurrence = counts.get(name, 0)
                counts[name] = occurrence + 1
                raw = self.text_of(row["id"], name, occurrence).encode("utf-8")
                for token in field["tokens"]:
                    with self.subTest(card=row["id"], token=token["t"]):
                        self.assertEqual(raw[token["start"]:token["end"]].decode("utf-8"),
                                         token["t"])
                        seen += 1
        self.assertGreater(seen, 10)

    def test_ids_resolve_to_the_same_word(self) -> None:
        for row in self.rows:
            for field in row["fields"]:
                for token in field["tokens"]:
                    if token["id"] == cc.OOV:
                        self.assertNotIn(token["t"], {r["term"] for r in self.terms})
                    else:
                        with self.subTest(token=token["t"]):
                            self.assertEqual(self.terms[token["id"]]["term"], token["t"])

    def test_provenance_fields_are_not_coded(self) -> None:
        for row in self.rows:
            self.assertNotIn("source_note", {field["field"] for field in row["fields"]})

    def test_coverage_is_reported_and_honest(self) -> None:
        manifest = json.loads((self.dir / "cards.coded.manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["cards"], len(self.rows))
        self.assertEqual(manifest["tokens"], sum(len(f["tokens"]) for r in self.rows
                                                 for f in r["fields"]))
        self.assertGreaterEqual(manifest["coverage"], 0.0)
        self.assertLessEqual(manifest["coverage"], 1.0)
        self.assertEqual(manifest["language"], "zh")
        self.assertIn("never committed", manifest["note"])

    def test_rebuild_is_byte_identical(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="coded-"))
        package = make_package(tmp)
        bl.build(package, built="2026-11-26")
        cc.build(package, built="2026-11-26")
        first = (package / "data" / "lexicon" / "cards.coded.jsonl").read_bytes()
        cc.build(package, built="2026-11-26")
        self.assertEqual(first,
                         (package / "data" / "lexicon" / "cards.coded.jsonl").read_bytes())


class Preconditions(unittest.TestCase):
    def test_missing_lexicon_is_named(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="coded-"))
        package = make_package(tmp)
        with self.assertRaises(SystemExit) as caught:
            cc.build(package)
        self.assertIn("build_lexicon", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
