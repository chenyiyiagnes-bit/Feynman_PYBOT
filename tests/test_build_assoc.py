"""Mined association edges are candidates, and they say so.

A fresh clone has cards and no edges. The mine may only produce what the cards
actually share — ``origin="cooccur"``, ``kind="same_domain"``, no mechanism —
and it must be reproducible, so the same cards give the same file.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "tools", ROOT / "right"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import build_assoc as ba  # noqa: E402
from omega_mind import assoc, lang  # noqa: E402

CARDS = [
    {"id": "a", "type": "card", "title": "库珀对",
     "text": "电子带着声子在晶格里走动，两个电子配对。"},
    {"id": "b", "type": "card", "title": "超导",
     "text": "电子配对之后电阻消失，声子还在晶格里振动。"},
    {"id": "c", "type": "card", "title": "声子",
     "text": "声子是晶格振动，电子和声子交换能量。"},
    {"id": "d", "type": "card", "title": "彩虹",
     "text": "阳光在水滴里折射，颜色分开成一条彩虹。"},
]


def make_package(tmp: Path, cards=None) -> Path:
    package = tmp / "pkg"
    (package / "knowledge").mkdir(parents=True, exist_ok=True)
    (package / "persona.json").write_text(
        json.dumps({"schema": 1, "id": "pkg", "name": "t", "language": "zh"}),
        encoding="utf-8")
    with (package / "knowledge" / "cards.jsonl").open("w", encoding="utf-8") as handle:
        for card in cards if cards is not None else CARDS:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    return package


class Mining(unittest.TestCase):
    def setUp(self) -> None:
        self.core = lang.load("zh")

    def test_cards_that_share_terms_are_linked(self) -> None:
        rows = ba.mine(self.core, CARDS)
        pairs = {(row["a"], row["b"]) for row in rows}
        self.assertTrue(pairs, "the three electron cards share more than three terms")
        self.assertFalse(any("d" in pair for pair in pairs), "the rainbow card shares nothing")

    def test_mined_edges_are_candidates_without_prose(self) -> None:
        for row in ba.mine(self.core, CARDS):
            self.assertEqual(row["origin"], "cooccur")
            self.assertEqual(row["kind"], "same_domain")
            self.assertNotIn("mechanism", row)
            self.assertNotIn("difference", row)
            self.assertTrue(row["terms"])

    def test_the_mine_is_reproducible(self) -> None:
        first = ba.mine(self.core, CARDS)
        second = ba.mine(lang.load("zh"), CARDS)
        self.assertEqual(first, second)

    def test_fan_out_is_capped(self) -> None:
        many = [{"id": f"c{index}", "type": "card", "title": f"卡{index}",
                 "text": "电子声子晶格配对电阻温度热，这些词反复出现。"}
                for index in range(12)]
        rows = ba.mine(self.core, many)
        degree: dict = {}
        for row in rows:
            degree[row["a"]] = degree.get(row["a"], 0) + 1
            degree[row["b"]] = degree.get(row["b"], 0) + 1
        self.assertTrue(all(count <= ba.MAX_PER_CARD for count in degree.values()))


class Output(unittest.TestCase):
    def test_written_edges_load_without_drops(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="assoc-"))
        package = make_package(tmp)
        manifest = ba.build(package, built="2026-11-26")
        path = package / "data" / "assoc" / ba.CANDIDATES
        self.assertTrue(path.is_file())
        graph = assoc.load(path)
        self.assertEqual(graph.dropped, 0)
        self.assertEqual(len(graph.edges), manifest["edges"])
        self.assertTrue(all(edge.origin == "cooccur" for edge in graph.edges))
        self.assertTrue(all(not edge.spoken if hasattr(edge, "spoken")
                            else edge.kind not in assoc.SPOKEN
                            for edge in graph.edges))

    def test_rebuild_is_byte_identical(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="assoc-"))
        package = make_package(tmp)
        ba.build(package, built="2026-11-26")
        path = package / "data" / "assoc" / ba.CANDIDATES
        first = path.read_bytes()
        ba.build(package, built="2026-11-26")
        self.assertEqual(first, path.read_bytes())

    def test_confirmed_edges_are_never_overwritten(self) -> None:
        """The mine writes candidates; a hand-confirmed file is left alone."""
        tmp = Path(tempfile.mkdtemp(prefix="assoc-"))
        package = make_package(tmp)
        out = package / "data" / "assoc"
        out.mkdir(parents=True, exist_ok=True)
        confirmed = out / "assoc_edges.jsonl"
        confirmed.write_text(json.dumps({
            "a": "a", "b": "b", "kind": "same_mechanism", "origin": "human",
            "mechanism": "手写", "difference": "手写"}) + "\n", encoding="utf-8")
        before = confirmed.read_bytes()
        ba.build(package, built="2026-11-26")
        self.assertEqual(confirmed.read_bytes(), before)
        # Even the candidates file refuses to clobber a confirmed row.
        (out / ba.CANDIDATES).write_bytes(before)
        with self.assertRaises(SystemExit):
            ba.build(package, built="2026-11-26")

    def test_empty_package_is_refused(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="assoc-"))
        package = make_package(tmp)
        (package / "knowledge" / "cards.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(SystemExit):
            ba.build(package)


if __name__ == "__main__":
    unittest.main()
