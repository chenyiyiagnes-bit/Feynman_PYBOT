"""Promotion is a decision with a measurement, and it survives a rebuild.

A compound is only accepted when adding it to the segmenter measurably helps
the hand-segmented reference set; a candidate that fails is written down as
discarded, with the numbers, and left alone until someone asks to retry.
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

import build_lexicon as bl  # noqa: E402
import promote_terms as pt  # noqa: E402
from omega_mind import lang  # noqa: E402

CARDS = [
    {"id": "a", "type": "card", "title": "耗散",
     "text": "电流电阻一起决定导线耗散了多少热。"},
    {"id": "b", "type": "card", "title": "测量",
     "text": "测量时电流电阻都要读，温度也得记下来。"},
    {"id": "c", "type": "card", "title": "材料",
     "text": "这种材料电流电阻很小，冷却后为零。"},
    {"id": "d", "type": "card", "title": "彩虹",
     "text": "阳光在水滴里折射，颜色分开成一条彩虹。"},
    {"id": "e", "type": "card", "title": "颜色",
     "text": "不同的颜色折射角度不同，彩虹就排开了。"},
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


class Candidates(unittest.TestCase):
    def setUp(self) -> None:
        self.core = lang.load("zh")
        self.cards = CARDS

    def test_a_recurring_pair_is_a_candidate(self) -> None:
        found = {row["term"] for row in pt.candidates(self.core, self.cards, set())}
        self.assertIn("电流电阻", found)

    def test_a_pair_seen_once_is_not(self) -> None:
        found = {row["term"] for row in pt.candidates(self.core, self.cards, set())}
        self.assertNotIn("水滴折射", found)

    def test_a_known_term_is_not_a_candidate(self) -> None:
        found = {row["term"] for row in pt.candidates(self.core, self.cards,
                                                      {"超导磁体"})}
        self.assertNotIn("超导磁体", found)

    def test_candidates_are_chinese_compounds(self) -> None:
        for row in pt.candidates(self.core, self.cards, set()):
            self.assertLessEqual(len(row["term"]), pt.MAX_LEN)
            self.assertTrue(all("一" <= char <= "鿿" for char in row["term"]))

    def test_no_change_scores_zero(self) -> None:
        measured = pt.evaluate(self.core, "超导磁体")
        self.assertEqual(measured["gain"], 0.0)


class Decisions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="promote-"))
        self.package = make_package(self.tmp)
        bl.build(self.package)

    def test_dry_run_writes_nothing(self) -> None:
        report = pt.run(self.package, apply=False)
        self.assertFalse(report["written"])
        self.assertFalse((self.package / "data" / "lexicon" / pt.PROMOTIONS).exists())

    def test_a_discard_is_recorded_and_not_retried(self) -> None:
        first = pt.run(self.package, apply=True)
        log = pt.load_log(self.package / "data" / "lexicon" / pt.PROMOTIONS)
        self.assertTrue(log)
        self.assertTrue(all(row["status"] in {"promoted", "discarded"} for row in log.values()))
        discarded = [term for term, row in log.items() if row["status"] == "discarded"]
        if discarded:
            again = pt.run(self.package, apply=True)
            self.assertNotIn(discarded[0], {row["term"] for row in again["examined"]})
            retried = pt.run(self.package, apply=True, retry=True)
            self.assertIn(discarded[0], {row["term"] for row in retried["examined"]})
        self.assertTrue(first["examined"])

    def test_a_promoted_term_survives_a_rebuild(self) -> None:
        log = self.package / "data" / "lexicon" / pt.PROMOTIONS
        log.write_text(json.dumps({
            "term": "超导磁体", "status": "promoted", "cards": 3,
            "sources": ["a", "b", "c"], "reason": "test",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest = bl.build(self.package)
        self.assertGreater(manifest["terms"], 0)
        terms = json.loads((self.package / "data" / "lexicon" / "terms.json")
                           .read_text(encoding="utf-8"))["terms"]
        row = next(row for row in terms if row["term"] == "超导磁体")
        self.assertTrue(row.get("promoted"))
        self.assertEqual(row["category"], "promoted")
        self.assertEqual(row["cards"], ["a", "b", "c"])

    def test_a_discarded_term_does_not_survive(self) -> None:
        log = self.package / "data" / "lexicon" / pt.PROMOTIONS
        log.write_text(json.dumps({
            "term": "超导磁体", "status": "discarded", "cards": 3,
            "sources": ["a"], "reason": "test",
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        bl.build(self.package)
        terms = json.loads((self.package / "data" / "lexicon" / "terms.json")
                           .read_text(encoding="utf-8"))["terms"]
        self.assertNotIn("超导磁体", {row["term"] for row in terms})


if __name__ == "__main__":
    unittest.main()
