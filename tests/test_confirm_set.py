"""A confirmation set that was never trained on, and never tuned against.

The old held-out sets are worthless as evidence: they went into training. This
one was written by reading what each card says and asking again in different
words. The acceptable answers were chosen from the card text *before* the
retrieval was run, so a miss is a miss and stays in the file.

The floor below only catches collapse. It is not a quality bar: a low score
here is information for the next retrieval change, not a reason to loosen the
set.
"""

from __future__ import annotations

import datetime
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind.feynman_kb import FeynmanKB  # noqa: E402
from omega_mind.golden_queries import GOLDEN, NEGATIVES  # noqa: E402

DATA = ROOT / "tests" / "data" / "confirm_set_v1.json"
PERSONAS = ROOT / "personas"

# Measured when the set was built (15/30). The floor is deliberately below it:
# this test exists to notice a collapse, not to police paraphrase quality.
FLOOR = 0.40


def _cases() -> list:
    return json.loads(DATA.read_text(encoding="utf-8"))["cases"]


def _card_text_rows() -> list:
    """Title-like strings the set must not reuse verbatim."""
    phrases = []
    for lang in ("feynman-zh", "feynman-en"):
        for path in (PERSONAS / lang / "knowledge").glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ("title", "entry_question"):
                    value = row.get(key)
                    if isinstance(value, str):
                        phrases.append(value.strip())
                aliases = row.get("aliases")
                if isinstance(aliases, list):
                    phrases.extend(a.strip() for a in aliases if isinstance(a, str))
    return phrases


class Shape(unittest.TestCase):
    def test_provenance_is_registered(self) -> None:
        data = json.loads(DATA.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertTrue(data["author"])
        self.assertTrue(data["provenance"])
        built = datetime.date.fromisoformat(data["built"])
        self.assertGreaterEqual(built, datetime.date(2026, 11, 26))

    def test_every_case_is_complete(self) -> None:
        for case in _cases():
            with self.subTest(query=case["query"]):
                self.assertIn(case["language"], ("zh", "en"))
                self.assertTrue(case["query"].strip())
                self.assertTrue(case["acceptable"])
                self.assertTrue(all(isinstance(card, str) for card in case["acceptable"]))

    def test_queries_are_unique(self) -> None:
        queries = [case["query"] for case in _cases()]
        self.assertEqual(len(queries), len(set(queries)))


class Cleanliness(unittest.TestCase):
    def test_not_copied_from_the_golden_set(self) -> None:
        known = {query for query, _, _ in GOLDEN} | {query for query, _ in NEGATIVES}
        for case in _cases():
            with self.subTest(query=case["query"]):
                self.assertNotIn(case["query"], known)

    def test_not_a_card_title_or_alias(self) -> None:
        phrases = set(_card_text_rows())
        for case in _cases():
            with self.subTest(query=case["query"]):
                self.assertNotIn(case["query"].strip(), phrases)


class Retrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = {
            "zh": FeynmanKB(knowledge_dir=str(PERSONAS / "feynman-zh" / "knowledge"),
                            language="zh"),
            "en": FeynmanKB(knowledge_dir=str(PERSONAS / "feynman-en" / "knowledge"),
                            language="en"),
        }
        for language, kb in cls.kb.items():
            if not kb.available():
                raise unittest.SkipTest(f"no {language} knowledge on this machine")

    def test_hit_rate_has_not_collapsed(self) -> None:
        hits, misses = 0, []
        for case in _cases():
            kb = self.kb[case["language"]]
            _, source = kb.answer(case["query"], language=case["language"])
            card = source.split(":", 1)[1] if ":" in source else source
            if card in case["acceptable"]:
                hits += 1
            else:
                misses.append((case["query"], card, case["acceptable"]))
        rate = hits / len(_cases())
        self.assertGreaterEqual(rate, FLOOR, f"hit rate {hits}/{len(_cases())}; "
                               f"misses: {misses}")


if __name__ == "__main__":
    unittest.main()
