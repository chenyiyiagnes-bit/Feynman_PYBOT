"""Recall expansion from the derived term graph, and its two hard limits.

The graph may add evidence, never dilute what the question asked, and never
open the precision gate on its own. It is off unless a package asks for it:
the graph is rebuilt data, so a checkout without it must answer exactly as
before.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "right", ROOT / "bridge"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from omega_mind.feynman_kb import FeynmanKB, RECALL_TOP  # noqa: E402
from omega_mind.golden_queries import GOLDEN, NEGATIVES  # noqa: E402

PERSONA = ROOT / "personas" / "feynman-zh"


def write_corpus(tmp: str, cards: list) -> str:
    with open(Path(tmp) / "cards.jsonl", "w", encoding="utf-8") as handle:
        for card in cards:
            handle.write(json.dumps(card, ensure_ascii=False) + "\n")
    return tmp


def write_edges(tmp: str, rows: list) -> str:
    directory = Path(tmp) / "lexicon"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "edges.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    return str(directory)


CARDS = [
    {"id": "a", "title": "库珀对", "text": "两个电子在晶格里相互吸引，形成配对。",
     "tags": ["超导"]},
    {"id": "b", "title": "彩虹", "text": "阳光在水滴里折射分成七种颜色。",
     "tags": ["光学"]},
]


class Neighbours(unittest.TestCase):
    def test_no_lexicon_means_no_neighbours(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = FeynmanKB(knowledge_dir=write_corpus(tmp, CARDS), language="zh")
            self.assertEqual(kb._recall_neighbours({"电子"}), {})
            self.assertEqual(kb._edges(), {})

    def test_neighbours_look_both_ways_and_are_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"source": "成双", "target": f"配对{i}", "npmi": 0.9,
                     "pmi": 1.0, "cards": 2, "relation": "co_occurs_in_card"}
                    for i in range(RECALL_TOP + 3)]
            lexicon = write_edges(tmp, rows)
            kb = FeynmanKB(knowledge_dir=write_corpus(tmp, CARDS),
                           language="zh", lexicon_dir=lexicon)
            forward = kb._recall_neighbours({"成双"})
            backward = kb._recall_neighbours({"配对1"})
            self.assertEqual(len(forward), RECALL_TOP)
            self.assertIn("成双", backward)
            self.assertTrue(all(0 < weight <= 0.35 for weight in forward.values()))

    def test_missing_edges_file_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kb = FeynmanKB(knowledge_dir=write_corpus(tmp, CARDS), language="zh",
                           lexicon_dir=str(Path(tmp) / "nowhere"))
            self.assertEqual(kb._recall_neighbours({"电子"}), {})


class Dilution(unittest.TestCase):
    def test_a_neighbour_does_not_lower_the_question_s_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = write_corpus(tmp, CARDS)
            plain = FeynmanKB(knowledge_dir=corpus, language="zh")
            # A neighbour no card holds cannot contribute, so the score must
            # be exactly what it was without the graph.
            rows = [{"source": "库珀", "target": "巧克力", "npmi": 1.0,
                     "pmi": 1.0, "cards": 2, "relation": "co_occurs_in_card"}]
            expanded = FeynmanKB(knowledge_dir=corpus, language="zh",
                                 lexicon_dir=write_edges(tmp, rows))
            left = plain.search("库珀对", language="zh")
            right = expanded.search("库珀对", language="zh")
            self.assertTrue(left)
            self.assertEqual([doc.id for _, doc in left], [doc.id for _, doc in right])
            self.assertAlmostEqual(left[0][0], right[0][0], places=9)

    def test_a_neighbour_can_only_add(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = write_corpus(tmp, CARDS)
            plain = FeynmanKB(knowledge_dir=corpus, language="zh")
            rows = [{"source": "库珀", "target": "配对", "npmi": 1.0,
                     "pmi": 1.0, "cards": 2, "relation": "co_occurs_in_card"}]
            expanded = FeynmanKB(knowledge_dir=corpus, language="zh",
                                 lexicon_dir=write_edges(tmp, rows))
            left = plain.search("库珀对", language="zh")
            right = expanded.search("库珀对", language="zh")
            self.assertEqual([doc.id for _, doc in left], [doc.id for _, doc in right])
            self.assertGreaterEqual(right[0][0], left[0][0])

    def test_a_neighbour_alone_cannot_open_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cards = CARDS + [
                {"id": "c", "title": "金属", "text": "配对只是这里顺带写到的一个词。",
                 "tags": ["材料"]},
            ]
            rows = [{"source": "成双", "target": "配对", "npmi": 1.0,
                     "pmi": 1.0, "cards": 2, "relation": "co_occurs_in_card"}]
            kb = FeynmanKB(knowledge_dir=write_corpus(tmp, cards), language="zh",
                           lexicon_dir=write_edges(tmp, rows))
            for score, doc in kb.search("成双", language="zh"):
                self.assertNotEqual(doc.id, "c")


class RealCorpus(unittest.TestCase):
    """The graph may only ship enabled where it does not lose ground."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (PERSONA / "knowledge").is_dir():
            raise unittest.SkipTest("feynman cards are not on this machine")
        cls.plain = FeynmanKB(knowledge_dir=str(PERSONA / "knowledge"), language="zh")
        if not cls.plain.available():
            raise unittest.SkipTest("no zh knowledge on this machine")
        cls.lexicon = PERSONA / "data" / "lexicon"
        if not (cls.lexicon / "edges.jsonl").exists():
            raise unittest.SkipTest("run tools/build_lexicon.py for this package")

    def measure(self, kb) -> tuple:
        hits = sum(1 for query, language, acceptable in GOLDEN if language == "zh"
                   and kb.answer(query, language=language)[1].split(":", 1)[1]
                   in acceptable)
        rejected = sum(1 for query, language in NEGATIVES if language == "zh"
                       and kb.answer(query, language=language)[1].endswith(":no_match"))
        return hits, rejected

    def test_golden_hits_do_not_drop(self) -> None:
        expanded = FeynmanKB(knowledge_dir=str(PERSONA / "knowledge"), language="zh",
                             lexicon_dir=str(self.lexicon))
        base_hits, base_rejected = self.measure(self.plain)
        hits, rejected = self.measure(expanded)
        self.assertGreaterEqual(hits, base_hits, "expansion lost golden queries")
        self.assertGreaterEqual(rejected, base_rejected, "expansion leaked a trap")


class PersonaSwitch(unittest.TestCase):
    def test_packages_do_not_expand_unless_they_say_so(self) -> None:
        from persona import load  # noqa: PLC0415 - bridge module
        for folder in ("feynman-zh", "feynman-en", "laozhuang-en"):
            path = ROOT / "personas" / folder / "persona.json"
            if not path.exists():
                continue
            with self.subTest(package=folder):
                self.assertFalse(load(path).recall_expansion)


if __name__ == "__main__":
    unittest.main()
