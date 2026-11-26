"""The segmentation evaluator's scoring has to be trustworthy before its verdict is.

These tests cover the parts that do not need the knowledge base: boundary
scoring, the hand-segmented reference set, report rendering and the regression
gate. The retrieval comparison itself is what `tools/eval_segmentation.py`
runs by hand or in CI.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_segmentation as ev  # noqa: E402


class BoundaryScores(unittest.TestCase):
    def test_identical_cuts_score_one(self) -> None:
        self.assertEqual(ev.boundary_scores(["熵", "是", "什么"], ["熵", "是", "什么"]),
                         (1.0, 1.0, 1.0))

    def test_boundaries_are_character_positions(self) -> None:
        self.assertEqual(ev.boundaries(["熵", "是", "什么"]), [1, 2])
        self.assertEqual(ev.boundaries(["热力学"]), [])

    def test_bigram_cut_against_word_cut(self) -> None:
        # gold 熵|是|什么, predicted 熵是|什么 -> only the boundary at 2 is shared
        precision, recall, f1 = ev.boundary_scores(["熵", "是", "什么"], ["熵是", "什么"])
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_empty_prediction_has_zero_recall(self) -> None:
        self.assertEqual(ev.boundary_scores(["熵", "是", "什么"], []), (0.0, 0.0, 0.0))

    def test_single_word_needs_no_boundaries(self) -> None:
        self.assertEqual(ev.boundary_scores(["熵"], ["熵"]), (1.0, 1.0, 1.0))


class ReferenceSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(ev.REFERENCE.read_text(encoding="utf-8"))

    def test_file_is_self_describing(self) -> None:
        self.assertEqual(self.data["version"], 1)
        self.assertTrue(self.data["built"])
        self.assertTrue(self.data["author"])

    def test_cases_reconstruct_their_text(self) -> None:
        for case in self.data["cases"]:
            with self.subTest(text=case["text"]):
                # Segmentation drops punctuation and whitespace; the words must
                # rebuild everything else in order.
                plain = "".join(ch for ch in case["text"] if ch.isalnum())
                self.assertEqual("".join(case["words"]), plain)

    def test_cases_are_unique(self) -> None:
        texts = [case["text"] for case in self.data["cases"]]
        self.assertEqual(len(texts), len(set(texts)))

    def test_reference_beats_bigrams_on_its_own_set(self) -> None:
        bigram = ev.evaluate_reference(ev._bigram)
        words = ev.evaluate_reference(ev._words)
        self.assertGreater(words["mean_f1"], bigram["mean_f1"])


class RegressionGate(unittest.TestCase):
    def _report(self, baseline_hits: int, candidate_hits: int,
                baseline_rejected: int = 10, candidate_rejected: int = 10) -> dict:
        return {"baseline": "bigram", "schemes": {
            "bigram": {"retrieval": {"hits": baseline_hits, "rejected": baseline_rejected}},
            "words": {"retrieval": {"hits": candidate_hits, "rejected": candidate_rejected}},
        }}

    def test_equal_schemes_pass(self) -> None:
        self.assertEqual(ev.regressions(self._report(65, 65)), [])

    def test_fewer_hits_blocks(self) -> None:
        problems = ev.regressions(self._report(65, 60))
        self.assertTrue(problems)
        self.assertIn("words", problems[0])

    def test_fewer_rejected_traps_blocks(self) -> None:
        problems = ev.regressions(self._report(65, 65, 10, 9))
        self.assertTrue(any("traps" in problem for problem in problems))

    def test_unavailable_candidate_is_ignored(self) -> None:
        report = self._report(65, 65)
        report["schemes"]["jieba"] = {"unavailable": "No module named 'jieba'"}
        self.assertEqual(ev.regressions(report), [])

    def test_missing_baseline_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            ev.regressions({"baseline": "bigram", "schemes": {}})

    def test_render_marks_the_baseline(self) -> None:
        text = ev.render(ev.run({"bigram": ev._bigram}))
        self.assertIn("baseline", text)
        self.assertIn("bigram", text)


class SchemeTable(unittest.TestCase):
    def test_schemes_are_named(self) -> None:
        self.assertEqual(sorted(ev.SCHEMES), ["bigram", "jieba", "words"])

    def test_jieba_absence_is_reported_not_hidden(self) -> None:
        try:
            ev._jieba("熵是什么")
        except ImportError as error:
            self.assertIn("jieba", str(error))


if __name__ == "__main__":
    unittest.main()
