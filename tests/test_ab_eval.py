"""ab_eval.py: anonymous pairwise answers, scored blind."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import ab_eval


def write_package(root: Path, persona_id: str, language: str, rows) -> Path:
    package = root / persona_id
    knowledge = package / "knowledge"
    knowledge.mkdir(parents=True)
    (package / "persona.json").write_text(
        json.dumps({"id": persona_id, "language": language}), encoding="utf-8")
    (knowledge / "cards.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8")
    return package


ZH = [{"id": "a", "type": "physics_card", "title": "熵",
       "text": "熵是无序的度量。", "tags": ["熵"]}]


class TestAbEval(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_package(self.root, "mind-a", "zh", ZH)
        write_package(self.root, "mind-b", "zh", [
            {"id": "a", "type": "physics_card", "title": "熵",
             "text": "熵是混乱程度的度量。", "tags": ["熵"]}])
        self.questions = self.root / "questions.jsonl"
        self.questions.write_text(
            json.dumps({"id": "q1", "question": "熵是什么"},
                       ensure_ascii=False) + "\n", encoding="utf-8")
        self.pairs = self.root / "pairs.jsonl"
        self.key = self.root / "key.json"
        self.votes = self.root / "votes.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, seed="1"):
        args = ["make", "--root", str(self.root),
                "--a", "mind-a", "--b", "mind-b",
                "--questions", str(self.questions),
                "--pairs", str(self.pairs), "--key", str(self.key),
                "--seed", seed]
        self.assertEqual(ab_eval.main(args), 0)
        pair = json.loads(self.pairs.read_text(encoding="utf-8").splitlines()[0])
        key = json.loads(self.key.read_text(encoding="utf-8"))
        return pair, key["q1"]

    def test_pairs_are_anonymous_and_the_key_restores_order(self):
        pair, key = self.make()
        self.assertIn("left", pair)
        self.assertIn("right", pair)
        # The pair must not name either package; only the key does.
        text = self.pairs.read_text(encoding="utf-8")
        self.assertNotIn("mind-a", text)
        self.assertNotIn("mind-b", text)
        self.assertEqual(sorted(key), ["a", "b"])
        self.assertIn({key["a"]: "mind-a", key["b"]: "mind-b"},
                      ({"left": "mind-a", "right": "mind-b"},
                       {"left": "mind-b", "right": "mind-a"}))
        self.assertTrue(pair["left"] and pair["right"])

    def test_the_same_seed_gives_the_same_order(self):
        first, _ = self.make(seed="7")
        second, _ = self.make(seed="7")
        self.assertEqual(first["left"], second["left"])

    def test_score_counts_wins_by_package(self):
        pair, key = self.make()
        # Say "left" wins; that is package key["a"].
        self.votes.write_text(json.dumps({"id": "q1", "pick": "left"}) + "\n",
                              encoding="utf-8")
        full = json.loads(self.key.read_text(encoding="utf-8"))
        report = ab_eval.score(pairs=self._read(self.pairs), key=full,
                               votes=self._read(self.votes))
        side = next(k for k, v in key.items() if v == "left")
        winner = full["_packages"][side]
        self.assertEqual(report["wins"][winner], 1)
        self.assertEqual(sum(report["wins"].values()), 1)

    def test_a_score_run_without_votes_is_refused(self):
        self.make()
        self.assertEqual(ab_eval.main(
            ["score", "--pairs", str(self.pairs), "--key", str(self.key),
             "--votes", str(self.votes)]), 2)

    @staticmethod
    def _read(path: Path):
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
