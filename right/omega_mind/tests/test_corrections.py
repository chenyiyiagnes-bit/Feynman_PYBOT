"""A correction is only worth writing down if it can be counted and replayed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omega_mind import corrections


class Recording(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="corrections-"))

    def test_a_record_keeps_its_provenance(self) -> None:
        row = corrections.record(
            "segmentation", "量子隧穿怎么办", produced=["量子", "隧穿"],
            expected=["量子隧穿"], source="词表", note="切碎了",
            lexicon="0a1b2c3d4e5f", path=self.tmp)
        self.assertEqual(row["kind"], "segmentation")
        self.assertEqual(row["expected"], ["量子隧穿"])
        self.assertEqual(row["lexicon"], "0a1b2c3d4e5f")
        self.assertTrue(row["ts"].endswith("+00:00"))
        back = corrections.replay(self.tmp)
        self.assertEqual(back, [row])

    def test_records_append_in_order(self) -> None:
        corrections.record("answer", "q1", "a", "b", path=self.tmp)
        corrections.record("answer", "q2", "a", "b", path=self.tmp)
        self.assertEqual([row["query"] for row in corrections.replay(self.tmp)],
                         ["q1", "q2"])

    def test_scalars_are_stored_as_strings(self) -> None:
        row = corrections.record("answer", "q", "wrong", "right", path=self.tmp)
        self.assertEqual(row["produced"], "wrong")
        self.assertEqual(row["expected"], "right")

    def test_unknown_kind_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            corrections.record("vibes", "q", "a", "b", path=self.tmp)
        self.assertEqual(corrections.replay(self.tmp), [])

    def test_state_dir_follows_the_environment(self) -> None:
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {corrections.ENV: str(self.tmp)}):
            corrections.record("answer", "q", "a", "b")
            self.assertTrue((self.tmp / corrections.FILE).is_file())


class Statistics(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="corrections-"))
        corrections.record("segmentation", "库珀对怎么形成", "a", "b",
                           source="词表", lexicon="v1", path=self.tmp)
        corrections.record("answer", "库珀对怎么形成", "a", "b",
                           source="C_COOPER", lexicon="v1", path=self.tmp)
        corrections.record("answer", "彩虹是什么", "a", "b",
                           source="C_COOPER", lexicon="v2", path=self.tmp)

    def test_counts_by_kind_and_source(self) -> None:
        stats = corrections.summary(self.tmp)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["by_kind"], {"answer": 2, "segmentation": 1})
        self.assertEqual(stats["by_source"], {"C_COOPER": 2, "词表": 1})

    def test_a_repeated_question_is_surfaced(self) -> None:
        stats = corrections.summary(self.tmp)
        self.assertEqual(stats["repeat_offenders"], [["库珀对怎么形成", 2]])

    def test_records_from_another_build_are_unresolved(self) -> None:
        rows = corrections.replay(self.tmp)
        stale = corrections.unresolved(rows, lexicon="v2")
        self.assertEqual([row["source"] for row in stale], ["词表", "C_COOPER"])

    def test_empty_log_is_harmless(self) -> None:
        empty = Path(tempfile.mkdtemp(prefix="corrections-"))
        self.assertEqual(corrections.replay(empty), [])
        self.assertEqual(corrections.summary(empty)["total"], 0)


if __name__ == "__main__":
    unittest.main()
