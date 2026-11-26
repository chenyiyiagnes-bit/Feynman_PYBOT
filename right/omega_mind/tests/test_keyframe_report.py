"""The keyframe report must stay a faithful read of the real engine."""

from __future__ import annotations

import unittest

from omega_mind.keyframe_report import (
    DEFAULT_LINES,
    Keyframe,
    pick_peak,
    render_side_by_side,
    run_report,
)


class TestSideBySide(unittest.TestCase):
    def test_both_panes_get_the_same_number_of_rows(self):
        left = "a\nbb\nccc"
        right = "1"
        text = render_side_by_side("L", left, "R", right)
        rows = text.splitlines()[2:]  # skip titles and rule
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[0].startswith("a"))
        self.assertIn("1", rows[0])

    def test_empty_side_still_renders(self):
        text = render_side_by_side("L", "", "R", "x\ny")
        self.assertEqual(len(text.splitlines()), 4)


class TestPeakSelection(unittest.TestCase):
    def test_highest_verbosity_wins(self):
        frames = [Keyframe(i, "", v, "", 0, {}, "")
                  for i, v in enumerate((0.3, 0.9, 0.5), start=1)]
        self.assertEqual(pick_peak(frames).turn, 2)

    def test_first_of_equal_peaks_wins(self):
        frames = [Keyframe(i, "", 0.7, "", 0, {}, "") for i in (1, 2, 3)]
        self.assertEqual(pick_peak(frames).turn, 1)


class TestReportRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_report(persona="feynman", turns=40, dream_at=32)

    def test_runs_every_turn(self):
        self.assertEqual(len(self.report.frames), 40)
        self.assertEqual([f.turn for f in self.report.frames], list(range(1, 41)))

    def test_peak_matches_the_recorded_maximum(self):
        best = max(f.verbosity for f in self.report.frames)
        self.assertEqual(self.report.peak.verbosity, best)

    def test_dream_does_not_increase_displacement(self):
        dream = self.report.dream
        self.assertIsNotNone(dream)
        self.assertLessEqual(dream.after_displacement,
                             dream.before_displacement)

    def test_dream_nets_are_rendered(self):
        dream = self.report.dream
        self.assertIn("moves:", dream.before_net)
        self.assertIn("moves:", dream.after_net)

    def test_dream_snapshot_records_salience_and_ratio(self):
        dream = self.report.dream
        self.assertIsNotNone(dream)
        self.assertGreaterEqual(dream.salience, 0.0)
        self.assertIsNotNone(dream.restore_ratio)
        self.assertGreater(dream.restore_ratio, 0.0)
        self.assertLessEqual(dream.restore_ratio, 0.45)
        self.assertIn("salience=", self.report.to_text())

    def test_text_report_names_both_keyframes(self):
        text = self.report.to_text()
        self.assertIn("verbosity peak at turn", text)
        self.assertIn("before dream", text)
        self.assertIn("after dream", text)

    def test_script_is_reused(self):
        self.assertEqual(self.report.frames[0].text, DEFAULT_LINES[0])
        self.assertEqual(self.report.frames[10].text, DEFAULT_LINES[0])


if __name__ == "__main__":
    unittest.main()
