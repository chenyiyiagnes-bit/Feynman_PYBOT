"""doctor.py is the first thing a new user runs; keep it working."""

from __future__ import annotations

import unittest

from tools import doctor


class TestDoctor(unittest.TestCase):
    def test_reports_no_failures(self):
        self.assertEqual(doctor.main([]), 0)

    def test_unknown_persona_fails_loudly(self):
        self.assertEqual(doctor.main(["--persona", "no-such-person"]), 1)


if __name__ == "__main__":
    unittest.main()
