"""The diagnostic scripts must be runnable, not just importable.

They used to treat ``--help`` as a snapshot path and die with a traceback,
and they hardcoded the Chinese Feynman package for every run. Both are
command-line contracts, so they are pinned here through the real process.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

RIGHT = Path(__file__).resolve().parents[2]
SCRIPTS = RIGHT / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / f"{name}.py"), *args],
        cwd=str(RIGHT), capture_output=True, text=True, timeout=120)


class TestHelp(unittest.TestCase):
    def test_every_script_has_usage(self):
        for name in ("evaluate_ab", "evaluate_learning", "evolve_curriculum"):
            with self.subTest(script=name):
                done = run_script(name, "--help")
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn("usage", done.stdout.lower())


class TestFlagParsing(unittest.TestCase):
    def test_a_flag_without_a_value_is_a_clean_error(self):
        for name, flag in (("evaluate_ab", "--persona"),
                           ("evolve_curriculum", "--learned-dir")):
            with self.subTest(script=name):
                done = run_script(name, flag)
                self.assertNotEqual(done.returncode, 0)
                self.assertNotIn("Traceback", done.stderr)

    def test_persona_is_accepted_by_the_ab_script(self):
        done = run_script("evaluate_ab", "--persona", "no-such-person")
        # The persona is resolved, not ignored: a bogus one fails loudly.
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no-such-person", (done.stderr or "") + (done.stdout or ""))


if __name__ == "__main__":
    unittest.main()
