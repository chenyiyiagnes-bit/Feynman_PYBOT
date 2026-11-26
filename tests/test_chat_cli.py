"""scripts/chat.py is the user-facing entry; exercise it as a subprocess."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_chat(stdin: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "chat.py"), *args],
        input=stdin, text=True, capture_output=True, timeout=120,
    )


class TestChatCLI(unittest.TestCase):
    def test_help_and_state_commands(self):
        proc = run_chat("/help\n/state\n/quit\n", "--no-state")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("/help", proc.stdout)
        self.assertIn("dopamine=", proc.stdout)

    def test_turn_answers_and_names_speaker(self):
        proc = run_chat("原子是什么？\n/quit\n", "--no-state")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("feynman>", proc.stdout)
        self.assertIn("原子", proc.stdout)

    def test_list(self):
        proc = run_chat("", "--list")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("feynman-zh", proc.stdout)

    def test_evidence_bridge_runs_on_a_fresh_checkout(self):
        proc = run_chat("为什么天是蓝的？\n/quit\n", "--no-state", "--bridge", "--debug")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("feynman-zh:opt-012", proc.stdout)
        self.assertNotIn("证据排版不可用", proc.stdout)
        self.assertIn("\n蓝光的频率", proc.stdout)

    def test_unknown_persona_exits_nonzero(self):
        proc = run_chat("", "--persona", "nobody", "--no-state")
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
