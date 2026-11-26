"""Repo hygiene: no personal paths, secrets or corpus products get committed.

These run against the git index, so they fail exactly when a bad file would be
released — not when it is merely sitting untracked on disk.
"""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

SECRET_NAMES = {".en", ".env", ".probe_write"}
SAMPLE_IGNORES = [
    "left/.en",
    "left/.env",
    ".env",
    "cpt_zh.jsonl",
    "left/data/cpt_zh.jsonl",
    "p8_corpus_raw.jsonl",
    "left/data/feynman_style_vault.json",
    "feynman_style_vault_v2.json",
    "left/data/topology.npz",
    "left/data/assoc/a.json",
    "personas/feynman/data/kb_learned/a.json",
    "left/data/solve_cache.json",
]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, text=True,
                         capture_output=True, check=True)
    return out.stdout.split()


class TestHygiene(unittest.TestCase):
    def test_no_secret_files_tracked(self):
        for name in tracked_files():
            self.assertNotIn(Path(name).name, SECRET_NAMES, name)

    def test_no_personal_absolute_paths_in_tracked_text(self):
        offenders = []
        for name in tracked_files():
            path = ROOT / name
            if path.resolve() == SELF:  # this file must spell the needle out
                continue
            if path.suffix not in {".py", ".md", ".json", ".yml", ".yaml",
                                   ".txt", ".sh", ".toml"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if "/Users/" in text:
                offenders.append(name)
        self.assertEqual(offenders, [])

    def test_red_line_paths_are_ignored(self):
        for sample in SAMPLE_IGNORES:
            proc = subprocess.run(["git", "check-ignore", "-q", sample],
                                  cwd=ROOT)
            self.assertEqual(proc.returncode, 0, f"{sample} is not ignored")


if __name__ == "__main__":
    unittest.main()
