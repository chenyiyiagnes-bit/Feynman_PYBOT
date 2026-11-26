"""Portability: a fresh checkout must answer without this machine's paths."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "right"))

from omega_mind.feynman_kb import DEFAULT_DIRS, FeynmanKB  # noqa: E402


class TestDefaults(unittest.TestCase):
    def test_default_dir_is_inside_this_repo(self):
        for path in DEFAULT_DIRS:
            self.assertTrue(path.is_relative_to(ROOT), path)
            self.assertTrue(path.is_dir(), path)
            self.assertTrue(any(path.glob("*.jsonl")), path)

    def test_learned_overlay_works_without_a_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = FeynmanKB(autoload=False, learned_dir=tmp)
            self.assertIsNotNone(kb.learned)
            kb.learned.add_card({"id": "learned-1", "title": "蜡封信件",
                                 "text": "一串首尾相扣的蜡封信件。", "tags": []})
            files = list(Path(tmp).iterdir())
            self.assertTrue(files, "nothing written to the learned dir")


if __name__ == "__main__":
    unittest.main()
