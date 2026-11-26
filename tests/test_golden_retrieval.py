"""Golden retrieval: stable questions must keep hitting the right card.

These are regression anchors, not a quality benchmark. If a card edit or a
scorer change makes one of these miss, that is the signal to look — do not
"fix" the test by loosening it without checking the card first.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.runtime import build

PERSONAS = Path(__file__).resolve().parents[1] / "personas"

GOLDEN = [
    ("原子是什么？", "feynman_kb:flp-i-01", "原子"),
    ("我是谁？", "feynman_kb:self-001", "费曼"),
    ("量子电动力学是什么？", "feynman_kb:qed-001", "QED"),
    ("为什么天是蓝的？", "feynman_kb:opt-012", "散射"),
    ("能量守恒是什么？", "feynman_kb:flp-i-04", "能量"),
    ("你对科学是什么态度？", "feynman_kb:persona-science-001", "科学"),
]


class TestGoldenRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build("feynman", personas_root=PERSONAS, enable_bridge=False)

    def test_golden_questions(self):
        for query, want_source, want_word in GOLDEN:
            with self.subTest(query=query):
                result = self.engine.turn(query, now=1000.0)
                self.assertEqual(result.source, want_source)
                self.assertIn(want_word, result.base_reply)

    def test_off_domain_abstains(self):
        result = self.engine.turn("今天股票怎么买？", now=1000.0)
        self.assertFalse(result.source.startswith("feynman_kb"),
                         (result.source, result.base_reply))
        self.assertTrue(result.reply)

    def test_no_cross_language_leak(self):
        result = self.engine.turn("原子是什么？", now=1000.0)
        latin = sum(1 for ch in result.base_reply if ch.isascii() and ch.isalpha())
        self.assertLess(latin / max(len(result.base_reply), 1), 0.2,
                        result.base_reply)


if __name__ == "__main__":
    unittest.main()
