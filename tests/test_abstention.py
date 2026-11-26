"""Abstention: off-domain questions must not be answered with card text.

These guard the "不知道就说不知道" property. If someone lowers the retrieval
thresholds or rewrites the fallback, this is what should stop them.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from app.runtime import build

PERSONAS = Path(__file__).resolve().parents[1] / "personas"

OFF_DOMAIN = [
    "今天股票怎么买？",
    "帮我写一封辞职信",
    "红烧肉怎么做",
    "明天天气怎么样",
    "给我讲讲明朝历史",
]

ABSTAIN_MARK = "翻翻记忆库"


class TestAbstention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = build("feynman", personas_root=PERSONAS, enable_bridge=False)

    def test_off_domain_does_not_answer_from_cards(self):
        for query in OFF_DOMAIN:
            with self.subTest(query=query):
                result = self.engine.turn(query, now=1000.0)
                self.assertFalse(result.source.startswith("feynman_kb"),
                                 (result.source, result.base_reply))

    def test_abstention_says_it_plainly(self):
        for query in OFF_DOMAIN:
            with self.subTest(query=query):
                reply = self.engine.turn(query, now=1000.0).reply
                self.assertIn(ABSTAIN_MARK, reply)

    def test_english_off_domain_abstains_too(self):
        engine = build("feynman", language="en", personas_root=PERSONAS,
                       enable_bridge=False)
        for query in ("How do I buy stocks?", "Cook me a steak recipe"):
            with self.subTest(query=query):
                result = engine.turn(query, now=1000.0)
                self.assertFalse(result.source.startswith("feynman_kb"),
                                 (result.source, result.base_reply))


if __name__ == "__main__":
    unittest.main()
