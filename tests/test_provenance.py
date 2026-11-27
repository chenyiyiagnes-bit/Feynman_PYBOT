"""Provenance has one canonical shape; the others are read, not invented."""

from __future__ import annotations

import unittest

from omega_mind import provenance


class TestEntries(unittest.TestCase):
    def test_a_quote_support_block_passes_through(self):
        entry = {"source": "FLP I-1-1", "quote": "光。",
                 "quote_sha256": "0" * 64}
        self.assertEqual(provenance.entries({"support": [entry]}), [entry])

    def test_a_cite_support_block_passes_through(self):
        row = {"support": [{"source": "FLP I-1-1"}]}
        self.assertEqual(provenance.entries(row), row["support"])

    def test_a_top_level_string_is_a_cite(self):
        self.assertEqual(provenance.entries({"source": " 生平 "}),
                         [{"source": "生平"}])

    def test_a_book_chapter_mapping_is_a_cite(self):
        self.assertEqual(
            provenance.entries({"source": {"book": "道德经", "chapter": "第一章"}}),
            [{"source": "道德经 · 第一章"}])

    def test_no_source_at_all_is_empty(self):
        self.assertEqual(provenance.entries({}), [])
        self.assertFalse(provenance.cited({}))

    def test_a_malformed_support_is_declared_but_empty(self):
        row = {"support": "not a list"}
        self.assertEqual(provenance.entries(row), [])
        self.assertTrue(provenance.declared(row))
        self.assertFalse(provenance.cited(row))

    def test_support_wins_over_a_top_level_source(self):
        row = {"support": [{"source": "a"}], "source": "b"}
        self.assertEqual(provenance.entries(row), [{"source": "a"}])


if __name__ == "__main__":
    unittest.main()
