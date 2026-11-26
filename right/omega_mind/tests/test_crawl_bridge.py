"""Crawler → right-brain candidate bridge.

The crawler only writes `domain_extensions.json`; this bridge turns its
co-occurrence entities into quarantined candidate pairs (`same_domain`,
`origin="cooccur"`, empty mechanism). Nothing it produces may speak: the pairs
must pass the same one-to-one review as mined candidates before they can become
edges.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_RIGHT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RIGHT))

_SCRIPT = _RIGHT / "scripts" / "import_crawl_candidates.py"
_spec = importlib.util.spec_from_file_location("import_crawl_candidates", _SCRIPT)
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


def doc(doc_id, title="", answer="", weights=None, language="zh"):
    return SimpleNamespace(id=doc_id, title=title, answer=answer,
                           weights=weights or {}, language=language)


class TestMatching(unittest.TestCase):
    def test_title_beats_answer_beats_weight_bigrams(self):
        in_title = doc("a", title="量子退火是什么")
        in_answer = doc("b", answer="量子退火利用量子涨落")
        in_weights = doc("c", weights={"量子": 1.0, "子退": 1.0, "退火": 1.0})
        self.assertEqual(bridge.match_strength(in_title, "量子退火"), 2.0)
        self.assertEqual(bridge.match_strength(in_answer, "量子退火"), 1.0)
        self.assertEqual(bridge.match_strength(in_weights, "量子退火"), 0.5)
        self.assertEqual(bridge.match_strength(doc("d"), "量子退火"), 0.0)

    def test_weak_bigram_hits_are_dropped_when_a_strong_match_exists(self):
        docs = [doc("strong", title="量子退火"),
                doc("weak", weights={"量子": 1.0, "子退": 1.0, "退火": 1.0})]
        matched = bridge.match_docs(docs, "量子退火")
        self.assertEqual([d.id for _s, d in matched], ["strong"])

    def test_english_cards_are_not_matched_with_chinese_entities(self):
        docs = [doc("en", title="量子退火", language="en"),
                doc("zh", title="量子退火")]
        self.assertEqual([d.id for _s, d in bridge.match_docs(docs, "量子退火")],
                         ["zh"])

    def test_matches_are_deterministic_and_capped(self):
        docs = [doc(f"c{i}", title="熵") for i in range(10)]
        first = [d.id for _s, d in bridge.match_docs(docs, "熵", limit=3)]
        second = [d.id for _s, d in bridge.match_docs(list(reversed(docs)), "熵", limit=3)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)


class TestPairs(unittest.TestCase):
    def test_pairs_are_quarantined_same_domain_candidates(self):
        docs = [doc("opt-1", title="色散"), doc("opt-2", answer="色散让光分开"),
                doc("misc", title="别的东西")]
        report = bridge.build_report(
            docs, {"crawl_runs": [{"entity": "色散", "status": "ok",
                                   "source_url": "https://zh.wikipedia.org/wiki/色散"}]})
        self.assertEqual(len(report["pairs"]), 1)
        pair = report["pairs"][0]
        self.assertEqual((pair["a"], pair["b"]), ("opt-1", "opt-2"))
        self.assertEqual(pair["kind"], "same_domain")
        self.assertEqual(pair["origin"], "cooccur")
        self.assertEqual(pair["mechanism"], "")
        self.assertEqual(pair["shared"], ["色散"])
        self.assertGreater(pair["score"], 0)
        self.assertEqual(report["hits"]["色散"][0]["id"], "opt-1")
        self.assertEqual(report["unmatched"], [])
        self.assertEqual(report["sources"]["色散"], "https://zh.wikipedia.org/wiki/色散")

    def test_two_entities_on_the_same_pair_add_up(self):
        docs = [doc("a", title="熵与温度"), doc("b", answer="熵和温度有关")]
        hits = bridge.build_entity_hits(docs, ["熵", "温度"])
        pairs = bridge.build_pairs(hits)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["shared"], sorted(["熵", "温度"]))
        self.assertAlmostEqual(pairs[0]["score"], 3.0)

    def test_entities_come_from_successful_runs_and_keep_order(self):
        ext = {"crawl_runs": [
            {"entity": "甲", "status": "no_summary"},
            {"entity": "乙", "status": "ok"},
            {"entity": "乙", "status": "ok"},
            {"entity": "丙", "status": "ok"},
        ]}
        self.assertEqual(bridge.crawl_entities(ext), ["乙", "丙"])

    def test_old_library_without_runs_falls_back_to_log_keys(self):
        ext = {"crawl_log": {"甲": {}, "乙": {}}}
        self.assertEqual(bridge.crawl_entities(ext), ["甲", "乙"])


if __name__ == "__main__":
    unittest.main()
