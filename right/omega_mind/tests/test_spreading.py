"""Opt-in spreading retrieval: neighbours, the precision gate, the bonus."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import reflect, spreading
from omega_mind.feynman_kb import FeynmanKB


class FakeDoc:
    def __init__(self, doc_id: str, weights: dict):
        self.id = doc_id
        self.weights = weights


class TestNpmiNeighbors(unittest.TestCase):
    def test_co_occurring_tokens_become_neighbours(self):
        docs = [FakeDoc(f"d{i}", {"阻尼": 4.0, "振动": 1.0})
                for i in range(3)]
        docs += [FakeDoc(f"e{i}", {"彩虹": 4.0, "水滴": 1.0}) for i in range(3)]
        neighbours = spreading.npmi_neighbors(docs, min_cooccur=2)
        self.assertIn("阻尼", neighbours)
        self.assertAlmostEqual(neighbours["阻尼"][0][1], 1.0, places=6)
        self.assertEqual(neighbours["阻尼"][0][0], "振动")

    def test_a_single_co_occurrence_is_not_enough(self):
        docs = [FakeDoc("a", {"阻尼": 4.0, "振动": 1.0}),
                FakeDoc("b", {"彩虹": 4.0}),
                FakeDoc("c", {"彩虹": 4.0})]
        neighbours = spreading.npmi_neighbors(docs, min_cooccur=2)
        self.assertNotIn("阻尼", neighbours)

    def test_neighbour_lists_are_capped(self):
        # 中心 appears in all five docs; the 旁 tokens appear with it in only
        # three, so NPMI is defined and the list needs trimming.
        docs = []
        for i in range(3):
            weights = {"中心": 4.0}
            weights.update({f"旁{k}": 1.0 for k in range(8)})
            docs.append(FakeDoc(f"d{i}", weights))
        docs.append(FakeDoc("e0", {"彩虹": 4.0}))
        docs.append(FakeDoc("e1", {"水滴": 4.0}))
        neighbours = spreading.npmi_neighbors(docs, top=5, min_cooccur=2)
        self.assertEqual(len(neighbours["中心"]), 5)


class TestPrecisionGate(unittest.TestCase):
    def test_title_hit_opens_the_door(self):
        doc = FakeDoc("a", {"阻尼": 4.0})
        self.assertTrue(spreading.precision_open({"阻尼"}, doc))

    def test_body_only_hit_does_not(self):
        doc = FakeDoc("a", {"阻尼": 1.0})
        self.assertFalse(spreading.precision_open({"阻尼"}, doc))

    def test_bonus_uses_the_strongest_neighbour(self):
        doc = FakeDoc("a", {"振动": 1.0, "声音": 1.0})
        neighbours = {"阻尼": [("振动", 0.8), ("声音", 0.4)]}
        self.assertAlmostEqual(
            spreading.spread_bonus({"阻尼"}, doc, neighbours),
            spreading.SPREAD_ENERGY * 0.8)

    def test_no_neighbour_means_no_bonus(self):
        doc = FakeDoc("a", {"彩虹": 1.0})
        neighbours = {"阻尼": [("振动", 0.8)]}
        self.assertEqual(spreading.spread_bonus({"阻尼"}, doc, neighbours), 0.0)


class TestKbSwitch(unittest.TestCase):
    def test_spreading_is_off_by_default(self):
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-spread-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        self.assertFalse(kb.spreading)

    def test_turning_it_on_changes_golden_answers(self):
        # This is not a bug report but the reason the feature stays opt-in:
        # with the real corpus, spreading changes which card answers
        # 熵是什么 (sm-005 instead of the guardrail) and lifts part-004 until
        # it ties part-006 at 1.0 on 中微子是什么, where only the verbatim-
        # alias tie-break keeps the neutrino card on top. Both answers are
        # acceptable to the golden net, but the winning margin is gone.
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-spread-"),
                       spreading=True)
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        self.assertTrue(kb.spreading)
        kb.answer("什么是阻尼")
        self.assertIsNotNone(kb._spread)
        plain = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-spread-"))
        self.assertEqual(plain.answer("熵是什么")[1],
                         "feynman_kb:thermo-entropy-disorder")
        self.assertEqual(kb.answer("熵是什么")[1], "feynman_kb:sm-005")
        top = kb.search("中微子是什么", limit=2)
        self.assertEqual([doc.id for _, doc in top], ["part-006", "part-004"])
        self.assertEqual(top[0][0], top[1][0])
        margin = plain.search("中微子是什么", limit=2)
        self.assertGreater(margin[0][0], margin[1][0])
        ok, failures = reflect.golden_ok(kb)
        self.assertTrue(ok, failures)

if __name__ == "__main__":
    unittest.main()
