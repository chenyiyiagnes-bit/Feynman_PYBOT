"""Typed association edges: schema, candidate quarantine, spoken line.

Rules borrowed from the Atlas design and enforced here:

* an edge has a *kind* and a *reason* (mechanism + strongest difference),
  not just a weight;
* machine-mined edges (origin="cooccur") are candidates and may never be
  spoken — attention is not evidence;
* at most one association line, and it never changes which card answered.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from omega_mind import assoc
from omega_mind.feynman_kb import FeynmanKB


class FakeDoc:
    def __init__(self, doc_id: str, weights: dict):
        self.id = doc_id
        self.weights = weights


def _edge(**over):
    base = dict(a="x", b="y", kind="same_mechanism", origin="human",
                mechanism="同一个机制", difference="差别在这里",
                mechanism_en="one mechanism", difference_en="where it parts")
    base.update(over)
    return assoc.Edge(**base)


class TestEdgeSchema(unittest.TestCase):
    def test_a_valid_human_edge_has_no_problems(self):
        self.assertEqual(assoc.problems(_edge()), [])

    def test_self_loop_is_rejected(self):
        self.assertTrue(assoc.problems(_edge(a="x", b="x")))

    def test_unknown_kind_is_rejected(self):
        self.assertTrue(assoc.problems(_edge(kind="vibes")))

    def test_unknown_origin_is_rejected(self):
        self.assertTrue(assoc.problems(_edge(origin="maybe")))

    def test_human_spoken_edge_needs_a_mechanism(self):
        self.assertTrue(assoc.problems(_edge(mechanism="")))

    def test_same_mechanism_needs_a_difference(self):
        # "Same mechanism" without "where it breaks" is the failure the
        # Atlas method calls out: a link that explains nothing.
        self.assertTrue(assoc.problems(_edge(difference="")))

    def test_sentence_fields_may_not_contain_a_full_stop(self):
        self.assertTrue(assoc.problems(_edge(mechanism="一。二")))

    def test_cooccur_edge_needs_no_prose(self):
        e = _edge(origin="cooccur", kind="same_domain",
                  mechanism="", difference="")
        self.assertEqual(assoc.problems(e), [])


class TestLoadAndQuarantine(unittest.TestCase):
    def _write(self, rows):
        path = Path(tempfile.mkdtemp(prefix="assoc-")) / "edges.jsonl"
        path.write_text("\n".join(rows), encoding="utf-8")
        return path

    def test_invalid_rows_are_dropped_not_fatal(self):
        import json
        good = json.dumps(_edge().to_row(), ensure_ascii=False)
        bad = json.dumps({"a": "x", "b": "x", "kind": "nope"}, ensure_ascii=False)
        graph = assoc.load(self._write([good, bad]))
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.dropped, 1)

    def test_candidates_are_kept_but_not_confirmed(self):
        import json
        cand = assoc.Edge(a="x", b="y", kind="same_domain", origin="cooccur")
        graph = assoc.load(self._write([json.dumps(cand.to_row())]))
        self.assertEqual(graph.candidates("x"), [cand])
        self.assertEqual(graph.confirmed("x"), [])


class TestSentence(unittest.TestCase):
    def test_same_mechanism_is_one_line_with_both_halves(self):
        graph = assoc.AssocGraph([_edge(mechanism="都是阻尼", difference="一个是摩擦")])
        line = graph.sentence("x", titles={"x": "甲卡", "y": "乙卡"})
        self.assertIn("乙卡", line)
        self.assertIn("都是阻尼", line)
        self.assertIn("一个是摩擦", line)
        self.assertLessEqual(len(line), assoc.MAX_SENTENCE_CHARS)

    def test_symmetric_edge_speaks_from_either_side(self):
        graph = assoc.AssocGraph([_edge()])
        self.assertIsNotNone(graph.sentence("x"))
        self.assertIsNotNone(graph.sentence("y"))

    def test_premise_edge_only_speaks_from_the_conclusion(self):
        edge = _edge(kind="premise")
        graph = assoc.AssocGraph([edge])
        self.assertIsNotNone(graph.sentence("x"))
        self.assertIsNone(graph.sentence("y"))

    def test_boundary_edge_names_the_other_card(self):
        edge = _edge(kind="boundary")
        line = assoc.AssocGraph([edge]).sentence("x", titles={"y": "乙卡"})
        self.assertIn("乙卡", line)

    def test_same_domain_is_data_not_prose(self):
        edge = _edge(kind="same_domain", difference="")
        self.assertIsNone(assoc.AssocGraph([edge]).sentence("x"))

    def test_candidate_never_speaks(self):
        edge = _edge(origin="cooccur")
        self.assertIsNone(assoc.AssocGraph([edge]).sentence("x"))

    def test_excluded_card_is_not_surfaced(self):
        graph = assoc.AssocGraph([_edge()])
        self.assertIsNone(graph.sentence("x", exclude={"y"}))

    def test_english_needs_english_prose(self):
        graph = assoc.AssocGraph([_edge(mechanism_en="",
                                        difference_en="")])
        self.assertIsNone(graph.sentence("x", language="en"))
        en = _edge(mechanism_en="same damping", difference_en="friction vs air")
        self.assertIsNotNone(
            assoc.AssocGraph([en]).sentence("x", language="en"))

    def test_english_never_borrows_a_chinese_title(self):
        # The two languages are branches, not a fallback chain: if the other
        # card has no English name there is no English line.
        en = _edge(mechanism_en="same damping", difference_en="friction vs air")
        graph = assoc.AssocGraph([en])
        self.assertIsNone(
            graph.sentence("x", language="en", titles={"y": "乙卡"}))
        self.assertIsNotNone(
            graph.sentence("x", language="en", titles={"y": "Card Y"}))

    def test_english_prose_may_not_contain_chinese(self):
        bad = _edge(mechanism_en="same damping 都是阻尼",
                    difference_en="friction vs air")
        self.assertTrue(assoc.problems(bad))

    def test_priority_prefers_same_mechanism_over_premise(self):
        graph = assoc.AssocGraph([
            _edge(kind="premise", b="p"),
            _edge(kind="same_mechanism", b="m"),
        ])
        line = graph.sentence("x", titles={"m": "机制卡", "p": "前提卡"})
        self.assertIn("机制卡", line)


class TestMining(unittest.TestCase):
    def _load_script(self):
        path = ROOT / "scripts" / "mine_assoc.py"
        spec = importlib.util.spec_from_file_location("mine_assoc", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_shared_significant_tokens_become_a_candidate(self):
        miner = self._load_script()
        docs = [
            FakeDoc("a", {"折射": 4.0, "色散": 3.0, "彩虹": 3.0}),
            FakeDoc("b", {"折射": 4.0, "色散": 3.0, "雨滴": 3.0}),
            FakeDoc("c", {"量子": 4.0}),
            FakeDoc("d", {"熵": 4.0}),
        ]
        found = miner.candidates(docs)
        pairs = {tuple(sorted((row["a"], row["b"]))) for row in found}
        self.assertIn(("a", "b"), pairs)
        self.assertNotIn(("a", "c"), pairs)

    def test_candidates_are_quarantined(self):
        miner = self._load_script()
        docs = [FakeDoc("a", {"折射": 4.0, "色散": 3.0}),
                FakeDoc("b", {"折射": 4.0, "色散": 3.0}),
                FakeDoc("c", {"熵": 4.0})]
        for row in miner.candidates(docs):
            self.assertEqual(row["origin"], "cooccur")
            self.assertEqual(row["kind"], "same_domain")
            self.assertEqual(row["mechanism"], "")

    def test_same_concept_variants_are_skipped(self):
        # C_RELAX and U0_RELAX are the same idea filed twice; an association
        # edge between them would be a duplicate, not a link.
        miner = self._load_script()
        docs = [FakeDoc("C_RELAX", {"弛豫": 4.0, "平衡": 3.0}),
                FakeDoc("U0_RELAX", {"弛豫": 4.0, "平衡": 3.0,
                                     "时间常数": 3.0, "衰减": 3.0}),
                FakeDoc("opt-001", {"时间常数": 4.0, "衰减": 3.0,
                                    "激光": 3.0}),
                FakeDoc("opt-002", {"折射": 4.0})]
        pairs = {tuple(sorted((row["a"], row["b"])))
                 for row in miner.candidates(docs)}
        self.assertNotIn(("C_RELAX", "U0_RELAX"), pairs)
        self.assertIn(("U0_RELAX", "opt-001"), pairs)


class TestRuntimeLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not cls.kb.available():
            raise unittest.SkipTest("feynman-open knowledge not on this machine")

    def test_default_lookup_has_no_association(self):
        found = self.kb.lookup("极化是什么")
        self.assertEqual(found.source, "feynman_kb:opt-020")
        self.assertIsNone(found.association)

    def test_association_never_changes_the_source(self):
        plain = self.kb.lookup("极化是什么")
        with_line = self.kb.lookup("极化是什么", associate=True)
        self.assertEqual(plain.source, with_line.source)
        self.assertEqual(plain.score, with_line.score)
        self.assertIsNotNone(with_line.association)

    def test_dielectric_polarization_links_to_refraction(self):
        found = self.kb.lookup("极化是什么", associate=True)
        self.assertIn("折射", found.association)

    def test_boundary_edge_disambiguates_vacuum_polarization(self):
        # 真空极化 must say it is not the dielectric story.
        found = self.kb.lookup("真空极化是什么", associate=True)
        self.assertEqual(found.source, "feynman_kb:qed-007")
        self.assertIn("电介质", found.association)

    def test_no_match_gets_no_association(self):
        found = self.kb.lookup("今天中午吃什么", associate=True)
        self.assertIsNone(found.text)
        self.assertIsNone(found.association)


class TestEngineSpeaks(unittest.TestCase):
    def test_engine_appends_the_association_after_the_answer(self):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False)
        result = mind.turn("极化是什么", now=1000.0)
        self.assertEqual(result.source, "feynman_kb:opt-020")
        self.assertIn("同一个机制", result.reply)
        self.assertIsNotNone(result.retrieval.get("association"))

    def test_english_turn_gets_the_english_branch(self):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False, language="en")
        result = mind.turn("What is vacuum polarization?", now=1000.0)
        self.assertTrue(result.source.startswith("feynman_kb_en:"),
                        result.source)
        association = result.retrieval.get("association")
        self.assertIn("Don't confuse it with", association)
        self.assertNotRegex(association, r"[\u4e00-\u9fff]")


class TestCandidateReview(unittest.TestCase):
    """Every mined pair has a human verdict, and promotions are real edges."""

    def _load(self, name):
        import json
        path = ROOT / "scripts" / "data" / name
        if not path.exists():
            self.skipTest(f"{name} not on this machine")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_every_candidate_has_a_verdict(self):
        cands = self._load("assoc_candidates.json")["pairs"]
        review = self._load("assoc_review.json")["decisions"]
        self.assertEqual(len(cands), len(review))
        keys = {(c["a"], c["b"]) for c in cands}
        self.assertEqual(keys, {(r["a"], r["b"]) for r in review})
        verdicts = {r["verdict"] for r in review}
        self.assertLessEqual(verdicts, {"promote", "reject", "already"})
        for row in review:
            self.assertTrue(row["reason"].strip(), row)

    def test_promoted_pairs_exist_as_confirmed_edges(self):
        review = self._load("assoc_review.json")["decisions"]
        graph = assoc.load()
        have = {(min(e.a, e.b), max(e.a, e.b))
                for e in graph.edges if e.origin == "human"}
        for row in review:
            if row["verdict"] != "promote":
                continue
            key = (min(row["a"], row["b"]), max(row["a"], row["b"]))
            self.assertIn(key, have, row)


class TestCommittedEdges(unittest.TestCase):
    """The shipped data file obeys the same rules as the code."""

    @classmethod
    def setUpClass(cls):
        cls.graph = assoc.load()
        cls.kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not cls.kb.available():
            raise unittest.SkipTest("feynman-open knowledge not on this machine")
        cls.titles = {doc.id: doc.title for doc in cls.kb.docs}
        cls.ids = set(cls.titles)

    def test_file_exists_and_has_fifty_confirmed_edges(self):
        human = [e for e in self.graph.edges if e.origin == "human"]
        self.assertGreaterEqual(len(human), 50, len(human))
        self.assertEqual(self.graph.dropped, 0)

    def test_every_edge_is_valid(self):
        for edge in self.graph.edges:
            self.assertEqual(assoc.problems(edge), [], edge)

    def test_every_card_id_exists_in_the_corpus(self):
        for edge in self.graph.edges:
            self.assertIn(edge.a, self.ids, edge)
            self.assertIn(edge.b, self.ids, edge)

    def test_every_spoken_edge_fits_the_sentence_cap(self):
        for edge in self.graph.edges:
            if edge.origin != "human" or edge.kind not in assoc.SPOKEN:
                continue
            sides = ("x", "y") if edge.kind == "same_mechanism" else ("x",)
            for _ in sides:
                line = self.graph.sentence(edge.a if edge.kind != "same_mechanism"
                                           else edge.a, titles=self.titles)
                self.assertIsNotNone(line, edge)
                self.assertLessEqual(len(line), assoc.MAX_SENTENCE_CHARS, edge)
                break

    def test_every_spoken_edge_has_english_prose(self):
        for edge in self.graph.edges:
            if edge.origin != "human" or edge.kind not in assoc.SPOKEN:
                continue
            self.assertTrue(edge.mechanism_en.strip(), edge)
            if edge.kind in ("same_mechanism", "boundary"):
                self.assertTrue(edge.difference_en.strip(), edge)

    def test_english_branch_speaks_with_english_titles(self):
        en_titles = {doc.id: doc.title for doc in self.kb.docs_en}
        line = self.graph.sentence("opt-020", language="en",
                                   titles=en_titles)
        self.assertIsNotNone(line)
        self.assertIn("same mechanism", line)
        self.assertIn("refractive index", line)
        self.assertNotRegex(line, r"[\u4e00-\u9fff]")
        self.assertLessEqual(len(line), assoc.MAX_SENTENCE_CHARS_EN)

    def test_kind_coverage_is_not_one_note(self):
        kinds = {e.kind for e in self.graph.edges if e.origin == "human"}
        self.assertIn("same_mechanism", kinds)
        self.assertIn("premise", kinds)
        self.assertIn("boundary", kinds)


if __name__ == "__main__":
    unittest.main()
