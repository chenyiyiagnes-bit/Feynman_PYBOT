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


REPO = Path(__file__).resolve().parents[3]
PERSONA_EDGES = (REPO / "personas" / "feynman-zh" / "data" / "assoc"
                 / "assoc_edges.jsonl")


def _edges_path() -> Path:
    """Edges are local-only (gitignored); they may sit with the persona or
    with the engine, whichever this machine has."""
    for path in (PERSONA_EDGES, assoc.DEFAULT_PATH):
        if path.exists():
            return path
    return PERSONA_EDGES


def _edge_data_present() -> bool:
    path = _edges_path()
    return path.exists() and bool(assoc.load(path).edges)


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


def _speaking_sides(edge):
    """Which endpoints may speak this edge (symmetric vs one-way)."""
    return (edge.a, edge.b) if edge.kind in assoc.SYMMETRIC else (edge.a,)


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

    def test_same_equation_needs_a_mapping(self):
        e = _edge(kind="same_equation", mapping="")
        self.assertTrue(any("mapping" in p for p in assoc.problems(e)))
        ok = _edge(kind="same_equation", mapping="x↔q、m↔L",
                   mapping_en="x↔q, m↔L")
        self.assertEqual(assoc.problems(ok), [])

    def test_mapping_must_be_pairs(self):
        bad = _edge(kind="same_equation", mapping="x~q")
        self.assertTrue(assoc.problems(bad))
        empty_side = _edge(kind="same_equation", mapping="x↔")
        self.assertTrue(assoc.problems(empty_side))

    def test_limit_of_needs_a_condition(self):
        e = _edge(kind="limit_of", condition="", condition_en="")
        self.assertTrue(any("condition" in p for p in assoc.problems(e)))
        ok = _edge(kind="limit_of", condition="ζ≫1", condition_en="zeta >> 1")
        self.assertEqual(assoc.problems(ok), [])

    def test_tension_needs_the_missing_condition(self):
        e = _edge(kind="tension", difference="", difference_en="")
        self.assertTrue(any("difference" in p for p in assoc.problems(e)))
        ok = _edge(kind="tension", mechanism="一条说可逆，一条说熵增")
        self.assertEqual(assoc.problems(ok), [])

    def test_english_mapping_may_not_contain_chinese(self):
        bad = _edge(kind="same_equation", mapping="x↔q",
                    mapping_en="x↔电荷")
        self.assertTrue(assoc.problems(bad))


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

    def test_same_equation_names_the_mapping_and_inverts_it(self):
        edge = _edge(kind="same_equation", mechanism="同一个二阶方程",
                     mapping="x↔q、m↔L",
                     mechanism_en="the same second-order equation",
                     mapping_en="x↔q, m↔L")
        graph = assoc.AssocGraph([edge])
        forward = graph.sentence("x", titles={"y": "乙卡"})
        self.assertIn("x↔q", forward)
        self.assertIn("乙卡", forward)
        backward = graph.sentence("y", titles={"x": "甲卡"})
        self.assertIn("q↔x", backward)
        self.assertNotIn("x↔q", backward)

    def test_same_equation_outranks_same_mechanism(self):
        graph = assoc.AssocGraph([
            _edge(kind="same_mechanism", b="m"),
            _edge(kind="same_equation", b="eq", mapping="x↔q",
                  mapping_en="x↔q"),
        ])
        line = graph.sentence("x", titles={"m": "机制卡", "eq": "方程卡"})
        self.assertIn("方程卡", line)
        self.assertIn("同一个方程", line)

    def test_limit_of_speaks_only_from_the_special_case(self):
        edge = _edge(kind="limit_of", condition="ζ≫1", condition_en="zeta >> 1")
        graph = assoc.AssocGraph([edge])
        line = graph.sentence("x", titles={"y": "一般理论"})
        self.assertIn("一般理论", line)
        self.assertIn("ζ≫1", line)
        self.assertIsNone(graph.sentence("y", titles={"x": "特例"}))

    def test_tension_states_the_missing_condition(self):
        edge = _edge(kind="tension", mechanism="一条说可逆，一条说熵增",
                     difference="可逆说的是全部微观自由度")
        line = assoc.AssocGraph([edge]).sentence("x", titles={"y": "乙卡"})
        self.assertIn("看着矛盾", line)
        self.assertIn("乙卡", line)
        self.assertIn("缺的条件", line)
        self.assertIn("全部微观自由度", line)

    def test_continuity_line_continues_the_previous_card(self):
        graph = assoc.AssocGraph([_edge()])
        line = graph.sentence("x", titles={"y": "乙卡"}, continuity_with="y")
        self.assertIn("接着刚才", line)
        self.assertIn("乙卡", line)
        self.assertNotIn("这和", line)

    def test_continuity_needs_an_edge_to_the_previous_card(self):
        graph = assoc.AssocGraph([_edge(b="z")])
        line = graph.sentence("x", titles={"z": "丙卡"}, continuity_with="y")
        self.assertNotIn("接着刚才", line)
        self.assertIn("丙卡", line)  # the ordinary line still speaks

    def test_continuity_falls_back_when_it_does_not_fit(self):
        long_mechanism = "很长" * 80
        edge = _edge(mechanism=long_mechanism)
        graph = assoc.AssocGraph([edge])
        line = graph.sentence("x", titles={"y": "乙卡"}, continuity_with="y")
        self.assertTrue(line is None or "接着刚才" not in line)

    def test_english_continuity_says_following_on(self):
        edge = _edge(mechanism_en="same damping",
                     difference_en="friction vs air")
        line = assoc.AssocGraph([edge]).sentence(
            "x", language="en", titles={"y": "Card Y"},
            continuity_with="y")
        self.assertIn("Following on from", line)
        self.assertNotRegex(line, r"[一-鿿]")


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
        if not _edge_data_present():
            raise unittest.SkipTest("data/assoc/assoc_edges.jsonl not present")
        cls.kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"),
                                     assoc_path=str(_edges_path()))
        if not cls.kb.available():
            raise unittest.SkipTest("feynman-open knowledge not on this machine")

    def test_default_lookup_has_no_association(self):
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
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
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"),
                                     assoc_path=str(_edges_path()))
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
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        # One package per language: an English turn needs the English
        # package, not a language switch inside a Chinese index.
        kb = FeynmanKB(language="en",
                       learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"),
                       assoc_path=str(_edges_path()))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False, language="en")
        result = mind.turn("What is vacuum polarization?", now=1000.0)
        self.assertTrue(result.source.startswith(kb.source_root + ":"),
                        result.source)
        association = result.retrieval.get("association")
        self.assertIn("Don't confuse it with", association)
        self.assertNotRegex(association, r"[\u4e00-\u9fff]")

    def test_next_turn_continues_the_previous_card(self):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False)
        first = mind.turn("\u9971\u548c\u662f\u4ec0\u4e48", now=1000.0)
        self.assertEqual(first.source, "feynman_kb:C_SATURATE")
        second = mind.turn("\u6b63\u53cd\u9988\u662f\u4ec0\u4e48", now=1001.0)
        self.assertEqual(second.source, "feynman_kb:C_FEEDBACK")
        association = second.retrieval.get("association")
        self.assertIn("\u63a5\u7740\u521a\u624d", association)
        self.assertIn("\u9971\u548c", association)
        self.assertIn("\u6781\u9650", association)

    def test_small_talk_breaks_the_thread(self):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False)
        mind.turn("\u9971\u548c\u662f\u4ec0\u4e48", now=1000.0)
        mind.turn("\u4eca\u5929\u5929\u6c14\u4e0d\u9519", now=1001.0)
        third = mind.turn("\u6b63\u53cd\u9988\u662f\u4ec0\u4e48", now=1002.0)
        association = third.retrieval.get("association") or ""
        self.assertNotIn("\u63a5\u7740\u521a\u624d", association)

    def test_asking_the_same_question_twice_does_not_continue(self):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        mind = MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False)
        mind.turn("\u9971\u548c\u662f\u4ec0\u4e48", now=1000.0)
        again = mind.turn("\u9971\u548c\u662f\u4ec0\u4e48", now=1001.0)
        association = again.retrieval.get("association") or ""
        self.assertNotIn("\u63a5\u7740\u521a\u624d", association)


class TestWhyChain(unittest.TestCase):
    """A bare "why?" after an answer walks one confirmed premise hop."""

    def _graph(self):
        return assoc.AssocGraph([
            _edge(a="x", b="y", kind="premise", mechanism="x 要先有 y",
                  mechanism_en="x needs y first", difference="",
                  difference_en=""),
            assoc.Edge(a="y", b="z", kind="premise", origin="cooccur",
                       mechanism="机器猜的", mechanism_en="machine guess"),
        ])

    def test_premise_step_returns_the_card_and_the_link(self):
        self.assertEqual(self._graph().premise_step("x"), ("y", "x 要先有 y"))

    def test_premise_step_speaks_english(self):
        self.assertEqual(self._graph().premise_step("x", language="en"),
                         ("y", "x needs y first"))

    def test_premise_points_one_way(self):
        self.assertIsNone(self._graph().premise_step("y"))

    def test_cooccurrence_is_never_a_premise(self):
        graph = assoc.AssocGraph([
            assoc.Edge(a="y", b="z", kind="premise", origin="cooccur",
                       mechanism="机器猜的")])
        self.assertIsNone(graph.premise_step("y"))

    def test_avoid_stops_the_chain_going_in_a_circle(self):
        self.assertIsNone(self._graph().premise_step("x", avoid={"y"}))


class TestWhyFollowUp(unittest.TestCase):
    """Engine-level: "为什么？" continues from the previous card's premise."""

    def _mind(self, language="zh"):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.pybot_bridge import PyBotBridge
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-why-"))
        if not kb.available():
            self.skipTest("feynman-open knowledge not on this machine")
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(), kb=kb,
            auto_save=False, language=language)

    def test_a_bare_why_walks_one_premise_hop(self):
        mind = self._mind()
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        self.assertEqual(first.source, "feynman_kb:opt-009")
        why = mind.turn("为什么？", now=1001.0)
        self.assertEqual(why.source, "feynman_kb:opt-002")
        self.assertTrue(why.base_reply)
        self.assertIn("折射率", why.base_reply)

    def test_the_hop_uses_the_english_branch_too(self):
        mind = self._mind("en")
        first = mind.turn("Why is a rainbow curved?", now=1000.0)
        self.assertEqual(first.source, "feynman_kb_en:opt-009")
        why = mind.turn("Why?", now=1001.0)
        self.assertEqual(why.source, "feynman_kb_en:opt-002")
        self.assertNotRegex(why.reply, r"[一-鿿]")

    def test_an_unrelated_turn_breaks_the_chain(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        mind.turn("今天天气不错", now=1001.0)
        why = mind.turn("为什么？", now=1002.0)
        self.assertNotEqual(why.source, "feynman_kb:opt-002")

    def test_a_why_with_its_own_subject_is_not_a_follow_up(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        why = mind.turn("为什么天空是蓝的", now=1001.0)
        self.assertNotEqual(why.source, "feynman_kb:opt-002")
        self.assertEqual(why.source, "feynman_kb:opt-012")


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
        if not _edge_data_present():
            self.skipTest("data/assoc/assoc_edges.jsonl not present")
        review = self._load("assoc_review.json")["decisions"]
        graph = assoc.load(_edges_path())
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
        if not _edge_data_present():
            raise unittest.SkipTest("data/assoc/assoc_edges.jsonl not present")
        cls.graph = assoc.load(_edges_path())
        cls.kb = FeynmanKB(learned_dir=tempfile.mkdtemp(prefix="omega-assoc-"),
                                     assoc_path=str(_edges_path()))
        cls.en_kb = FeynmanKB(language="en")
        if not cls.kb.available() or not cls.en_kb.available():
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
            for side in _speaking_sides(edge):
                line = self.graph.sentence(side, titles=self.titles)
                self.assertIsNotNone(line, edge)
                self.assertLessEqual(len(line), assoc.MAX_SENTENCE_CHARS, edge)

    def test_every_spoken_edge_has_english_prose(self):
        for edge in self.graph.edges:
            if edge.origin != "human" or edge.kind not in assoc.SPOKEN:
                continue
            self.assertTrue(edge.mechanism_en.strip(), edge)
            if edge.kind in ("same_mechanism", "same_equation", "boundary"):
                self.assertTrue(edge.difference_en.strip(), edge)
            if edge.kind == "same_equation":
                self.assertTrue(edge.mapping_en.strip(), edge)
            if edge.kind == "limit_of":
                self.assertTrue(edge.condition_en.strip(), edge)

    def test_english_branch_speaks_with_english_titles(self):
        en_titles = {doc.id: doc.title for doc in self.en_kb.docs}
        line = self.graph.sentence("opt-020", language="en",
                                   titles=en_titles)
        self.assertIsNotNone(line)
        self.assertIn("same mechanism", line)
        self.assertIn("refractive index", line)
        self.assertNotRegex(line, r"[\u4e00-\u9fff]")
        self.assertLessEqual(len(line), assoc.MAX_SENTENCE_CHARS_EN)

    def test_every_spoken_edge_speaks_in_both_languages(self):
        zh_titles = {doc.id: doc.title for doc in self.kb.docs}
        en_titles = {doc.id: doc.title for doc in self.en_kb.docs}
        for edge in self.graph.edges:
            if edge.origin != "human" or edge.kind not in assoc.SPOKEN:
                continue
            for side in _speaking_sides(edge):
                for lang, titles in (("zh", zh_titles), ("en", en_titles)):
                    line = self.graph.sentence(side, language=lang,
                                               titles=titles)
                    self.assertIsNotNone(line, (edge, side, lang))
                    if lang == "en":
                        self.assertNotRegex(line, r"[\u4e00-\u9fff]")

    def test_every_card_has_an_english_name(self):
        import re
        cjk = re.compile(r"[\u4e00-\u9fff]")
        for doc in self.en_kb.docs:
            self.assertTrue(doc.title.strip(), doc.id)
            self.assertIsNone(cjk.search(doc.title),
                              (doc.id, doc.title))

    def test_kind_coverage_is_not_one_note(self):
        kinds = {e.kind for e in self.graph.edges if e.origin == "human"}
        self.assertIn("same_mechanism", kinds)
        self.assertIn("premise", kinds)
        self.assertIn("boundary", kinds)
        self.assertIn("same_equation", kinds)
        self.assertIn("limit_of", kinds)
        self.assertIn("tension", kinds)


if __name__ == "__main__":
    unittest.main()
