"""Reflection: coming back to the same question from a different face.

Re-asking the very same question is not reflection — it wants the same
answer. Reflection is a short, subject-free nudge ("think again", "another
angle"): the mind keeps the card it just answered and says one face of it
that has not been said yet, drawn only from confirmed edges or the card's
own magnitude. When no face is left it says the card again, plainly, rather
than inventing an angle. A paraphrase with a subject of its own is a new
question and never enters this path.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import assoc
from omega_mind.config import get_persona
from omega_mind.engine import REFLECT_CAP, MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge

_CJK = __import__("re").compile(r"[㐀-鿿]")


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=None)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


def real_graph() -> assoc.AssocGraph:
    graph = assoc.load()
    if not graph.edges:
        raise unittest.SkipTest("no association edges on this machine")
    return graph


class TestFacetsAtTheGraph(unittest.TestCase):
    def setUp(self):
        self.graph = real_graph()

    def test_facets_follow_the_order_a_person_rethinks_in(self):
        # opt-009 carries a premise and an analogy (its boundary edge is
        # written from the other card's side and stays silent here): the
        # ground it stands on comes first, the analogy after it.
        keys = [key for key, _line in self.graph.facets("opt-009")]
        kinds = [key.rsplit(":", 1)[1] for key in keys]
        self.assertGreaterEqual(len(keys), 2)
        self.assertEqual(kinds[0], "premise")
        self.assertLess(kinds.index("premise"), kinds.index("same_mechanism"))

    def test_facets_are_confirmed_and_deterministic(self):
        first = self.graph.facets("opt-009")
        second = self.graph.facets("opt-009")
        self.assertEqual(first, second)
        confirmed = {edge.key for edge in self.graph.confirmed("opt-009")}
        for key, _line in first:
            self.assertIn(key, confirmed)

    def test_a_card_with_no_edges_has_no_facets(self):
        self.assertEqual(self.graph.facets("qed-010"), [])


class TestRethinkInTheKB(unittest.TestCase):
    def setUp(self):
        self.kb = real_kb()
        self.graph = real_graph()

    def test_a_spoken_facet_is_skipped(self):
        keys = [key for key, _line in self.graph.facets("opt-009")]
        first = self.kb.rethink("opt-009", spoken=())
        self.assertIsNotNone(first)
        self.assertEqual(first[1], keys[0])
        second = self.kb.rethink("opt-009", spoken=[keys[0]])
        self.assertIsNotNone(second)
        self.assertNotEqual(second[1], first[1])
        self.assertNotEqual(second[0], first[0])

    def test_no_facet_left_returns_none(self):
        keys = [key for key, _line in self.graph.facets("opt-009")]
        self.assertIsNone(self.kb.rethink("opt-009", spoken=keys))

    def test_an_edge_free_card_cannot_be_reflected(self):
        self.assertIsNone(self.kb.rethink("qed-010"))

    def test_english_reflection_never_mixes_languages(self):
        keys = [key for key, _line in self.graph.facets("opt-009")]
        hit = self.kb.rethink("opt-009", language="en", spoken=())
        if hit is None:
            self.skipTest("opt-009 has no fully English facet")
        self.assertFalse(_CJK.search(hit[0]))

    def test_the_card_can_be_said_again_plainly(self):
        body = self.kb.card_body("opt-009")
        self.assertTrue(body)
        found = self.kb.lookup("彩虹为什么是弯的", limit=3)
        self.assertEqual(found.source.split(":", 1)[1], "opt-009")


class TestReflectionInTheEngine(unittest.TestCase):
    def _mind(self, language="zh"):
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False, language=language)

    def test_reflection_keeps_the_card_and_changes_the_face(self):
        mind = self._mind()
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        again = mind.turn("再想想", now=1001.0)
        self.assertEqual(again.source, first.source)
        self.assertNotEqual(again.base_reply, first.base_reply)
        self.assertTrue(again.retrieval.get("reflection"))

    def test_two_reflections_never_say_the_same_line(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        lines = []
        for step in range(1, REFLECT_CAP + 1):
            result = mind.turn("换个角度", now=1000.0 + step)
            self.assertTrue(result.retrieval.get("reflection"))
            self.assertNotEqual(result.retrieval["reflection"], "repeat")
            lines.append(result.base_reply)
        self.assertEqual(len(lines), len(set(lines)))

    def test_the_cap_stops_new_faces_but_keeps_the_card(self):
        mind = self._mind()
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        for step in range(1, REFLECT_CAP + 1):
            mind.turn("再想想", now=1000.0 + step)
        after = mind.turn("再想想", now=1000.0 + REFLECT_CAP + 1)
        self.assertEqual(after.retrieval.get("reflection"), "repeat")
        self.assertEqual(after.source, first.source)

    def test_a_card_with_nothing_new_is_said_again(self):
        mind = self._mind()
        mind.turn("原子有多大", now=1000.0)
        card = mind._last_card
        again = mind.turn("再想想", now=1001.0)
        self.assertEqual(again.retrieval.get("reflection"), "repeat")
        self.assertEqual(again.source.split(":", 1)[1], card)

    def test_an_identical_repeat_is_not_a_reflection(self):
        mind = self._mind()
        first = mind.turn("彩虹为什么是弯的", now=1000.0)
        again = mind.turn("彩虹为什么是弯的", now=1001.0)
        self.assertEqual(again.base_reply, first.base_reply)
        self.assertNotIn("reflection", again.retrieval)

    def test_a_new_subject_breaks_the_thread(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        entropy = mind.turn("熵是什么", now=1001.0)
        self.assertNotEqual(entropy.source, "feynman_kb:opt-009")
        result = mind.turn("再想想", now=1002.0)
        self.assertEqual(result.source, entropy.source)

    def test_with_no_card_before_it_there_is_nothing_to_reflect_on(self):
        mind = self._mind()
        result = mind.turn("再想想", now=1000.0)
        self.assertNotIn("reflection", result.retrieval)
        self.assertTrue(result.reply)

    def test_spoken_facets_survive_a_snapshot(self):
        mind = self._mind()
        mind.turn("彩虹为什么是弯的", now=1000.0)
        first = mind.turn("再想想", now=1001.0)
        data = mind.snapshot()
        revived = MindEngine.load_from_dict(
            data, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False)
        second = revived.turn("再想想", now=1002.0)
        self.assertNotEqual(second.retrieval.get("reflection"),
                            first.retrieval.get("reflection"))
        self.assertNotEqual(second.base_reply, first.base_reply)


class TestReflectionPhrases(unittest.TestCase):
    def _mind(self, language="zh"):
        return MindEngine(
            persona=get_persona("generic"), store_path=None,
            clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
            kb=real_kb(), auto_save=False, language=language)

    def test_bare_nudges_are_reflection(self):
        mind = self._mind()
        for text in ("再想想", "再想一遍", "换个角度", "还有呢", "深入一点",
                     "再展开说说"):
            self.assertTrue(mind._is_reflection(text), text)

    def test_questions_with_a_subject_are_not_reflection(self):
        mind = self._mind()
        for text in ("熵是什么", "为什么天空是蓝的", "彩虹为什么是弯的",
                     "再想想熵和温度的关系"):
            self.assertFalse(mind._is_reflection(text), text)

    def test_english_nudges_are_reflection(self):
        mind = self._mind(language="en")
        for text in ("think again", "another angle", "what else",
                     "go deeper"):
            self.assertTrue(mind._is_reflection(text), text)
        self.assertFalse(mind._is_reflection("why is the sky blue"))


if __name__ == "__main__":
    unittest.main()
