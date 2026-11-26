"""End-to-end regressions for grounded follow-ups and response routing."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omega_mind import assoc, lang, left_brain
from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import Doc, FeynmanKB, Lookup
from omega_mind.pybot_bridge import PyBotBridge


class TestDialogueFlow(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def mind(self, cards, language="zh", profile=None, edges=(), responder=None):
        directory = self.root / "knowledge"
        directory.mkdir(exist_ok=True)
        (self.root / "persona.json").write_text(json.dumps({
            "id": self.root.name, "language": language,
        }), encoding="utf-8")
        (directory / "cards.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cards),
            encoding="utf-8")
        edge_path = self.root / "edges.jsonl"
        edge_path.write_text(
            "".join(json.dumps(edge.to_row(), ensure_ascii=False) + "\n"
                    for edge in edges), encoding="utf-8")
        kb = FeynmanKB(knowledge_dir=str(directory), language=language,
                       profile=profile, assoc_path=str(edge_path))
        return MindEngine(persona=get_persona("generic"),
                          kb=kb, language=language, left_responder=responder,
                          bridge=PyBotBridge.disabled(), clock=lambda: 1000.0)

    def test_card_miss_never_routes_to_the_typesetter_or_generic_facts(self):
        cards = [{"id": "ice", "title": "冰块", "text": "冰块会融化。"}]
        responder = Mock()
        mind = self.mind(cards, responder=responder)
        for query in ("为什么热水会把玻璃杯弄裂？", "如果没人看，电子就不存在了吗？"):
            with self.subTest(query=query):
                answer = mind.turn(query)
                self.assertEqual(answer.source, "fallback:zh")
                self.assertEqual(answer.reply, answer.base_reply)
        responder.reply.assert_not_called()

    def test_politeness_and_definition_keep_the_classical_term(self):
        cards = [{"id": "wu", "title": "无为", "tags": ["无为", "顺其自然"],
                  "text": "所谓无为，是在恰当的时候出手。"}]
        mind = self.mind(cards, profile="classical")
        for query in ("无为", "无为是什么意思？", "请问，无为是什么意思？"):
            with self.subTest(query=query):
                self.assertTrue(mind.turn(query).source.endswith(":wu"))
        core = lang.load("zh", "classical")
        self.assertEqual(core.query_text("请问，无为和躺平有什么区别？"),
                         "无为和躺平有什么区别？")

    def test_targeted_why_selects_a_passage_and_repeat_preserves_it(self):
        reason = "样品扩散是因为分子始终在运动。"
        mind = self.mind([{"id": "a", "title": "样品扩散",
                           "text": reason + "这里讨论的是一小管样品。"}])
        first = mind.turn("样品扩散")
        self.assertTrue(first.source.endswith(":a"))
        mind.turn("嗯")
        follow = mind.turn("那为什么样品会扩散？")
        self.assertEqual(follow.retrieval["dialogue_act"], "why")
        self.assertEqual(follow.reply, reason)
        self.assertEqual(follow.source, first.source)
        self.assertEqual(mind.turn("再说一遍").reply, reason)

    def test_followup_does_not_borrow_another_users_topic(self):
        mind = self.mind([{"id": "a", "title": "样品扩散",
                           "text": "样品扩散是因为分子在运动。"}])
        mind.turn("样品扩散", user="alice")
        follow = mind.turn("为什么？", user="bob")
        self.assertEqual(follow.source, "fallback:zh")
        self.assertNotIn("dialogue_act", follow.retrieval)

    def test_new_topic_can_take_over_the_followup(self):
        mind = self.mind([
            {"id": "a", "title": "样品扩散", "text": "样品扩散是因为分子在运动。"},
            {"id": "b", "title": "颜色变化",
             "text": "每次试验都有一张照片。颜色变化是因为入射光变了。"},
        ])
        mind.turn("样品扩散")
        switched = mind.turn("换个话题，颜色变化")
        self.assertTrue(switched.source.endswith(":b"))
        self.assertIn("入射光", mind.turn("为什么？").reply)

    def test_english_rephrase_uses_confirmed_evidence_once(self):
        cards = [
            {"id": "a", "title": "Crystal growth", "text": "Crystals grow in layers."},
            {"id": "b", "title": "Brick wall", "text": "A brick wall grows in layers."},
        ]
        reviewed = assoc.Edge(
            "a", "b", "same_mechanism", mechanism="层层叠加", difference="材料不同",
            mechanism_en="each layer builds on the last",
            difference_en="molecules are smaller than bricks")
        mind = self.mind(cards, "en", edges=[reviewed])
        mind.state.set("energy", 0.20)
        first = mind.turn("Crystal growth")
        self.assertTrue(first.source.endswith(":a"))
        recast = mind.turn("I don't understand.")
        self.assertEqual(recast.retrieval["dialogue_act"], "rephrase")
        self.assertIn("each layer builds on the last", recast.reply)
        self.assertEqual(mind.turn("Say that again.").reply, recast.reply)
        missed = mind.turn("I still don't understand.")
        self.assertTrue(missed.source.startswith("fallback"))
        self.assertTrue(missed.reply.endswith("?"))

    def test_targeted_why_can_take_one_confirmed_premise_hop(self):
        cards = [
            {"id": "a", "title": "湿布变干", "text": "湿布晾一会儿就干了。"},
            {"id": "b", "title": "水分蒸发", "text": "水分进入了空气。"},
        ]
        edge = assoc.Edge("a", "b", "premise",
                          mechanism="湿布变干时水分蒸发",
                          mechanism_en="the water evaporates as the cloth dries")
        mind = self.mind(cards, edges=[edge])
        mind.state.set("energy", 0.20)
        mind.turn("湿布变干")
        answer = mind.turn("那为什么湿布变干？")
        self.assertEqual(answer.retrieval["dialogue_act"], "why")
        self.assertIn("水分蒸发", answer.reply)

    def test_conditional_uses_the_relevant_part_of_the_example(self):
        mind = self.mind([{"id": "a", "title": "照料幼苗",
                           "text": "先看它是否需要帮助。比如照料一株苗：土干了就浇水。"}])
        mind.turn("照料幼苗")
        answer = mind.turn("那如果苗快干死了，还应该等吗？")
        self.assertEqual(answer.retrieval["dialogue_act"], "boundary")
        self.assertEqual(answer.reply, "土干了就浇水。")

    def test_shared_characters_do_not_attach_an_unrelated_condition(self):
        mind = self.mind([{"id": "a", "title": "照料幼苗",
                           "text": "比如照料一株苗：土干了就浇水。"}])
        mind.turn("照料幼苗")
        answer = mind.turn("那如果海水干了，全球气候会怎么样？")
        self.assertEqual(answer.source, "fallback:zh")
        self.assertNotIn("浇水", answer.reply)

        mind.turn("照料幼苗")
        near_miss = Lookup("河水从高处流向低处。", mind.kb.source_root + "_soft:b")
        with patch.object(mind.kb, "lookup", return_value=near_miss):
            answer = mind.turn("那如果海水干了，全球气候会怎么样？")
        self.assertEqual(answer.source, "fallback:zh")
        self.assertTrue(answer.retrieval["unverified_followup"])

    def test_greeting_and_clarification_keep_question_punctuation(self):
        mind = self.mind([])
        self.assertEqual(mind.turn("你好").reply, "我在。想聊点什么？")
        clarify = "你问的是左边还是右边？"
        self.assertEqual(mind.turn("那里呢", base_reply=clarify).reply, clarify)

    def test_soft_match_compares_a_runner_up_even_at_limit_one(self):
        mind = self.mind([])
        first = Doc("a", "fact", "Alpha", "The first candidate.")
        second = Doc("b", "fact", "Beta", "The second candidate.")
        mind.kb.search = Mock(side_effect=lambda *args, limit, **kw:
                              [(0.27, first), (0.26, second)][:limit])
        normal = mind._base_reply("What happened?", left_brain.LeftBrainPolicy(limit=3))
        brief = mind._base_reply(
            "What happened?", left_brain.LeftBrainPolicy(limit=1, detail="brief"))
        self.assertEqual(normal, brief)
        self.assertEqual(brief[1], "fallback:zh")
        self.assertEqual(mind.last_retrieval["considered"], 2)


if __name__ == "__main__":
    unittest.main()
