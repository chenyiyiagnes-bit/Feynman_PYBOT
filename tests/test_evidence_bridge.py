"""The left bridge may lay out selected evidence, without editing claims."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from app.runtime import build
from bridge.responder import LeftResponder
from omega_mind import assoc, evidence
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.pybot_bridge import PyBotBridge


ZH_OPEN = "量子纠缠可以发送超光速消息吗？"
ZH_WHY = "那为什么一定得交换经典信息？"


class TestEvidenceBreaks(unittest.TestCase):
    def test_protects_markup_and_existing_layout(self):
        for text, language in (
            ("第一段。\n第二段。另留一段。", "zh"),
            ("公式 $x = 0.2$。有两项。再检查。", "zh"),
            ("读原句：“来。再走。然后回答。最后核对。", "zh"),
        ):
            with self.subTest(text=text):
                self.assertEqual(evidence.candidates(text, language), ())

    def test_can_pause_outside_a_quote_or_parenthesis(self):
        text = "读原句：“来。再走。” 然后回答。最后核对。"
        self.assertEqual(evidence.candidates(text, "zh"), (text.index("最后"),))
        text = "Compare (See A. And B) before the final check. Keep the order."
        self.assertEqual(evidence.candidates(text, "en"), (text.index(" Keep"),))

    def test_keeps_decimal_abbreviation_numbers_and_negation(self):
        text = "Dr. Lin checks 2.5 V. The result is not above 3.0 volts. Repeat the test."
        allowed = evidence.candidates(text, "en")
        self.assertEqual(allowed, (text.index(" Repeat"),))
        result = evidence.apply_breaks(text, allowed, allowed)
        self.assertEqual(result.replace("\n", " "), text)

    def test_rejects_invalid_duplicate_or_unordered_offsets(self):
        text = "这是第一句。第二句也是证据。最后一句在这里。"
        allowed = evidence.candidates(text, "zh")
        for plan in ("其他答案", (0,), (True,), (3,), allowed[::-1],
                     (allowed[0], allowed[0]), (allowed[0], len(text))):
            with self.subTest(plan=plan):
                self.assertIsNone(evidence.apply_breaks(text, allowed, plan))


class TestEvidenceBridge(unittest.TestCase):
    def test_followups_use_selected_evidence_without_a_vault(self):
        dialogues = [
            ("feynman-zh", ZH_OPEN, ZH_WHY, ""),
            ("feynman-en", "Can quantum entanglement send a message faster than light?",
             "So why do they need classical information?", " "),
        ]
        with TemporaryDirectory() as tmp, patch.dict("os.environ", {
            "QBOT_FEYNMAN_VAULT": str(Path(tmp) / "absent.json"),
            "PYBOT_STATE_DIR": tmp,
        }):
            for package, question, followup, joiner in dialogues:
                with self.subTest(package=package):
                    plain = build(package, enable_bridge=False, clock=lambda: 1000.0)
                    formatted = build(package, enable_bridge=True, clock=lambda: 1000.0)
                    self.assertTrue(formatted.left_responder.evidence_available())
                    with patch.object(formatted.left_responder, "_load_modules",
                                      side_effect=AssertionError("opened a vault path")):
                        plain.turn(question)
                        formatted.turn(question)
                        before = plain.turn(followup)
                        after = formatted.turn(followup)
                    self.assertEqual(after.base_reply, before.base_reply)
                    self.assertEqual(after.source, before.source)
                    self.assertEqual(after.retrieval["evidence"], before.retrieval["evidence"])
                    self.assertEqual(joiner.join(after.reply.splitlines()), before.reply)
                    self.assertTrue(after.retrieval["evidence_layout"])

    def test_long_answer_keeps_its_source_and_repeats_literal_output(self):
        question = "为什么天是蓝的？"
        formatted = build("feynman-zh", enable_bridge=True, clock=lambda: 1000.0)
        result = formatted.turn(question)
        self.assertGreater(len(result.reply.splitlines()), 1)
        self.assertEqual("".join(result.reply.splitlines()), result.base_reply)
        self.assertEqual(result.source, "feynman-zh:opt-012")
        repeated = formatted.turn("再说一遍")
        self.assertEqual(repeated.reply, result.reply)
        self.assertNotIn("evidence_layout", repeated.retrieval)

    def test_reviewed_analogy_can_pause_outside_the_quoted_title(self):
        cards = [
            {"id": "cloth", "title": "布条吸水", "text": "把布条的一端放进水里。"},
            {"id": "paper", "title": "纸巾吸水", "text": "水会沿纸巾的缝隙往前走。"},
        ]
        reviewed = assoc.Edge(
            "cloth", "paper", "same_mechanism",
            mechanism="液体沿纤维之间的缝隙继续往前走",
            difference="布料与纸巾的纤维排列不同，吸水的速度可以不同",
            mechanism_en="liquid travels through gaps between fibres",
            difference_en="the materials have different fibre arrangements")
        with TemporaryDirectory() as tmp:
            directory = Path(tmp) / "knowledge"
            directory.mkdir()
            (directory / "cards.jsonl").write_text(
                "".join(json.dumps(card, ensure_ascii=False) + "\n" for card in cards),
                encoding="utf-8")
            edge_path = Path(tmp) / "edges.jsonl"
            edge_path.write_text(json.dumps(reviewed.to_row()), encoding="utf-8")
            kb = FeynmanKB(knowledge_dir=str(directory), source_prefix="sample-zh",
                           assoc_path=str(edge_path))
            mind = MindEngine(kb=kb, bridge=PyBotBridge.disabled(),
                              left_responder=LeftResponder(enabled=True), clock=lambda: 1000.0)
            mind.state.set("energy", 0.20)
            mind.turn("布条吸水")
            mind.state.set("energy", mind.persona.baseline("energy"))
            answer = mind.turn("没懂")
        self.assertEqual(answer.retrieval["dialogue_act"], "rephrase")
        self.assertEqual(answer.source, "sample-zh:cloth")
        self.assertIn("《纸巾吸水》", answer.reply)
        self.assertIn("\n差别是", answer.reply)
        self.assertEqual("".join(answer.reply.splitlines()), answer.base_reply)

    def test_misses_and_supplied_replies_bypass_the_bridge(self):
        mind = build("feynman-zh", enable_bridge=True, clock=lambda: 1000.0)
        mind.left_responder = Mock()
        self.assertEqual(mind.turn("为什么装满热水的玻璃杯有时会裂开？").source, "fallback:zh")
        mind.turn("选哪一个", base_reply="先告诉我你指的是哪个。然后我们再比较。")
        mind.left_responder.plan_evidence.assert_not_called()
        mind.left_responder.reply.assert_not_called()

    def test_invalid_or_failed_bridge_preserves_the_entire_turn(self):
        plain = build("feynman-zh", enable_bridge=False, clock=lambda: 1000.0)
        plain.turn(ZH_OPEN)
        expected = plain.turn(ZH_WHY).to_dict()
        for planner in (Mock(return_value=(1,)), Mock(side_effect=RuntimeError)):
            with self.subTest(planner=planner):
                mind = build("feynman-zh", enable_bridge=False, clock=lambda: 1000.0)
                mind.turn(ZH_OPEN)
                mind.left_responder = Mock(plan_evidence=planner)
                self.assertEqual(mind.turn(ZH_WHY).to_dict(), expected)
                planner.assert_called_once()

    def test_disabled_bridge_has_the_same_output_as_no_responder(self):
        plain = build("feynman-zh", enable_bridge=False, clock=lambda: 1000.0)
        disabled = build("feynman-zh", enable_bridge=False, clock=lambda: 1000.0)
        disabled.left_responder = LeftResponder(enabled=False)
        for question in (ZH_OPEN, ZH_WHY):
            self.assertEqual(disabled.turn(question).to_dict(), plain.turn(question).to_dict())


if __name__ == "__main__":
    unittest.main()
