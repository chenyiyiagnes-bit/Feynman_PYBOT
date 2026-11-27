"""Per-turn review: every failed turn is written down, nothing is promoted.

The trail is the honest list of what the cards could not answer. One odd
question stays a line; the second sighting becomes a review candidate, and
even that waits for a human decision.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import review, reviewer
from omega_mind.learner import LearnedStore


def store(tmp: str) -> LearnedStore:
    return LearnedStore(Path(tmp) / "learned")


class TestJudge(unittest.TestCase):
    def test_fallback_is_an_abstain(self):
        self.assertEqual(reviewer.judge("fallback:zh", {}), {"reason": "abstain"})
        self.assertEqual(reviewer.judge("pkg:no_match", {}), {"reason": "abstain"})

    def test_a_soft_hit_is_weak(self):
        self.assertEqual(reviewer.judge("pkg_soft:card-1", {}), {"reason": "weak"})

    def test_a_low_score_is_weak(self):
        self.assertEqual(
            reviewer.judge("pkg:card-1", {"score": 0.1}, 0.3, 0.05),
            {"reason": "weak"})

    def test_a_near_tie_is_weak(self):
        self.assertEqual(
            reviewer.judge("pkg:card-1", {"score": 0.9, "margin": 0.01}, 0.3, 0.05),
            {"reason": "weak"})

    def test_an_ungrounded_followup_is_flagged(self):
        self.assertEqual(
            reviewer.judge("pkg:card-1",
                           {"unverified_followup": True}, 0.3, 0.05),
            {"reason": "unverified_followup"})

    def test_a_confident_answer_is_not_a_finding(self):
        self.assertIsNone(
            reviewer.judge("pkg:card-1", {"score": 0.9, "margin": 0.4}, 0.3, 0.05))

    def test_sources_without_a_card_are_not_cards(self):
        self.assertIsNone(reviewer.judge("provided", {}, 0.3, 0.05))
        self.assertIsNone(reviewer.judge("ack:zh", {}, 0.3, 0.05))
        self.assertIsNone(reviewer.judge("memory:update", {}, 0.3, 0.05))


class TestTrail(unittest.TestCase):
    def test_each_record_is_one_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            reviewer.record(s, {"text": "熵是什么", "reason": "abstain", "turn": 1})
            reviewer.record(s, {"text": "熵是什么", "reason": "abstain", "turn": 2})
            lines = reviewer.path_for(s).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(reviewer.count_of(s, "熵是什么"), 2)

    def test_the_trail_survives_a_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            reviewer.record(s, {"text": "量子隧穿", "reason": "weak", "turn": 3})
            again = store(tmp)
            self.assertEqual(len(reviewer.events(again)), 1)

    def test_summary_groups_by_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            for turn in (1, 2, 3):
                reviewer.record(s, {"text": "同一问", "reason": "abstain", "turn": turn})
            reviewer.record(s, {"text": "另一问", "reason": "weak", "turn": 4})
            rows = reviewer.summary(s)
            self.assertEqual(rows[0]["text"], "同一问")
            self.assertEqual(rows[0]["count"], 3)
            self.assertIn("abstain", rows[0]["reasons"])


class TestPromotion(unittest.TestCase):
    def test_one_sighting_queues_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            reviewer.record(s, {"text": "只问一次", "reason": "abstain", "turn": 1})
            self.assertEqual(reviewer.queue_repeats(s), [])
            self.assertEqual(review.read(s)["candidates"], {})

    def test_the_second_sighting_queues_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            reviewer.record(s, {"text": "问两次", "reason": "abstain", "turn": 1,
                                "language": "zh"})
            reviewer.record(s, {"text": "问两次", "reason": "abstain", "turn": 2})
            queued = reviewer.queue_repeats(s)
            self.assertEqual(len(queued), 1)
            candidate = review.read(s)["candidates"][queued[0]]
            self.assertEqual(candidate["kind"], "gap")
            self.assertEqual(candidate["question"], "问两次")
            self.assertEqual(candidate["seen"], 2)

    def test_a_candidate_is_not_a_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = store(tmp)
            for turn in (1, 2):
                reviewer.record(s, {"text": "重复问", "reason": "weak", "turn": turn})
            reviewer.queue_repeats(s)
            data = review.read(s)
            self.assertEqual(data["decisions"], [])
            self.assertEqual(review.active(data), [])


class TestEngineWritesTheTrail(unittest.TestCase):
    def _pair(self, tmp: str):
        base = Path(tmp)
        knowledge = base / "knowledge"
        knowledge.mkdir()
        (knowledge / "cards.jsonl").write_text(
            json.dumps({"id": "c-1", "title": "熵",
                        "text": "熵是系统混乱程度的度量。", "tags": ["热力学"]},
                       ensure_ascii=False) + "\n", encoding="utf-8")
        learned = base / "learned"
        return knowledge, learned

    def _engine(self, knowledge: Path, learned: Path):
        from omega_mind.config import get_persona
        from omega_mind.engine import MindEngine
        from omega_mind.feynman_kb import FeynmanKB
        from omega_mind.pybot_bridge import PyBotBridge

        kb = FeynmanKB(knowledge_dir=str(knowledge), learned_dir=str(learned))
        return MindEngine(persona=get_persona("generic"), store_path=None,
                          clock=lambda: 1000.0, bridge=PyBotBridge.disabled(),
                          kb=kb, auto_save=False, language="zh")

    def test_an_unanswerable_turn_is_written_at_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            knowledge, learned = self._pair(tmp)
            mind = self._engine(knowledge, learned)
            mind.turn("今天股票怎么买？", now=1000.0)
            rows = reviewer.events(mind.kb.learned)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reason"], "abstain")
            self.assertEqual(rows[0]["text"], "今天股票怎么买？")

    def test_a_confident_answer_leaves_no_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            knowledge, learned = self._pair(tmp)
            mind = self._engine(knowledge, learned)
            reply = mind.turn("熵是什么", now=1000.0)
            if not reply.source.startswith(mind.kb.source_prefix):
                self.skipTest(f"the tiny shelf did not answer: {reply.source}")
            self.assertEqual(reviewer.events(mind.kb.learned), [])

    def test_the_same_miss_twice_becomes_a_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            knowledge, learned = self._pair(tmp)
            mind = self._engine(knowledge, learned)
            mind.turn("脉冲星是什么？", now=1000.0)
            mind.turn("脉冲星是什么？", now=1001.0)
            candidates = review.read(mind.kb.learned)["candidates"]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(next(iter(candidates.values()))["question"], "脉冲星是什么？")


if __name__ == "__main__":
    unittest.main()
