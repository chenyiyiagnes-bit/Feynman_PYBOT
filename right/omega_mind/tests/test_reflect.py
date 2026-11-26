"""Dream-time self-audit: only repetition earns a bounded, gated repair.

Every test writes into a throwaway learned directory. The real corpus is only
read, never modified.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import reflect
from omega_mind.dream import Dreamer
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.learner import LearnedStore, normalize
from omega_mind.memory import Episode, MemoryTiers, _episode_from

TEMP = tempfile.mkdtemp(prefix="omega-reflect-")
# A question the corpus genuinely cannot answer. It must stay outside the
# golden net: a negative there would (correctly) block any repair that makes
# it answerable, and every test below writes something for it.
MISS = "为什么猫半夜总是跑来跑去"
MISS_VARIANT = "为什么猫半夜总是跑来跑去呢"


def real_kb() -> FeynmanKB:
    kb = FeynmanKB(learned_dir=TEMP)
    if not kb.available():
        raise unittest.SkipTest("feynman-open knowledge not on this machine")
    return kb


def miss_episode(turn: int, text: str = MISS,
                 alternatives=()) -> Episode:
    return Episode(
        turn=turn, now=1000.0 + turn, user="default", text=text,
        reply="", source="feynman_kb:no_match", tags=[],
        retrieval={"source": "feynman_kb:no_match", "score": 0.0,
                   "alternatives": [list(a) for a in alternatives]},
    )


def gap_cards(store: LearnedStore) -> list:
    return [c for c in store.cards if c.get("origin") == "dream_gap"]


class TestRepeatRule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def setUp(self):
        self.store = LearnedStore(Path(tempfile.mkdtemp(prefix="omega-reflect-")))

    def test_one_miss_is_recorded_but_not_acted_on(self):
        report = reflect.audit(self.kb, self.store,
                               [miss_episode(1)], cortisol=0.0)
        self.assertEqual(report.findings[0].count, 1)
        self.assertEqual(report.repaired, 0)
        self.assertEqual(gap_cards(self.store), [])
        history_file = self.store.dir / reflect.HISTORY_FILE
        self.assertTrue(history_file.exists())
        data = json.loads(history_file.read_text(encoding="utf-8"))
        self.assertIn(normalize(MISS), data["misses"])

    def test_second_miss_mints_exactly_one_gap_card(self):
        reflect.audit(self.kb, self.store, [miss_episode(1)], cortisol=0.0)
        report = reflect.audit(self.kb, self.store, [miss_episode(2)],
                               cortisol=0.0)
        self.assertEqual(report.findings[0].count, 2)
        cards = gap_cards(self.store)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["title"], MISS)
        self.assertIn(MISS, cards[0]["aliases"])
        self.assertTrue(any(a.applied and a.kind == "gap_card"
                            for a in report.actions))

    def test_third_night_does_not_duplicate_the_gap_card(self):
        for turn in (1, 2, 3):
            reflect.audit(self.kb, self.store, [miss_episode(turn)],
                          cortisol=0.0)
        self.assertEqual(len(gap_cards(self.store)), 1)

    def test_being_answered_by_a_gap_card_still_counts_as_unanswered(self):
        for turn in (1, 2):
            reflect.audit(self.kb, self.store, [miss_episode(turn)],
                          cortisol=0.0)
        card = gap_cards(self.store)[0]
        again = miss_episode(3)
        again.source = f"feynman_kb:{card['id']}"
        report = reflect.audit(self.kb, self.store, [again], cortisol=0.0)
        self.assertEqual(report.findings[0].count, 3)
        self.assertEqual(report.findings[0].kind, "unanswered")

    def test_near_identical_wording_shares_one_counter(self):
        reflect.audit(self.kb, self.store, [miss_episode(1)], cortisol=0.0)
        report = reflect.audit(
            self.kb, self.store,
            [miss_episode(2, MISS_VARIANT)], cortisol=0.0)
        self.assertEqual(report.findings[0].count, 2)
        self.assertEqual(len(gap_cards(self.store)), 1)


class TestSafetyGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def setUp(self):
        self.store = LearnedStore(Path(tempfile.mkdtemp(prefix="omega-reflect-")))

    def test_high_cortisol_is_read_only(self):
        reflect.audit(self.kb, self.store, [miss_episode(1)], cortisol=0.0)
        report = reflect.audit(self.kb, self.store, [miss_episode(2)],
                               cortisol=0.9)
        self.assertTrue(report.read_only)
        self.assertEqual(report.repaired, 0)
        self.assertEqual(gap_cards(self.store), [])

    def test_failed_golden_gate_blocks_every_write(self):
        reflect.audit(self.kb, self.store, [miss_episode(1)], cortisol=0.0)
        with mock.patch.object(reflect, "golden_ok",
                               return_value=(False, ["fake regression"])):
            report = reflect.audit(self.kb, self.store,
                                   [miss_episode(2)], cortisol=0.0)
        self.assertFalse(report.gate_ok)
        self.assertEqual(report.repaired, 0)
        self.assertEqual(gap_cards(self.store), [])

    def test_alias_is_refused_when_gate_fails(self):
        with mock.patch.object(reflect, "golden_ok",
                               return_value=(False, ["fake regression"])):
            action = reflect._try_alias(
                self.kb, self.store, "扩散是怎么回事呢",
                [["C_DIFFUSE", 0.26]])
        self.assertFalse(action.applied)
        self.assertIn("golden gate", action.reason)
        self.assertEqual(self.store.aliases.get("C_DIFFUSE", []), [])

    def test_alias_needs_a_close_card(self):
        action = reflect._try_alias(self.kb, self.store, "彩虹是怎么回事呢",
                                    [["C_DIFFUSE", 0.05]])
        self.assertFalse(action.applied)
        self.assertIn("below", action.reason)

    def test_alias_applies_and_persists_when_the_gate_passes(self):
        directory = tempfile.mkdtemp(prefix="omega-reflect-")
        kb = FeynmanKB(learned_dir=directory)
        store = LearnedStore(Path(directory))
        action = reflect._try_alias(kb, store, "阻尼到底怎么理解",
                                    [["C_DAMP", 0.2506]])
        self.assertTrue(action.applied, action.reason)
        _, source = kb.answer("阻尼到底怎么理解")
        self.assertEqual(source, "feynman_kb:C_DAMP")
        self.assertIn("阻尼到底怎么理解", kb.learned.aliases.get("C_DAMP", []))

    def test_queue_is_written_every_night(self):
        reflect.audit(self.kb, self.store, [miss_episode(1)], cortisol=0.0)
        queue = reflect.load_queue(self.store)
        self.assertEqual(queue[0]["question"], MISS)
        self.assertEqual(queue[0]["kind"], "unanswered")
        self.assertEqual(queue[0]["count"], 1)


class TestBrokenFollowUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_an_explicit_correction_is_stored_the_first_time(self):
        directory = tempfile.mkdtemp(prefix="omega-reflect-")
        kb = FeynmanKB(learned_dir=directory)
        store = kb.learned
        first = miss_episode(1)
        correction = Episode(
            turn=2, now=1002.0, user="default",
            text="不对，其实它是听见了楼下的动静，想去看看。",
            reply="", source="fallback:chat", tags=[])
        report = reflect.audit(kb, store, [first, correction], cortisol=0.0)
        kinds = {f.kind for f in report.findings}
        self.assertIn("broken_followup", kinds)
        self.assertEqual(report.repaired, 1)
        card = next(c for c in store.cards
                    if c.get("origin") == "dream_correction")
        self.assertEqual(card["verification_status"], "user_claim")
        self.assertIn(MISS, card["aliases"])
        _, source = kb.answer(MISS)
        self.assertEqual(source, f"feynman_kb:{card['id']}")

    def test_a_short_correction_is_not_stored(self):
        store = LearnedStore(Path(tempfile.mkdtemp(prefix="omega-reflect-")))
        first = miss_episode(1)
        correction = Episode(turn=2, now=1002.0, user="default",
                             text="不对", reply="", source="fallback:chat",
                             tags=[])
        report = reflect.audit(self.kb, store, [first, correction], cortisol=0.0)
        self.assertEqual(
            [c for c in store.cards if c.get("origin") == "dream_correction"],
            [])
        self.assertEqual(report.repaired, 0)

    def test_high_cortisol_does_not_store_a_correction(self):
        store = LearnedStore(Path(tempfile.mkdtemp(prefix="omega-reflect-")))
        first = miss_episode(1)
        correction = Episode(
            turn=2, now=1002.0, user="default",
            text="不对，其实它是听见了楼下的动静，想去看看。",
            reply="", source="fallback:chat", tags=[])
        report = reflect.audit(self.kb, store, [first, correction],
                               cortisol=0.9)
        self.assertTrue(report.read_only)
        self.assertEqual(
            [c for c in store.cards if c.get("origin") == "dream_correction"],
            [])


class TestContextBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="omega-reflect-")
        self.kb = FeynmanKB(learned_dir=self.directory)
        self.store = LearnedStore(Path(self.directory))

    def _hit(self, turn: int, text: str, card_id: str) -> Episode:
        return Episode(
            turn=turn, now=1000.0 + turn, user="default", text=text,
            reply="", source=f"feynman_kb:{card_id}", tags=[],
            retrieval={"score": 0.5, "alternatives": [[card_id, 0.5]]})

    def test_a_follow_up_miss_bridges_to_the_previous_answer(self):
        prev = self._hit(1, "什么是阻尼", "C_DAMP")
        follow = miss_episode(2, "那阻尼和摩擦是什么关系呢")
        report = reflect.audit(self.kb, self.store, [prev, follow],
                               cortisol=0.0)
        applied = [a for a in report.actions
                   if a.kind == "bridge_alias" and a.applied]
        self.assertTrue(applied, report.to_dict())
        self.assertEqual(applied[0].target, "C_DAMP")
        self.assertIn("那阻尼和摩擦是什么关系呢",
                      self.kb.learned.aliases.get("C_DAMP", []))

    def test_bridge_refused_when_the_previous_turn_was_not_a_hit(self):
        prev = miss_episode(1)
        follow = miss_episode(2, "那它和摩擦是什么关系呢")
        report = reflect.audit(self.kb, self.store, [prev, follow],
                               cortisol=0.0)
        self.assertFalse(any(a.kind == "bridge_alias" and a.applied
                             for a in report.actions))

    def test_bridge_works_across_the_night_boundary(self):
        prior = self._hit(1, "什么是阻尼", "C_DAMP")
        follow = miss_episode(2, "那阻尼和摩擦是什么关系呢")
        report = reflect.audit(self.kb, self.store, [follow], cortisol=0.0,
                               prior=prior)
        self.assertTrue(any(a.kind == "bridge_alias" and a.applied
                            for a in report.actions), report.to_dict())


class TestWeakAnswer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_near_tie_is_reported_even_when_answered(self):
        store = LearnedStore(Path(tempfile.mkdtemp(prefix="omega-reflect-")))
        episode = Episode(
            turn=1, now=1001.0, user="default", text="什么是阻尼",
            reply="……", source="feynman_kb_soft:U0_DAMP", tags=[],
            retrieval={"score": 0.28, "margin": 0.01,
                       "alternatives": [["U0_DAMP", 0.28], ["C_DAMP", 0.27]]})
        report = reflect.audit(self.kb, store, [episode], cortisol=0.0)
        self.assertEqual(report.findings[0].kind, "weak")
        self.assertEqual(report.findings[0].count, 1)


class TestEngineIntegration(unittest.TestCase):
    def test_dream_runs_the_audit_and_stores_it(self):
        kb = real_kb()
        engine = MindEngine(kb=kb, auto_save=False)
        for turn in (1, 2):
            engine.tiers.record(miss_episode(turn))
        dream = engine.dream(now=2000.0, force=True)
        self.assertIsNotNone(dream)
        self.assertIn("findings", dream.audit)
        self.assertEqual(dream.audit["repaired"], 1)
        self.assertTrue(dream.to_dict()["audit"]["actions"][0]["applied"])
        # The event carries the same report, so the log alone can replay it.
        event = engine.event_log[-1]
        self.assertEqual(event["audit"]["repaired"], 1)
        self.assertTrue(gap_cards(kb.learned))


class TestOldSaves(unittest.TestCase):
    def test_episode_without_retrieval_loads(self):
        episode = _episode_from({"turn": 1, "now": 1.0, "user": "u",
                                 "text": "t", "reply": "r", "source": "s"})
        self.assertIsNotNone(episode)
        self.assertEqual(episode.retrieval, {})

    def test_dream_without_audit_loads(self):
        dreamer = Dreamer()
        dreamer.restore({"dreams": [{"index": 1, "now": 1.0, "episodes": 3}]})
        self.assertEqual(dreamer.dreams[0].audit, {})


class TestSurpriseTrigger(unittest.TestCase):
    def _memory(self, episodes):
        tiers = MemoryTiers()
        for episode in episodes:
            tiers.record(episode)
        return tiers

    def _normal(self, turn):
        return Episode(turn=turn, now=1000.0 + turn, user="default",
                       text="今天天气不错", reply="嗯", source="chat", tags=[])

    def test_eight_ordinary_turns_still_make_a_dream(self):
        episodes = [self._normal(i) for i in range(1, 9)]
        self.assertTrue(Dreamer().due(self._memory(episodes)))
        self.assertFalse(Dreamer().due(self._memory(episodes[:-1])))

    def test_two_misses_alone_still_wait(self):
        episodes = [miss_episode(1), miss_episode(2)]
        self.assertFalse(Dreamer().due(self._memory(episodes)))

    def test_three_misses_plus_a_correction_get_to_sleep_early(self):
        correction = Episode(
            turn=4, now=1004.0, user="default",
            text="不对，其实它是听见了楼下的动静，想去看看。",
            reply="", source="feynman_kb:no_match", tags=[])
        episodes = [miss_episode(1), miss_episode(2), miss_episode(3),
                    correction]
        self.assertTrue(Dreamer().due(self._memory(episodes)))

    def test_a_single_correction_is_not_enough(self):
        correction = Episode(
            turn=1, now=1001.0, user="default",
            text="不对，其实它是听见了楼下的动静，想去看看。",
            reply="", source="feynman_kb:no_match", tags=[])
        self.assertFalse(Dreamer().due(self._memory([correction])))


if __name__ == "__main__":
    unittest.main()
