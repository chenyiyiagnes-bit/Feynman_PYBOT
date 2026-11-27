"""Experience that survives a restart: learned cards, aliases, soft recall.

Everything here writes into a temporary learned directory — the shipped
corpus and the repository's own learned files are never touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind import learner
from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import FeynmanKB
from omega_mind.learner import (
    EVENTS_FILE,
    MAX_ALIASES_PER_CARD,
    LearnedStore,
    card_id_for,
    card_row,
    explicit_confirmation,
    looks_like_metaphor,
    metaphor_candidate,
    similarity,
)
from omega_mind.pybot_bridge import PyBotBridge


def temp_dir() -> str:
    return tempfile.mkdtemp()


class TestMetaphorGate(unittest.TestCase):
    def test_a_comparison_is_a_card(self):
        text = "区块链的哈希指针可以想成一串首尾相扣的蜡封信件，改一封就得重盖章。"
        row = card_row(text, origin="test")
        self.assertIsNotNone(row)
        self.assertEqual(row["type"], "learned_card")
        self.assertEqual(row["id"], card_id_for(text))
        self.assertIn("蜡封信件", row["text"])

    def test_questions_and_plain_facts_are_not_cards(self):
        for text in ("区块链是不是防篡改的？", "哈希指针是什么东西啊",
                     "嗯", "好"):
            with self.subTest(text=text):
                self.assertIsNone(card_row(text))
                self.assertFalse(looks_like_metaphor(text))

    def test_id_is_stable_under_case_and_punctuation(self):
        self.assertEqual(card_id_for("像水流下山坡！"),
                         card_id_for("像水流下山坡"))


class TestLearnedStore(unittest.TestCase):
    def test_card_round_trips_and_dedupes(self):
        directory = temp_dir()
        store = LearnedStore(Path(directory))
        row = card_row("神经网络的反向传播就像水流下山坡时修河道。")
        self.assertTrue(store.add_card(row))
        self.assertFalse(store.add_card(row))
        again = LearnedStore(Path(directory))
        self.assertEqual([r["id"] for r in again.cards], [row["id"]])

    def test_aliases_persist_and_are_capped(self):
        directory = temp_dir()
        store = LearnedStore(Path(directory))
        for i in range(MAX_ALIASES_PER_CARD + 3):
            store.add_alias("math-comp-001", f"第{i}种问法")
        again = LearnedStore(Path(directory))
        self.assertEqual(len(again.aliases["math-comp-001"]),
                         MAX_ALIASES_PER_CARD)
        self.assertFalse(store.add_alias("math-comp-001", "第0种问法"))

    def test_a_bad_alias_is_refused(self):
        store = LearnedStore(Path(temp_dir()))
        self.assertFalse(store.add_alias("math-comp-001", "?"))
        self.assertFalse(store.add_alias("math-comp-001", "好"))


class TestKnowledgeRetrievalLearns(unittest.TestCase):
    QUESTION = "用蜡封信件这个比喻怎么理解防篡改的数据链？"

    def test_a_learned_card_is_retrievable(self):
        directory = temp_dir()
        store = LearnedStore(Path(directory))
        store.add_card(card_row(
            "区块链的哈希指针可以想成一串首尾相扣的蜡封信件，改一封后面全得重盖章。"))
        kb = FeynmanKB(learned_dir=directory)
        reply, source = kb.answer(self.QUESTION)
        self.assertIsNotNone(reply)
        self.assertIn("learned-", source)
        self.assertIn("蜡封", reply)

    def test_without_the_learned_dir_the_card_is_absent(self):
        kb = FeynmanKB(learned_dir=temp_dir())
        _, source = kb.answer(self.QUESTION)
        self.assertNotIn("learned-", source)

    def test_an_alias_lifts_its_card(self):
        directory = temp_dir()
        store = LearnedStore(Path(directory))
        store.add_alias("math-comp-001", "小箭头怎么加在一起")
        kb = FeynmanKB(learned_dir=directory)
        hits = kb.search("小箭头怎么加在一起", limit=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1].id, "math-comp-001")

    def test_persist_false_touches_memory_only(self):
        directory = temp_dir()
        kb = FeynmanKB(learned_dir=directory)
        self.assertTrue(kb.learn_alias("math-comp-001", "那种小箭头", persist=False))
        self.assertIn("math-comp-001", kb.aliases)
        self.assertFalse((Path(directory) / "learned_aliases.json").exists())


class TestEngineLearnsFromPraise(unittest.TestCase):
    def _engine(self) -> MindEngine:
        return MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=temp_dir()),
            auto_save=False,
            language="zh",
        )

    def test_praise_after_a_card_answer_teaches_the_phrasing(self):
        engine = self._engine()
        first = engine.turn("为什么电脑芯片会发热", now=1000.0)
        self.assertIn("math-comp-004", first.source)
        engine.turn("太棒了哈哈，这个比喻真好玩", now=1020.0)
        self.assertIn("为什么电脑芯片会发热",
                      engine.kb.aliases.get("math-comp-004", []))

    def test_praise_after_a_non_card_answer_teaches_nothing(self):
        engine = self._engine()
        engine.turn("我今天心情不错", now=1000.0)
        engine.turn("哈哈，有趣", now=1020.0)
        self.assertEqual(engine.kb.aliases, {})

    def test_aliases_survive_save_and_load(self):
        engine = self._engine()
        engine.turn("为什么电脑芯片会发热", now=1000.0)
        engine.turn("太棒了哈哈，这个比喻真好玩", now=1020.0)
        clone = MindEngine.load_from_dict(
            engine.snapshot(), bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=temp_dir()), auto_save=False)
        self.assertIn("为什么电脑芯片会发热",
                      clone.kb.aliases.get("math-comp-004", []))


class TestEngineMintsCards(unittest.TestCase):
    METAPHOR = ("区块链的哈希指针可以想成一串首尾相扣的蜡封信件，"
                "改一封后面全得重盖章。")

    def _engine(self, directory: str | None = None) -> MindEngine:
        self.directory = directory or temp_dir()
        return MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=self.directory),
            auto_save=False,
            language="zh",
        )

    def test_a_taught_metaphor_becomes_a_card(self):
        engine = self._engine()
        engine.turn(self.METAPHOR, now=1000.0)
        rows = (Path(self.directory) / "learned_cards.jsonl").read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertIn("蜡封信件", rows[0])

    def test_the_new_card_answers_a_later_paraphrase(self):
        engine = self._engine()
        engine.turn(self.METAPHOR, now=1000.0)
        later = engine.turn("哈希指针为什么能防篡改？蜡封信件的比喻再讲讲",
                            now=1100.0)
        self.assertIn("learned-", later.source)

    def test_the_teaching_turn_does_not_echo_itself(self):
        engine = self._engine()
        result = engine.turn(self.METAPHOR, now=1000.0)
        self.assertNotIn("learned-", result.source)

    def test_a_question_is_not_minted(self):
        engine = self._engine()
        before = len(engine.kb.docs)
        engine.turn("区块链的哈希指针是不是防篡改的？", now=1000.0)
        self.assertEqual(len(engine.kb.docs), before)

    def test_teaching_the_same_metaphor_twice_mints_once(self):
        engine = self._engine()
        engine.turn(self.METAPHOR, now=1000.0)
        engine.turn(self.METAPHOR, now=1100.0)
        self.assertEqual(engine.kb.learned.cards.count(
            engine.kb.learned.cards[0]), 1)

    def test_a_similar_metaphor_revises_the_same_card(self):
        engine = self._engine()
        engine.turn(self.METAPHOR, now=1000.0)
        similar = ("哈希指针可以想成一串首尾相扣的蜡封信件，"
                   "改一封后面全都得重新盖章。")
        engine.turn(similar, now=1100.0)
        self.assertEqual(len(engine.kb.learned.cards), 1)
        self.assertEqual(len(engine.kb.learned.cards[0]["revisions"]), 1)

    def test_an_answer_from_a_learned_card_counts_as_a_hit(self):
        engine = self._engine()
        engine.turn(self.METAPHOR, now=1000.0)
        result = engine.turn("哈希指针为什么能防篡改？蜡封信件的比喻再讲讲",
                             now=1100.0)
        card_id = result.source.split(":", 1)[1]
        self.assertTrue(card_id.startswith("learned-"))
        self.assertGreaterEqual(
            engine.kb.learned.stats.get(card_id, {}).get("hits", 0), 1)


class TestTwoStageGate(unittest.TestCase):
    LOOSE = "哈希指针相当于一串首尾相扣的蜡封信件，改一封后面全得重盖章。"
    STRICT = "哈希指针可以想成一串首尾相扣的蜡封信件，改一封后面全得重盖章。"

    def test_loose_markers_are_candidates_but_not_explicit(self):
        self.assertTrue(metaphor_candidate(self.LOOSE))
        self.assertFalse(looks_like_metaphor(self.LOOSE))
        self.assertTrue(metaphor_candidate(self.STRICT))
        self.assertTrue(looks_like_metaphor(self.STRICT))

    def test_pointing_at_an_earlier_comparison_is_not_a_new_one(self):
        self.assertFalse(metaphor_candidate("就像刚才说的蜡封信件那样，后面全得重盖章。"))
        self.assertFalse(looks_like_metaphor("就像刚才说的蜡封信件那样，后面全得重盖章。"))

    def test_a_word_that_merely_contains_xiang_is_not_a_comparison(self):
        for text in ("原来太阳光藏在树叶里，这个比喻太形象了哈哈！",
                     "这张图像的分辨率好像有点低。"):
            with self.subTest(text=text):
                self.assertFalse(looks_like_metaphor(text))


class TestPendingPromotion(unittest.TestCase):
    LOOSE = "哈希指针相当于一串首尾相扣的蜡封信件，改一封后面全得重盖章。"

    def _engine(self) -> MindEngine:
        self.directory = temp_dir()
        return MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=self.directory),
            auto_save=False,
            language="zh",
        )

    def test_a_loose_comparison_waits(self):
        engine = self._engine()
        engine.turn(self.LOOSE, now=1000.0)
        self.assertEqual(engine.kb.learned.cards, [])

    def test_praise_promotes_the_pending_comparison(self):
        engine = self._engine()
        engine.turn(self.LOOSE, now=1000.0)
        engine.turn("太棒了哈哈，这个比喻真好玩", now=1060.0)
        self.assertEqual(len(engine.kb.learned.cards), 1)

    def test_reusing_the_comparison_promotes_it(self):
        engine = self._engine()
        engine.turn(self.LOOSE, now=1000.0)
        engine.turn("蜡封信件改一封就得重盖章，这个防篡改的道理再展开讲讲",
                    now=1060.0)
        self.assertEqual(len(engine.kb.learned.cards), 1)

    def test_a_correction_cancels_it(self):
        engine = self._engine()
        engine.turn(self.LOOSE, now=1000.0)
        engine.turn("不对，我不是这个意思", now=1060.0)
        engine.turn("太棒了哈哈", now=1120.0)
        self.assertEqual(engine.kb.learned.cards, [])

    def test_an_unconfirmed_candidate_expires(self):
        engine = self._engine()
        engine.turn(self.LOOSE, now=1000.0)
        for i, text in enumerate(("今天下雨了", "路上有点堵", "晚上吃什么")):
            engine.turn(text, now=1060.0 + i * 60)
        self.assertEqual(engine.kb.learned.cards, [])


class TestHardNegatives(unittest.TestCase):
    """Praise, address and quotative uses of 「像」 are not comparisons."""

    PRAISE = (
        "费曼先生，像你这么聪明的人真是太少见了！",
        "像我们这种普通人哪里听得懂量子电动力学啊。",
        "像你这么大年纪还这么有好奇心，真让人佩服。",
        "我觉得像他那样的老师才叫好老师。",
    )

    def test_praise_never_mints_a_card(self):
        for text in self.PRAISE:
            with self.subTest(text=text):
                self.assertFalse(looks_like_metaphor(text))
                self.assertIsNone(card_row(text))

    def test_a_real_comparison_still_passes(self):
        self.assertTrue(looks_like_metaphor(
            "电子像一群看不见的蜜蜂，嗡嗡地围着原子核转。"))

    def test_the_engine_mints_nothing_from_praise(self):
        directory = temp_dir()
        engine = MindEngine(
            persona=get_persona("feynman-zh"), bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=directory), auto_save=False, language="zh")
        for i, text in enumerate(self.PRAISE):
            engine.turn(text, now=1000.0 + i * 60)
        self.assertEqual(engine.kb.learned.cards, [])


class TestExplicitConfirmation(unittest.TestCase):
    LOOSE = "哈希指针相当于一串首尾相扣的蜡封信件，改一封后面全得重盖章。"

    def test_a_short_repeat_confirms(self):
        self.assertTrue(explicit_confirmation(
            "对，就是首尾相扣的蜡封信件，改一封后面全得重盖章。", self.LOOSE))

    def test_a_quoted_span_confirms(self):
        self.assertTrue(explicit_confirmation(
            "记住这句：「首尾相扣的蜡封信件，改一封后面全得重盖章」。", self.LOOSE))

    def test_an_essay_that_merely_mentions_it_does_not(self):
        long_turn = ("今天开会讲了很长时间，先说了预算，又说了排期，"
                     "中间有人提到首尾相扣的蜡封信件，改一封后面全得重盖章，"
                     "最后也没结论，大家就先散了，明天还得继续。")
        self.assertFalse(explicit_confirmation(long_turn, self.LOOSE))

    def test_the_quote_promotes_even_at_high_cortisol(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"), bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=temp_dir()), auto_save=False,
            language="zh")
        engine.turn(self.LOOSE, now=1000.0)
        engine.state.set("cortisol", 1.0)
        engine.state.last_ts = 1060.0
        engine.turn("对，就是「一串首尾相扣的蜡封信件」这个说法",
                    now=1060.0)
        self.assertEqual(len(engine.kb.learned.cards), 1)

    def test_praise_does_not_promote_while_suspicious(self):
        engine = MindEngine(
            persona=get_persona("feynman-zh"), bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=temp_dir()), auto_save=False,
            language="zh")
        engine.turn(self.LOOSE, now=1000.0)
        engine.state.set("cortisol", 1.0)
        engine.state.last_ts = 1060.0
        engine.turn("太棒了哈哈，这个比喻真好玩", now=1060.0)
        self.assertEqual(engine.kb.learned.cards, [])
        # and the candidate is still waiting, not lost
        self.assertEqual(len(engine._pending_metaphors), 1)



    FIRST = ("区块链的哈希指针可以想成一串首尾相扣的蜡封信件，"
             "改一封后面全得重盖章。")
    SECOND = ("哈希指针可以想成一串首尾相扣的蜡封信件，"
              "改一封后面全都得重新盖章。")

    def test_a_similar_comparison_revises_instead_of_duplicating(self):
        store = LearnedStore(Path(temp_dir()))
        self.assertTrue(store.add_card(card_row(self.FIRST)))
        self.assertGreater(similarity(self.FIRST, self.SECOND), 0.55)
        self.assertTrue(store.add_card(card_row(self.SECOND)))
        self.assertEqual(len(store.cards), 1)
        self.assertEqual(len(store.cards[0]["revisions"]), 1)
        self.assertIn("蜡封信件", store.cards[0]["body"])

    def test_the_old_wording_stays_an_alias_of_the_revised_card(self):
        store = LearnedStore(Path(temp_dir()))
        store.add_card(card_row(self.FIRST))
        store.add_card(card_row(self.SECOND))
        cid = store.cards[0]["id"]
        self.assertTrue(any("哈希指针" in p for p in store.aliases[cid]))

    def test_identical_text_is_a_no_op(self):
        store = LearnedStore(Path(temp_dir()))
        store.add_card(card_row(self.FIRST))
        self.assertFalse(store.add_card(card_row(self.FIRST)))


class TestCapacityAndArchiving(unittest.TestCase):
    def _rows(self, n: int):
        bodies = [
            "摩擦力就像地上铺了一层细沙，推着箱子走总要先碾过沙粒。",
            "电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。",
            "光合作用就像一个把阳光存进糖里的存钱罐，叶绿素是罐口。",
            "记忆就像往抽屉里塞纸条，塞得越乱下一次越难抽出来。",
        ]
        return [card_row(bodies[i % len(bodies)].replace("就像", f"就像第{i}版"))
                for i in range(n)]

    def test_the_least_useful_card_is_archived_not_refused(self):
        store = LearnedStore(Path(temp_dir()))
        with mock.patch.object(learner, "MAX_CARDS", 2):
            for row in self._rows(3):
                self.assertTrue(store.add_card(row))
        self.assertEqual(len(store.cards), 2)
        self.assertEqual(len(store.archived), 1)

    def test_hits_protect_a_card_from_eviction(self):
        store = LearnedStore(Path(temp_dir()))
        with mock.patch.object(learner, "MAX_CARDS", 2):
            first, second, third = self._rows(3)
            store.add_card(first)
            store.add_card(second)
            for _ in range(5):
                store.record_hit(second["id"])
            store.add_card(third)
        self.assertIn(second["id"], [c["id"] for c in store.cards])

    def test_an_archived_card_can_be_restored(self):
        store = LearnedStore(Path(temp_dir()))
        row = card_row(self._rows(1)[0]["body"])
        store.add_card(row)
        self.assertTrue(store.archive_card(row["id"]))
        self.assertEqual(store.cards, [])
        self.assertTrue(store.restore_card(row["id"]))
        self.assertEqual(len(store.cards), 1)

    def test_a_card_minted_this_session_is_not_evicted_first(self):
        store = LearnedStore(Path(temp_dir()))
        with mock.patch.object(learner, "MAX_CARDS", 2):
            first, second, third = self._rows(3)
            store.add_card(first)
            store.add_card(second)
            store.add_card(third)   # cap exceeded; the new card is fresh
        ids = [c["id"] for c in store.cards]
        self.assertIn(third["id"], ids)

    def test_a_cold_topic_keeps_its_last_card(self):
        store = LearnedStore(Path(temp_dir()))
        hot = [
            card_row("电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。",
                     topic="physics"),
            card_row("摩擦力就像地上铺了一层细沙，推着箱子走总要先碾过沙粒。",
                     topic="physics"),
            card_row("光合作用就像一个把阳光存进糖里的存钱罐，叶绿素是罐口。",
                     topic="physics"),
            card_row("引力就像一张绷紧的床单，太阳把中间压出一个坑。",
                     topic="physics"),
        ]
        cold = card_row("记忆就像往抽屉里塞纸条，塞得越乱下一次越难抽出来。",
                        topic="art")
        for row in hot + [cold]:
            self.assertIsNotNone(row)
            store.add_card(row)
        self.assertEqual(len(store.cards), 5)
        for row in hot:
            for _ in range(5):
                store.record_hit(row["id"])
        store._fresh.clear()  # simulate a later session: no grace left
        with mock.patch.object(learner, "MAX_CARDS", 3):
            store._enforce_capacity()
        self.assertIn(cold["id"], [c["id"] for c in store.cards])
        self.assertGreater(len(store.archived), 0)


class TestSleepMerge(unittest.TestCase):
    HOST = "电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。"
    VICTIM = "电流像一条挤满人的走廊，你推一下这头，那头的人立刻往前挪。"

    def _store(self) -> LearnedStore:
        store = LearnedStore(Path(temp_dir()))
        store.add_card(card_row(self.HOST, topic="physics"))
        store.add_card(card_row(self.VICTIM, topic="physics"))
        return store

    def test_the_band_between_revision_and_novelty_is_absorbed(self):
        store = self._store()
        self.assertEqual(len(store.cards), 2)
        host = max(store.cards, key=store.utility)
        store.record_hit(host["id"])
        merged = store.sleep_merge()
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(store.cards), 1)
        self.assertEqual(store.archived[0]["archive_reason"],
                         "sleep_cluster_merge")
        self.assertEqual(store.archived[0]["merged_into"], host["id"])

    def test_the_victim_stays_findable_as_an_alias(self):
        store = self._store()
        host = max(store.cards, key=store.utility)
        store.record_hit(host["id"])
        store.sleep_merge()
        self.assertTrue(any("挤满人的走廊" in p
                            for p in store.aliases[host["id"]]))

    def test_no_revision_event_is_emitted(self):
        directory = Path(temp_dir())
        store = LearnedStore(directory)
        store.add_card(card_row(self.HOST, topic="physics"))
        store.add_card(card_row(self.VICTIM, topic="physics"))
        host = max(store.cards, key=store.utility)
        store.record_hit(host["id"])
        store.sleep_merge()
        kinds = [json.loads(line)["kind"] for line in
                 (directory / EVENTS_FILE).read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("revision", kinds)
        # replay rebuilds the same merged view
        again = LearnedStore(directory)
        self.assertEqual(len(again.cards), 1)

    def test_different_topics_are_never_merged(self):
        store = LearnedStore(Path(temp_dir()))
        store.add_card(card_row(self.HOST, topic="physics"))
        store.add_card(card_row(self.VICTIM, topic="art"))
        host = max(store.cards, key=store.utility)
        store.record_hit(host["id"])
        self.assertEqual(store.sleep_merge(), [])
        self.assertEqual(len(store.cards), 2)

    def test_never_used_cards_are_left_alone(self):
        store = self._store()
        self.assertEqual(store.sleep_merge(), [])
        self.assertEqual(len(store.cards), 2)

    def test_at_most_five_merges_per_night(self):
        store = LearnedStore(Path(temp_dir()))
        pairs = [
            ("电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。",
             "电流就像一串挤满人的走廊，你推一下这头，那头的人立刻往前挪。"),
            ("摩擦力就像地上铺了一层细沙，推着箱子走总要先碾过沙粒。",
             "摩擦力好比在地面上撒了一层细沙，推箱子时得先碾过这些沙子。"),
            ("光合作用就好像一个把阳光存进糖里的存钱罐，叶绿素是罐口。",
             "光合作用好比一个储蓄罐，把阳光一点点存成糖，叶绿素就是罐口。"),
        ]
        for i in range(4):
            for a, b in pairs:
                for body in (a, b):
                    row = card_row(body.replace("就像", f"就像第{i}组"),
                                   topic="physics")
                    self.assertIsNotNone(row)
                    store.add_card(row)
        for card in store.cards:
            store.record_hit(card["id"])
        merged = store.sleep_merge()
        self.assertLessEqual(len(merged), 5)
        self.assertGreater(len(merged), 0)


class TestColdShelf(unittest.TestCase):
    OLD = "分形就像海岸线，放大一寸还是锯齿，尺子越短量出的长度越长。"
    YOUNG = "湍流就像摇晃一杯加奶的咖啡，漩涡套着漩涡往下滚。"
    USED = "惰性气体就像一屋子懒人，谁也不愿先站起来握手。"
    ALIASED = "磁场就像看不见的传送带，带电粒子一上去就被带着拐弯。"

    def _seed(self, store: LearnedStore, text: str, days: int) -> dict:
        row = card_row(text, topic="physics")
        self.assertIsNotNone(row)
        row["learned_at"] = (date.today() - timedelta(days=days)).isoformat()
        store._append({"kind": "card", "row": row})
        store.load()
        return row

    def _store(self) -> LearnedStore:
        return LearnedStore(Path(temp_dir()))

    def test_a_never_used_old_card_is_cold(self):
        store = self._store()
        row = self._seed(store, self.OLD, days=90)
        cold = store.cold()
        self.assertEqual([item["id"] for item in cold], [row["id"]])
        self.assertGreaterEqual(cold[0]["days"], 89)
        self.assertEqual(cold[0]["topic"], "physics")
        again = LearnedStore(store.dir)
        self.assertEqual([item["id"] for item in again.cold()], [row["id"]])

    def test_a_reached_card_is_not_cold(self):
        store = self._store()
        row = self._seed(store, self.USED, days=90)
        store.record_hit(row["id"])
        self.assertEqual(store.cold(), [])

    def test_praise_or_an_alias_warms_a_card(self):
        store = self._store()
        praised = self._seed(store, self.USED, days=90)
        aliased = self._seed(store, self.ALIASED, days=90)
        store.record_praise(praised["id"])
        store.add_alias(aliased["id"], "看不见的传送带")
        self.assertEqual(store.cold(), [])

    def test_a_young_card_is_not_cold(self):
        store = self._store()
        self._seed(store, self.YOUNG, days=3)
        self.assertEqual(store.cold(), [])

    def test_curate_files_a_candidate_and_archives_nothing(self):
        from omega_mind import review

        store = self._store()
        row = self._seed(store, self.OLD, days=90)
        queued = store.curate()
        self.assertEqual(len(queued), 1)
        candidates = review.read(store)["candidates"]
        self.assertEqual(candidates[queued[0]]["kind"], "cold")
        self.assertEqual(candidates[queued[0]]["card_id"], row["id"])
        self.assertEqual([c["id"] for c in store.cards], [row["id"]])
        self.assertEqual(review.read(store)["decisions"], [])
        self.assertEqual(review.active(review.read(store)), [])

    def test_curate_is_idempotent_across_nights(self):
        from omega_mind import review

        store = self._store()
        self._seed(store, self.OLD, days=90)
        first = store.curate()
        second = store.curate()
        self.assertEqual(first, second)
        self.assertEqual(len(review.read(store)["candidates"]), 1)

    def test_curate_is_capped_per_night(self):
        store = self._store()
        for i in range(learner.MAX_COLD_PER_NIGHT + 3):
            self._seed(store, f"{self.OLD[:-1]}，第{i}号变体。", days=90 + i)
        self.assertEqual(len(store.curate()), learner.MAX_COLD_PER_NIGHT)


class TestEventLog(unittest.TestCase):
    def test_events_are_the_source_and_replay_rebuilds_the_view(self):
        directory = Path(temp_dir())
        store = LearnedStore(directory)
        row = card_row("记忆就像往抽屉里塞纸条，塞得越乱下一次越难抽出来。")
        store.add_card(row)
        store.add_alias(row["id"], "抽屉里的纸条")
        kinds = [json.loads(line)["kind"] for line in
                 (directory / EVENTS_FILE).read_text(encoding="utf-8").splitlines()]
        self.assertEqual(kinds[:2], ["card", "alias"])
        again = LearnedStore(directory)
        self.assertEqual([c["id"] for c in again.cards], [row["id"]])
        self.assertIn("抽屉里的纸条", again.aliases[row["id"]])

    def test_legacy_two_file_format_is_migrated(self):
        directory = Path(temp_dir())
        row = card_row("电流就像挤满人的走廊，一头推一下整条队伍同时往前挪。")
        (directory / learner.CARDS_FILE).write_text(
            json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        (directory / learner.ALIASES_FILE).write_text(
            json.dumps({row["id"]: ["挤满人的走廊"]}, ensure_ascii=False),
            encoding="utf-8")
        store = LearnedStore(directory)
        self.assertEqual([c["id"] for c in store.cards], [row["id"]])
        self.assertIn("挤满人的走廊", store.aliases[row["id"]])
        self.assertTrue((directory / EVENTS_FILE).exists())

    def test_stats_survive_a_reload(self):
        directory = Path(temp_dir())
        store = LearnedStore(directory)
        row = card_row("记忆就像往抽屉里塞纸条，塞得越乱下一次越难抽出来。")
        store.add_card(row)
        store.record_praise(row["id"])
        again = LearnedStore(directory)
        self.assertEqual(again.stats[row["id"]]["praise"], 1)


class TestLearnedCorpusTie(unittest.TestCase):
    HEAT = "电脑运行久了机箱烫手，这跟信息被擦除有什么物理关系吗？"

    def test_a_learned_restatement_does_not_veto_the_corpus_card(self):
        directory = temp_dir()
        store = LearnedStore(Path(directory))
        store.add_card(card_row(
            "擦除一位信息就像把两杯水倒进同一个盆里，这个比喻绝了。"))
        kb = FeynmanKB(learned_dir=directory)
        found = kb.lookup(self.HEAT)
        self.assertEqual(found.source, "feynman_kb_soft:math-comp-004")
        self.assertIn("发热", found.text)


class TestTopicAndAudit(unittest.TestCase):
    def test_a_taught_card_gets_the_topic_of_its_nearest_corpus_card(self):
        kb = FeynmanKB(learned_dir=temp_dir())
        topic = kb.topic_for("电脑芯片发热和擦除信息有关系吗？")
        self.assertNotEqual(topic, "general")

    def test_a_lookup_reports_the_tokens_it_matched(self):
        kb = FeynmanKB(learned_dir=temp_dir())
        found = kb.lookup("为什么电脑芯片会发热")
        self.assertTrue(found.matched)
        self.assertIn("matched", found.to_dict())

    def test_a_minted_card_carries_its_topic(self):
        directory = temp_dir()
        engine = MindEngine(
            persona=get_persona("feynman-zh"), bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(learned_dir=directory), auto_save=False, language="zh")
        engine.turn("哈希指针可以想成一串首尾相扣的蜡封信件，改一封后面全得重盖章。",
                    now=1000.0)
        card = engine.kb.learned.cards[0]
        self.assertTrue(card.get("topic"))
        self.assertNotEqual(card.get("topic"), "")


class TestSoftRecall(unittest.TestCase):
    def test_a_clear_near_threshold_card_is_still_answered(self):
        # 0.287: below the confident gate, clearly ahead of the runner-up.
        # (The old probe "能不能聊聊光和量子" now ties between two on-topic
        # cards — the light/quantum intro and the wave-particle intro — and a
        # same-class near-tie is refused by design.)
        kb = FeynmanKB(learned_dir=temp_dir())
        found = kb.lookup("黑洞里面会发生什么")
        self.assertIsNotNone(found.text)
        self.assertTrue(found.source.startswith("feynman_kb_soft:")
                        or found.source.startswith("feynman_kb:"))

    def test_a_weak_query_is_still_a_no_match(self):
        kb = FeynmanKB(learned_dir=temp_dir())
        found = kb.lookup("今天中午吃什么好呢")
        self.assertIsNone(found.text)
        self.assertTrue(found.source.endswith(":no_match"))


if __name__ == "__main__":
    unittest.main()
