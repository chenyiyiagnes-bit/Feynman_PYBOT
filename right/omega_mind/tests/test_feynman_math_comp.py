"""The math-and-computation card pack (Chinese source + English overlay).

These tests pin what the loader can actually see: the real card schema
(``title``/``text``/``tags``/``aliases``), one-to-one zh/en pairing, the
manifest entry, and that everyday questions reach the right card.  They skip
when the feynman-open knowledge directory is not on this machine.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from omega_mind.config import get_persona
from omega_mind.engine import MindEngine
from omega_mind.feynman_kb import DEFAULT_DIRS, FeynmanKB, package_language
from omega_mind.pybot_bridge import PyBotBridge

CJK = re.compile(r"[一-鿿]")

NEW_IDS = {f"math-comp-00{i}" for i in range(1, 9)}

ZH_QUERIES = {
    "路径积分里的小箭头怎么相加": {"math-comp-001"},
    "微积分的几何本质是什么": {"math-comp-002"},
    "自己发明数学符号好吗": {"math-comp-003"},
    "为什么电脑芯片会发热": {"math-comp-004"},
    # bio-026 is an older one-line card making the same 1981 point.
    "为什么要造量子计算机": {"math-comp-005", "bio-026"},
    "纳米机器和底层还有很大空间": {"math-comp-006"},
    "车牌号巧合的概率有多大": {"math-comp-007"},
    "你怎么心算 1729.03 的立方根": {"math-comp-008"},
}

EN_QUERIES = {
    "How do you add the little arrows in a path integral?": {"math-comp-001"},
    "What is calculus really about?": {"math-comp-002"},
    "Why invent your own math notation?": {"math-comp-003"},
    "Why do computer chips get hot?": {"math-comp-004"},
    # comp-002 is the physics-domain card on the same 1982 argument.
    "Why do we need a quantum computer to simulate nature?": {"math-comp-005",
                                                              "comp-002"},
    # bio-025 is the older biography card on the same 1959 talk.
    "Is there plenty of room at the bottom?": {"math-comp-006", "bio-025"},
    "What are the odds of seeing this license plate?": {"math-comp-007"},
    "How did you beat the abacus expert at mental math?": {"math-comp-008"},
}

REQUIRED_FIELDS = ("id", "type", "title", "text", "tags", "aliases",
                   "source", "source_class", "verification_status")


def _knowledge_dir(language: str = "zh") -> Path:
    """The package in ``language`` that ships the math/comp cards.

    Never the first directory that happens to have the file: the English
    package carries the same file names, and the row schema under test is
    the one the Chinese source rows use.
    """
    for path in DEFAULT_DIRS:
        if (package_language(path.parent) == language
                and (path / "feynman_math_comp.jsonl").is_file()):
            return path
    raise unittest.SkipTest("feynman-open knowledge not on this machine")


def _kb_or_skip(language: str = "zh") -> FeynmanKB:
    _knowledge_dir()
    kb = FeynmanKB(language=language)
    if not kb.available() or kb.language != language:
        raise unittest.SkipTest(f"{language!r} knowledge package did not load")
    return kb


class TestMathCompCardsLoad(unittest.TestCase):
    def setUp(self):
        self.kb = _kb_or_skip("zh")
        self.en = _kb_or_skip("en")

    def test_every_card_loads_in_both_languages(self):
        zh = {d.id for d in self.kb.docs}
        en = {d.id for d in self.en.docs}
        self.assertTrue(NEW_IDS <= zh, f"missing zh: {NEW_IDS - zh}")
        self.assertTrue(NEW_IDS <= en, f"missing en: {NEW_IDS - en}")

    def test_pairs_stay_aligned(self):
        zh = {d.id for d in self.kb.docs if d.id in NEW_IDS}
        en = {d.id for d in self.en.docs if d.id in NEW_IDS}
        self.assertEqual(zh, en)

    def test_english_overlay_has_no_chinese(self):
        by_id = {d.id: d for d in self.en.docs}
        for cid in NEW_IDS:
            doc = by_id[cid]
            self.assertFalse(CJK.search(doc.answer),
                             f"{cid} leaked Chinese into English mode")
            self.assertFalse(CJK.search(doc.title),
                             f"{cid} has a Chinese title in English mode")

    def test_rows_use_the_fields_the_loader_reads(self):
        path = _knowledge_dir() / "feynman_math_comp.jsonl"
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["id"] for r in rows}, NEW_IDS)
        for row in rows:
            for field in REQUIRED_FIELDS:
                self.assertIn(field, row, f"{row['id']} missing {field}")
            self.assertEqual(row["type"], "lecture_card")
            for dead in ("topic", "triggers", "content"):
                self.assertNotIn(dead, row, f"{row['id']} uses {dead}, "
                                           "which the loader ignores")

    def test_manifest_entry_matches_the_file(self):
        knowledge = _knowledge_dir()
        manifest = json.loads(
            (knowledge / "MANIFEST.json").read_text(encoding="utf-8"))
        entry = manifest["files"]["feynman_math_comp.jsonl"]
        data = (knowledge / "feynman_math_comp.jsonl").read_bytes()
        rows = [line for line in data.decode("utf-8").splitlines()
                if line.strip()]
        self.assertEqual(entry["rows"], len(rows))
        self.assertEqual(entry["withheld_private_rows"], 0)
        self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())


class TestMathCompRetrieval(unittest.TestCase):
    def setUp(self):
        self.kb = _kb_or_skip("zh")
        self.en = _kb_or_skip("en")

    def test_chinese_topics_reach_their_card(self):
        for query, expected in ZH_QUERIES.items():
            with self.subTest(query=query):
                hits = self.kb.search(query, limit=3)
                self.assertTrue(hits, f"no hit for {query}")
                score, doc = hits[0]
                self.assertGreaterEqual(score, self.kb.MIN_SCORE)
                self.assertIn(doc.id, expected)

    def test_english_topics_reach_their_card(self):
        for query, expected in EN_QUERIES.items():
            with self.subTest(query=query):
                hits = self.en.search(query, limit=3)
                self.assertTrue(hits, f"no hit for {query}")
                score, doc = hits[0]
                self.assertGreaterEqual(score, self.kb.MIN_SCORE)
                self.assertIn(doc.id, expected)


class TestMathCompInTheEngine(unittest.TestCase):
    def test_answer_comes_from_the_new_pack(self):
        _kb_or_skip()
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(),
            auto_save=False,
            language="zh",
        )
        result = engine.turn("路径积分里的小箭头怎么相加", now=1000.0)
        self.assertIn("小箭头", result.reply)
        self.assertIn("math-comp-001", result.source)

    def test_language_switch_uses_the_english_card(self):
        _kb_or_skip()
        engine = MindEngine(
            persona=get_persona("feynman-zh"),
            bridge=PyBotBridge.disabled(),
            kb=FeynmanKB(language="en"),
            auto_save=False,
            language="en",
        )
        result = engine.turn("What is calculus really about?", now=1000.0)
        self.assertIn("thin slices", result.reply)
        self.assertFalse(CJK.search(result.reply))


if __name__ == "__main__":
    unittest.main()
