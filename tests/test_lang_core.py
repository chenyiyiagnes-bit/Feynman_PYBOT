"""The language core keeps retrieval byte-identical while rules move to data.

`tests/data/lang_core_golden.json` was recorded from the pre-extraction
implementation. It is the contract: moving a rule into `lang/<language>/` must
change nothing the index or the query path sees.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind import feynman_kb as kb  # noqa: E402
from omega_mind import lang  # noqa: E402

GOLDEN = ROOT / "tests" / "data" / "lang_core_golden.json"


class GoldenEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        cls.queries = cls.golden["queries"]

    def assert_matches(self, name: str, produce) -> None:
        expected = self.golden[name]
        for query in self.queries:
            with self.subTest(rule=name, query=query):
                self.assertEqual(produce(query), expected[query])

    def test_zh_tokens(self) -> None:
        self.assert_matches("tokens", kb._tokens)

    def test_en_tokens(self) -> None:
        self.assert_matches("tokens_en", kb._tokens_en)

    def test_zh_expansion(self) -> None:
        self.assert_matches("expand_zh", kb._expand_zh)

    def test_en_expansion(self) -> None:
        self.assert_matches("expand_en", kb._expand_en)

    def test_zh_bridge(self) -> None:
        self.assert_matches("bridge_zh", kb._bridge_zh)

    def test_zh_unigrams(self) -> None:
        self.assert_matches("unigrams", kb._unigrams)

    def test_en_stemming(self) -> None:
        for word, expected in self.golden["stem_en"].items():
            with self.subTest(word=word):
                self.assertEqual(kb._stem_en(word), expected)


class Segmentation(unittest.TestCase):
    """``segment()`` is the shared word-level entry point for both brains."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.zh = lang.load("zh")
        cls.en = lang.load("en")

    def test_empty_input(self) -> None:
        self.assertEqual(self.zh.segment(""), [])
        self.assertEqual(self.zh.segment("   "), [])

    def test_lexicon_word_beats_characters(self) -> None:
        self.assertEqual(self.zh.segment("物理学家"), ["物理学家"])

    def test_particles_break_a_run(self) -> None:
        words = self.zh.segment("浮在水面上的木块")
        self.assertIn("木块", words)
        self.assertIn("的", words)
        self.assertIn("在", words)

    def test_unknown_run_stays_together(self) -> None:
        self.assertEqual(self.zh.segment("木块"), ["木块"])

    def test_long_unknown_run_is_chunked(self) -> None:
        words = self.zh.segment("超导磁悬浮列车跑得快不快")
        self.assertTrue(all(len(word) <= 2 for word in words if word not in ("不",)))

    def test_ascii_and_digits_survive(self) -> None:
        self.assertEqual(self.zh.segment("hello 量子 123"),
                         ["hello", "量子", "123"])

    def test_punctuation_is_dropped(self) -> None:
        self.assertEqual(self.zh.segment("熵，到底；是什么？"), ["熵", "到底", "是", "什么"])

    def test_english_segments_into_words(self) -> None:
        self.assertEqual(self.en.segment("What is the scientific method?"),
                         ["scientific", "method"])

    def test_left_brain_adapter_agrees_with_core(self) -> None:
        left = _left_segmentation()
        if left is None:
            self.skipTest("left brain plugins not importable here")
        for query in ("熵到底是什么", "物理学家怎么看待权威", "hello 量子 123"):
            with self.subTest(query=query):
                self.assertEqual(left.segment(query), self.zh.segment(query))

    def test_left_brain_content_words_drop_stop_characters(self) -> None:
        left = _left_segmentation()
        if left is None:
            self.skipTest("left brain plugins not importable here")
        self.assertEqual(left.content_words("物理学家怎么看待权威"),
                         ["物理学家", "怎么", "看待", "权威"])

    def test_spans_point_at_the_words(self) -> None:
        text = "  熵，到底；是什么？"
        spans = self.zh.segment_spans(text)
        self.assertEqual([word for word, _, _ in spans], self.zh.segment(text))
        for word, start, end in spans:
            with self.subTest(word=word):
                self.assertEqual(text[start:end], word)

    def test_english_spans_keep_the_surface_form(self) -> None:
        text = "Studies of the scientific method"
        spans = self.en.segment_spans(text)
        self.assertEqual(text[spans[0][1]:spans[0][2]], "Studies")
        self.assertIn("scientific", [word for word, _, _ in spans])
        for word, start, end in spans:
            self.assertEqual(text[start:end], word)

    def test_empty_text_has_no_spans(self) -> None:
        self.assertEqual(self.zh.segment_spans(""), [])
        self.assertEqual(self.zh.segment_spans("   "), [])


def _left_segmentation():
    path = ROOT / "left"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        from plugins import word_segmentation
    except Exception:  # noqa: BLE001 - the left tree is optional in some checkouts
        return None
    return word_segmentation


class ClassicalProfile(unittest.TestCase):
    """A profile adds to its language without taking anything away."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.classical = lang.load("zh", "classical")
        cls.modern = lang.load("zh")

    def test_profile_is_listed(self) -> None:
        self.assertIn("classical", lang.profiles("zh"))

    def test_classical_question_expands_to_classical_topics(self) -> None:
        expanded = self.classical.expand("何为道")
        for word in ("道", "自然", "无为"):
            self.assertIn(word, expanded)

    def test_classical_bridge_exists(self) -> None:
        self.assertIn("水", self.classical.bridge("上善若水"))
        self.assertIn("不争", self.classical.bridge("上善若水"))

    def test_modern_terms_survive_the_overlay(self) -> None:
        self.assertIn("熵", self.classical.expand("熵是什么"))

    def test_stop_bigrams_are_union_not_replacement(self) -> None:
        self.assertTrue(set(self.modern.data["stop_bigrams"])
                        <= set(self.classical.data["stop_bigrams"]))
        self.assertIn("之谓", self.classical.data["stop_bigrams"])

    def test_lexicon_is_union(self) -> None:
        modern = set(self.modern.data["lexicon"])
        classical = set(self.classical.data["lexicon"])
        self.assertTrue(modern <= classical)
        self.assertIn("庖丁解牛", classical)

    def test_classical_particles_are_stop_characters(self) -> None:
        self.assertIn("乎", self.classical._unigram_stop)
        self.assertEqual(self.classical.segment("庖丁解牛游刃有余"),
                         ["庖丁解牛", "游刃有余"])

    def test_kb_can_read_with_a_profile(self) -> None:
        kb = kbmod(autoload=False, profile="classical")
        self.assertEqual(kb.profile, "classical")
        self.assertIs(kb.core_zh, self.classical)
        self.assertIs(kbmod(autoload=False).core_zh, self.modern)

    def test_kb_names_an_unknown_profile(self) -> None:
        with self.assertRaises(lang.LanguageRuleError):
            kbmod(autoload=False, profile="no-such-profile")


def kbmod(autoload: bool = False, **kwargs):
    return kb.FeynmanKB(autoload=autoload, **kwargs)


class RuleFiles(unittest.TestCase):
    def test_supported_languages_load(self) -> None:
        self.assertEqual(lang.load("zh").language, "zh")
        self.assertEqual(lang.load("en").language, "en")

    def test_unknown_language_is_named(self) -> None:
        with self.assertRaises(lang.LanguageRuleError) as caught:
            lang.load("fr")
        self.assertIn("fr", str(caught.exception))

    def test_unknown_profile_is_named(self) -> None:
        with self.assertRaises(lang.LanguageRuleError) as caught:
            lang.load("zh", profile="no-such-profile")
        self.assertIn("no-such-profile", str(caught.exception))

    def test_profiles_are_discoverable(self) -> None:
        self.assertIn("default", lang.profiles("zh"))
        self.assertIn("default", lang.profiles("en"))

    def test_rules_are_shared_not_copied(self) -> None:
        self.assertIs(lang.load("zh"), lang.load("zh"))

    def test_zh_rules_and_engine_agree(self) -> None:
        core = lang.load("zh")
        self.assertEqual(core.tokens("熵是什么"), kb._tokens("熵是什么"))
        # zh has no stem table, so English-looking input passes through unchanged.
        self.assertEqual(core.stem("deviation"), "deviation")

    def test_en_core_has_no_zh_tables(self) -> None:
        core = lang.load("en")
        self.assertEqual(core.unigrams("中文字符"), [])
        self.assertEqual(core.bridge("发热"), "发热")


if __name__ == "__main__":
    unittest.main()
