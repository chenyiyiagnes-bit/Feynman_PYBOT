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
            lang.load("zh", profile="classical")
        self.assertIn("classical", str(caught.exception))

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
