"""The style band measures a reply; it never edits or blocks one.

P2 is log-only, so the tests here check two things: the measurement says
what it should on known-good and known-bad text, and the audit installed on
an engine's reply exit returns the original reply untouched.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app import style_audit
from bridge import style_gate as gate

PERSONAS = Path(__file__).resolve().parents[1] / "personas"

GOOD_ZH = "费曼说，如果你不能把一件事讲给大一新生听，就说明你自己没搞懂。"
GOOD_EN = "The thing about light is that it just keeps going, and nobody knows why."


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.zh = gate.band_for_docs("zh", [_doc("a", GOOD_ZH)] * 12)["overall"]
        self.en = gate.band_for_docs("en", [_doc("a", GOOD_EN)] * 12)["overall"]

    def test_reply_in_band_passes(self):
        self.assertTrue(gate.check(GOOD_ZH, self.zh)["ok"])
        self.assertTrue(gate.check(GOOD_EN, self.en)["ok"])

    def test_cliche_fails(self):
        text = "首先，我们要理解本质。其次，值得注意的是它涉及多个维度。综上所述，就是这样。"
        verdict = gate.check(text, self.zh)
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["no_ai_cliche"])
        self.assertIn("首先", verdict["metrics"]["cliche_hits"])

    def test_runaway_sentence_fails(self):
        verdict = gate.check("这是一个非常长的句子" * 40 + "。", self.zh)
        self.assertFalse(verdict["ok"])
        self.assertFalse(verdict["checks"]["no_runaway_sentence"])

    def test_wrong_language_fails(self):
        verdict = gate.check(GOOD_EN, self.zh)
        self.assertFalse(verdict["checks"]["language"])

    def test_empty_fails(self):
        verdict = gate.check("  ", self.zh)
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["checks"], {"non_empty": False})

    def test_band_ignores_short_fragments(self):
        # "好。" carries no rhythm and must not drag the mean down.
        band = gate.band_for_docs("zh", [_doc("a", "好。" + GOOD_ZH)] * 12)["overall"]
        self.assertGreater(band["mean_min"], gate.MIN_SENTENCE_CHARS)

    def test_split_appendix_takes_one_trailing_line(self):
        body, aside = gate.split_appendix(GOOD_ZH + "\n这和《另一张》是同一个机制：…")
        self.assertEqual(body, GOOD_ZH)
        self.assertTrue(aside.startswith("这和《"))

    def test_split_appendix_leaves_paragraph_joins_alone(self):
        text = "第一段。\n\n第二段。"
        self.assertEqual(gate.split_appendix(text), (text, ""))
        self.assertEqual(gate.split_appendix(GOOD_ZH), (GOOD_ZH, ""))


def _doc(doc_id: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(id=doc_id, answer_for=lambda: text)


class FakeEngine:
    def __init__(self, reply: str, source: str = "pkg:card-1"):
        self.reply, self.source = reply, source
        self.turn_index = 1

    def turn(self, text, **kwargs):
        self.turn_index += 1
        return SimpleNamespace(reply=self.reply, source=self.source)


class TestAuditInstall(unittest.TestCase):
    def test_logs_without_changing_the_result(self):
        engine = FakeEngine(GOOD_ZH)
        with tempfile.TemporaryDirectory() as tmp:
            path = style_audit.install(
                engine, "pkg", "zh",
                SimpleNamespace(docs=[_doc("a", GOOD_ZH)] * 12),
                directory=Path(tmp))
            result = engine.turn("随便问问")
            self.assertEqual(result.reply, GOOD_ZH)
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(record["ok"])
        self.assertEqual(record["source"], "pkg:card-1")
        self.assertEqual(record["exempt"], "")

    def test_fallback_is_exempt_from_the_length_band(self):
        engine = FakeEngine("嗯……这个问题我得再翻翻记忆库。", source="fallback:zh")
        with tempfile.TemporaryDirectory() as tmp:
            path = style_audit.install(
                engine, "pkg", "zh",
                SimpleNamespace(docs=[_doc("a", GOOD_ZH)] * 12),
                directory=Path(tmp))
            engine.turn("今天股票怎么买")
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(record["ok"])
        self.assertEqual(record["exempt"], "fallback")
        self.assertNotIn("mean_in_band", record["failed"])

    def test_appendix_is_not_measured_as_voice(self):
        long_aside = "The premise behind this sits in \"Renormalization\": " + \
                     "a single very long unpunctuated clause " * 12
        engine = FakeEngine(GOOD_EN + "\n" + long_aside, source="pkg:card-1")
        with tempfile.TemporaryDirectory() as tmp:
            path = style_audit.install(
                engine, "pkg", "en",
                SimpleNamespace(docs=[_doc("a", GOOD_EN)] * 12),
                directory=Path(tmp))
            engine.turn("what is a photon")
            record = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
        self.assertTrue(record["ok"])
        self.assertEqual(record["appendix_chars"], len(long_aside))
        self.assertLessEqual(record["metrics"]["max_sentence_chars"],
                             max(len(s) for s in gate.sentences(GOOD_EN, "en")))

    def test_measurement_failure_never_costs_the_reply(self):
        engine = FakeEngine(GOOD_ZH)
        with tempfile.TemporaryDirectory() as tmp:
            original = engine.turn
            style_audit.install(engine, "pkg", "zh",
                                SimpleNamespace(docs=[_doc("a", GOOD_ZH)] * 12),
                                directory=Path(tmp))
            # A reply the gate cannot measure is still returned unchanged.
            engine.reply = None
            result = engine.turn("x")
        self.assertIsNone(result.reply)
        self.assertIsNot(engine.turn, original)

    def test_no_cards_means_no_hook(self):
        engine = FakeEngine(GOOD_ZH)
        path = style_audit.install(engine, "pkg", "zh",
                                   SimpleNamespace(docs=[]))
        self.assertIsNone(path)
        self.assertNotIn("turn", vars(engine))


class TestRuntimeFlag(unittest.TestCase):
    def test_off_by_default_and_on_when_flagged(self):
        from app import runtime
        saved = {k: os.environ.get(k) for k in
                 (style_audit.FLAG_ENV, "PYBOT_STATE_DIR",
                  style_audit.DIR_ENV)}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["PYBOT_STATE_DIR"] = tmp
                os.environ.pop(style_audit.FLAG_ENV, None)
                os.environ.pop(style_audit.DIR_ENV, None)
                engine = runtime.build("feynman-zh", personas_root=PERSONAS,
                                       enable_bridge=False)
                self.assertNotIn("turn", vars(engine))

                os.environ[style_audit.FLAG_ENV] = "1"
                engine = runtime.build("feynman-zh", personas_root=PERSONAS,
                                       enable_bridge=False)
                self.assertIn("turn", vars(engine))
                engine.turn("什么是量子电动力学？", now=1000.0)
                log = Path(tmp) / "style_audit" / "feynman-zh.jsonl"
                self.assertTrue(log.is_file())
                json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
