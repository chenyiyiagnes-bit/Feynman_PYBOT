"""Cubie-table gains survive an override.

``override_triggers`` replaces which patterns fire. The gain stays the one
written on the matching cubie-table row. A pattern the table does not
contain keeps 1.0. These cases go through ``app.runtime.build`` so a
compile-time default of 1.0 cannot hide behind ``get_persona``.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.runtime import STATE_DIR_ENV, build
from omega_mind import review

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "personas"


def _strengths(result) -> dict:
    return {act.name: act.strength for act in result.activations}


def _pending_aliases(engine) -> set:
    """Aliases waiting for review.

    A managed package gates user-taught phrasings: praise queues the phrase,
    and only an approved decision turns it into a live alias.
    """
    data = review.read(engine.kb.learned)
    return {(row.get("card_id"), row.get("phrase"))
            for row in data.get("candidates", {}).values()
            if row.get("kind") == "alias"}


class TestCubieGains(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {STATE_DIR_ENV: self._tmp.name})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self._tmp.cleanup()

    def mind(self, package: str):
        # Each mind gets its own learned dir. A shared state directory would
        # carry an alias from the previous utterance into the next mind.
        slot = tempfile.mkdtemp(dir=self._tmp.name)
        os.environ[STATE_DIR_ENV] = slot
        return build(package, personas_root=PERSONAS, enable_bridge=False)

    def test_feynman_chinese_strengths_stay_put(self):
        engine = self.mind("feynman-zh")
        laugh = engine.turn("哈哈", now=1000.0)
        strengths = _strengths(laugh)
        # Replacement, not an appended English pattern: humor stays 0.8.
        self.assertAlmostEqual(strengths["edge_humor"], 0.8)
        self.assertAlmostEqual(strengths["edge_habit"], 0.2)
        self.assertEqual(engine.cube.history, ["R"])

        engine = self.mind("feynman-zh")
        murmur = engine.turn("嗯", now=1000.0)
        self.assertAlmostEqual(_strengths(murmur)["edge_habit"], 1.0)
        self.assertEqual(engine.cube.history, ["F"])

        engine = self.mind("feynman-zh")
        before = engine.state.get("dopamine")
        wrong = engine.turn("你说错了", now=1000.0)
        strengths = _strengths(wrong)
        self.assertAlmostEqual(strengths["core_intuition"], 1.0)
        self.assertNotIn("corner_shame", strengths)
        self.assertGreater(engine.state.get("dopamine") - before, 0.15)
        self.assertEqual(engine.cube.history, ["R"])

        engine = self.mind("feynman-zh")
        asked = engine.turn("为什么光会衍射", now=1000.0)
        self.assertAlmostEqual(_strengths(asked)["core_logic"], 0.6)
        self.assertEqual(engine.cube.history, [])
        self.assertIn("wave-interference-diffraction", asked.source)

    def test_feynman_praise_writes_an_alias_and_a_tint_does_not(self):
        engine = self.mind("feynman-zh")
        engine.turn("原子是什么？", now=1000.0)
        engine.turn("哈哈", now=1020.0)
        self.assertIn(("flp-i-01", "原子是什么？"), _pending_aliases(engine))

        engine = self.mind("feynman-zh")
        engine.turn("原子是什么？", now=1000.0)
        engine.turn("ok", now=1020.0)
        self.assertEqual(engine.kb.aliases, {})

    def test_english_feynman_hears_its_own_language(self):
        # One package, one language: Chinese laughter is not this mind's humor.
        engine = self.mind("feynman-en")
        chinese = engine.turn("哈哈", now=1000.0)
        self.assertNotIn("edge_humor", _strengths(chinese))
        self.assertAlmostEqual(_strengths(chinese)["edge_habit"], 0.2)

        engine = self.mind("feynman-en")
        engine.turn("Do you like painting?", now=1000.0)
        haha = engine.turn("haha", now=1020.0)
        strengths = _strengths(haha)
        self.assertGreaterEqual(strengths["edge_humor"], 0.5)
        self.assertAlmostEqual(strengths["edge_habit"], 0.2)
        self.assertEqual(engine.cube.history, ["R"])
        self.assertIn(("life-art-001", "Do you like painting?"),
                      _pending_aliases(engine))

        engine = self.mind("feynman-en")
        engine.turn("Do you like painting?", now=1000.0)
        amazing = engine.turn("that's amazing", now=1020.0)
        self.assertGreaterEqual(_strengths(amazing)["edge_humor"], 0.5)
        self.assertEqual(engine.cube.history, ["R"])
        self.assertIn(("life-art-001", "Do you like painting?"),
                      _pending_aliases(engine))

        for text in ("ok", "hmm", "hi"):
            engine = self.mind("feynman-en")
            result = engine.turn(text, now=1000.0)
            strengths = _strengths(result)
            self.assertAlmostEqual(strengths["edge_habit"], 0.2, msg=text)
            self.assertNotIn("edge_humor", strengths, text)
            self.assertEqual(engine.cube.history, [], text)

        shame = ("(looks away) ", "(voice hardens) ")
        for text in ("you're wrong", "I don't know", "I don't understand"):
            engine = self.mind("feynman-en")
            before = engine.state.get("dopamine")
            result = engine.turn(text, now=1000.0)
            strengths = _strengths(result)
            self.assertAlmostEqual(strengths["core_intuition"], 1.0, msg=text)
            self.assertNotIn("corner_shame", strengths, text)
            self.assertEqual(engine.cube.history, ["R"], text)
            self.assertGreater(engine.state.get("dopamine") - before, 0.15, text)
            self.assertFalse(any(p in shame for p in result.directive.prefixes), text)

    def test_laozhuang_uses_author_gains(self):
        for package in ("laozhuang-zh", "laozhuang-en"):
            with self.subTest(package=package):
                engine = self.mind(package)
                laugh = engine.turn("哈哈", now=1000.0)
                strengths = _strengths(laugh)
                self.assertAlmostEqual(strengths["edge_habit"], 0.2)
                self.assertAlmostEqual(strengths["edge_humor"], 0.8)
                self.assertEqual(engine.cube.history, ["R"])

                shame_zh = ("（别过头）", "（语气硬了一点）")
                shame_en = ("(looks away) ", "(voice hardens) ")
                engine = self.mind(package)
                wrong = engine.turn("你刚才说错了", now=1000.0)
                strengths = _strengths(wrong)
                self.assertAlmostEqual(strengths["core_intuition"], 0.6)
                self.assertNotIn("corner_shame", strengths)
                self.assertLess(wrong.resonance["energy"], 2.25)
                self.assertFalse(wrong.resonance["breakthrough"])
                self.assertFalse(any(p in shame_zh + shame_en
                                     for p in wrong.directive.prefixes))
                self.assertNotEqual(engine.cube.history, ["R", "R", "F", "R'"])

                engine = self.mind(package)
                told = engine.turn("you're wrong", now=1000.0)
                strengths = _strengths(told)
                self.assertAlmostEqual(strengths["core_intuition"], 0.6)
                self.assertNotIn("corner_shame", strengths)
                self.assertFalse(any(p in shame_zh + shame_en
                                     for p in told.directive.prefixes))

                engine = self.mind(package)
                putdown = engine.turn("nonsense", now=1000.0)
                self.assertAlmostEqual(_strengths(putdown)["corner_shame"], 0.9)
                self.assertTrue(any(p in shame_zh + shame_en
                                    for p in putdown.directive.prefixes))

                for text in ("嗯", "ok", "hmm"):
                    engine = self.mind(package)
                    murmur = engine.turn(text, now=1000.0)
                    self.assertAlmostEqual(
                        _strengths(murmur)["edge_habit"], 1.0, msg=text)

                for text in ("why", "为什么光会衍射"):
                    engine = self.mind(package)
                    asked = engine.turn(text, now=1000.0)
                    strengths = _strengths(asked)
                    self.assertAlmostEqual(strengths["core_logic"], 0.6, msg=text)
                    self.assertEqual(engine.cube.history, [], msg=text)

                engine = self.mind(package)
                defined = engine.turn("what is the definition of the dao", now=1000.0)
                self.assertNotIn("corner_pedantry", _strengths(defined))

                engine = self.mind(package)
                fine = engine.turn("it seems fine", now=1000.0)
                self.assertNotIn("core_intuition", _strengths(fine))

                engine = self.mind(package)
                interesting = engine.turn("why is that interesting", now=1000.0)
                self.assertNotIn("edge_humor", _strengths(interesting))

                engine = self.mind(package)
                praised = engine.turn("that's interesting", now=1000.0)
                self.assertAlmostEqual(_strengths(praised)["edge_humor"], 0.8)

                engine = self.mind(package)
                lost = engine.turn("I don't understand", now=1000.0)
                self.assertAlmostEqual(_strengths(lost)["core_intuition"], 0.6)
                self.assertEqual(engine.cube.history, [])

                for text in ("the experiment failed", "why did the experiment fail"):
                    engine = self.mind(package)
                    failed = engine.turn(text, now=1000.0)
                    self.assertAlmostEqual(
                        _strengths(failed)["edge_trauma"], 1.2, msg=text)

                engine = self.mind(package)
                missed = engine.turn("he failed the exam", now=1000.0)
                self.assertNotIn("edge_trauma", _strengths(missed))


if __name__ == "__main__":
    unittest.main()
