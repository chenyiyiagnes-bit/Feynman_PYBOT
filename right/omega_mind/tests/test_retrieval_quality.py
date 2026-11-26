"""Golden retrieval set: known question -> acceptable card, and known traps.

This is the regression net for the retrieval weights. When the corpus or a
threshold changes, these are the cases that must not silently flip. Positive
cases allow a small set of card ids where two cards genuinely answer the same
question; negative cases must stay unanswered.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.feynman_kb import FeynmanKB
from omega_mind.golden_queries import nets_by_package


def real_kb(language: str = "zh") -> FeynmanKB:
    """One package, one language: the golden set is split the same way."""
    kb = FeynmanKB(language=language)
    if not kb.available() or kb.language != language:
        raise unittest.SkipTest(f"no {language!r} knowledge package installed")
    return kb


def package_kb(name: str) -> FeynmanKB:
    """The package's own KB, over an empty learned dir.

    Each net's rows name cards from that package's own index, so they can
    only be checked against that package's KB — a Feynman KB cannot answer a
    Zhuangzi question, and answering the Zhuangzi traps would be leakage.
    A throwaway learned dir keeps the result independent of whatever a
    developer machine happened to teach into the shared state directory.
    """
    kb = FeynmanKB.for_package(name, learned_dir=tempfile.mkdtemp(prefix="omega-golden-"))
    if kb is None or not kb.available():
        raise unittest.SkipTest(f"package {name!r} is not installed")
    return kb


class TestGoldenRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nets = nets_by_package()
        cls.kbs = {}
        for name in cls.nets:
            try:
                cls.kbs[name] = package_kb(name)
            except unittest.SkipTest:
                continue

    def kb_for_package(self, name: str) -> FeynmanKB:
        if name not in self.kbs:
            raise unittest.SkipTest(f"package {name!r} is not installed")
        return self.kbs[name]

    def test_known_questions_hit_the_right_card(self):
        for name, (golden, _) in self.nets.items():
            for query, language, acceptable in golden:
                with self.subTest(package=name, query=query):
                    kb = self.kb_for_package(name)
                    self.assertEqual(kb.language, language)
                    _, source = kb.answer(query)
                    root = kb.source_root + ":"
                    self.assertTrue(source.startswith(root),
                                    f"{query!r} -> {source} (not the package's own index)")
                    card = source[len(root):]
                    self.assertIn(card, acceptable,
                                  f"{query!r} -> {card}, expected {acceptable}")

    def test_other_language_is_a_miss_not_a_fallback(self):
        zh = real_kb("zh")
        text, source = zh.answer("what is diffraction", language="en")
        self.assertIsNone(text, source)
        self.assertTrue(source.endswith(":no_match"), source)

    def test_traps_stay_unanswered(self):
        for name, (_, traps) in self.nets.items():
            for query, language in traps:
                with self.subTest(package=name, query=query):
                    kb = self.kb_for_package(name)
                    self.assertEqual(kb.language, language)
                    text, source = kb.answer(query)
                    self.assertIsNone(text, f"{query!r} answered with {source}")
                    self.assertTrue(source.endswith(":no_match"), source)

    def test_lookup_reports_margin_and_limit(self):
        wide = real_kb("zh").lookup("什么是阻尼", limit=3)
        self.assertIsNotNone(wide.margin)
        self.assertLess(wide.margin, 0.05)
        narrow = real_kb("zh").lookup("什么是阻尼", limit=1)
        self.assertIsNone(narrow.margin)
        self.assertEqual(narrow.considered, 1)

    def test_lookup_exposes_score_and_source(self):
        found = real_kb("zh").lookup("什么是扩散")
        self.assertEqual(found.source, "feynman_kb:C_DIFFUSE")
        self.assertGreater(found.score, 0.3)
        self.assertEqual(found.to_dict()["source"], "feynman_kb:C_DIFFUSE")


if __name__ == "__main__":
    unittest.main()


class TestNarrativeTie(unittest.TestCase):
    """An exact tie between a physics card and a story goes to physics.

    ``bio-023`` is a one-line biography ("部分子模型") and ``part-002`` is the
    card that explains it. Their scores tie exactly; without the kind
    tie-break the shorter card wins on sort order and the user gets a
    label instead of an explanation.
    """

    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()

    def test_equal_score_prefers_the_explanation(self):
        hits = self.kb.search("部分子是什么", limit=3)
        self.assertGreaterEqual(len(hits), 2)
        (s0, d0), (s1, d1) = hits[0], hits[1]
        self.assertAlmostEqual(s0, s1, places=9)
        self.assertEqual({d0.id, d1.id}, {"bio-023", "part-002"})
        self.assertEqual(d0.id, "part-002")

    def test_biography_still_wins_when_it_scores_higher(self):
        found = self.kb.lookup("费曼是谁")
        self.assertEqual(found.source, "feynman_kb:self-001")


class TestCardSelfRetrieval(unittest.TestCase):
    """Every card should at least find *a* card when asked its own title.

    Exact self-match is the stronger property, but a handful of near-duplicate
    cards (Caltech, Feynman diagrams) legitimately answer each other, so the
    exact rate is held at 95% while full coverage must be 100%.
    """

    @classmethod
    def setUpClass(cls):
        cls.kb = real_kb()
        cls.titled = [d for d in cls.kb.docs if d.title.strip()]

    def test_every_title_returns_an_answer(self):
        unanswered = [d.id for d in self.titled
                      if self.kb.lookup(d.title).text is None]
        self.assertEqual(unanswered, [])

    def test_most_titles_return_their_own_card(self):
        # One-line biography cards ("部分子模型") are deliberately allowed to
        # lose a subject question to the domain card that explains it, so the
        # exact-match rate is measured on the explanatory cards.
        solid = [d for d in self.titled
                 if d.kind not in ("biography", "persona_memory")]
        exact, near = [], []
        for doc in solid:
            source = self.kb.lookup(doc.title).source
            card = source.split(":", 1)[1]
            (exact if card == doc.id else near).append((doc.id, card))
        self.assertGreaterEqual(len(exact) / len(solid), 0.95, near)

    def test_teachable_units_find_themselves(self):
        for unit in self.kb.teachable_units():
            with self.subTest(unit=unit.id):
                found = self.kb.lookup(unit.title)
                self.assertEqual(found.source.split(":", 1)[1], unit.id)
