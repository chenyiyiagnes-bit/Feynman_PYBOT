"""Renderer shell: prefixes join the body without eating a word.

The persona config supplies openers like "Look," or "Hmm." for English and
"等一下，" for Chinese. English openers need a space before the body; Chinese
ones do not. Getting this wrong produces "Hmm.Is losing always bad?", which a
user sees as a typo on the first turn of a session.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omega_mind.policy import StyleDirective
from omega_mind.renderer import Renderer


def directive(prefixes=None, endings=None, verbosity=0.8) -> StyleDirective:
    return StyleDirective(
        tone="neutral", warmth=0.5, directness=0.5, verbosity=verbosity,
        humor=0.3, tempo=1.0, expression="neutral_calm",
        prefixes=list(prefixes or []), endings=list(endings or []),
    )


class TestPrefixSpacing(unittest.TestCase):
    def test_english_prefix_is_followed_by_a_space(self):
        reply = Renderer().render("The thing about light is that it moves.",
                                  directive(prefixes=["Look,"]), turn=1,
                                  language="en")
        self.assertTrue(reply.startswith("Look, The"), reply)

    def test_english_prefix_ending_in_a_stop_gets_a_space(self):
        reply = Renderer().render("Losing is not always bad.",
                                  directive(prefixes=["Hmm."]), turn=1,
                                  language="en")
        self.assertTrue(reply.startswith("Hmm. Losing"), reply)

    def test_english_prefix_with_its_own_space_is_not_doubled(self):
        reply = Renderer().render("The thing about light is that it moves.",
                                  directive(prefixes=["Hold on, "]), turn=1,
                                  language="en")
        self.assertTrue(reply.startswith("Hold on, The"), reply)
        self.assertNotIn("  ", reply)

    def test_chinese_prefix_keeps_its_own_punctuation(self):
        reply = Renderer().render("光一直在走，没人知道为什么。",
                                  directive(prefixes=["等一下，"]), turn=1,
                                  language="zh")
        self.assertTrue(reply.startswith("等一下，光"), reply)


if __name__ == "__main__":
    unittest.main()
