"""The slot gate: a user's word may only fill a slot its card accounts for.

Inactive means "cannot verify" (no coded data, no chosen card) and blocks
nothing — a fresh checkout has no coded data at all. Active means a word the
typesetter took from the question must appear among the tokens of the card
the answer came from.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "bridge", ROOT / "right"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from bridge import slot_gate  # noqa: E402
from bridge.responder import LeftResponder  # noqa: E402
from bridge.persona import load as load_persona  # noqa: E402

CODED = [
    {"id": "cooper", "fields": [
        {"field": "title", "tokens": [{"id": 0, "t": "库珀对", "start": 0, "end": 3}]},
        {"field": "text", "tokens": [{"id": 1, "t": "电子", "start": 0, "end": 2},
                                     {"id": 2, "t": "配对", "start": 2, "end": 4}]},
    ]},
    {"id": "rainbow", "fields": [
        {"field": "title", "tokens": [{"id": 3, "t": "彩虹", "start": 0, "end": 2}]},
    ]},
]


def write_coded(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cards.coded.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CODED),
        encoding="utf-8")
    return directory


def result_with(*words: str) -> dict:
    glyphs = {slot: {"word": word, "source": "user"}
              for slot, word in zip(("entity", "action"), words)}
    return {"sentences": [{"glyphs": glyphs}]}


class Gate(unittest.TestCase):
    def test_inactive_without_coded_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(slot_gate.load(Path(tmp)))
            self.assertEqual(slot_gate.unverified(result_with("量子"), "cooper", tmp), [])
            self.assertEqual(slot_gate.unverified(result_with("量子"), "cooper", None), [])

    def test_a_word_the_card_holds_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            self.assertEqual(slot_gate.unverified(result_with("电子", "配对"), "cooper",
                                                  directory), [])
            self.assertEqual(slot_gate.unverified(result_with("库珀对"), "cooper",
                                                  directory), [])

    def test_a_word_the_card_lacks_is_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            stray = slot_gate.unverified(result_with("电子", "股票"), "cooper", directory)
            self.assertEqual(stray, ["股票"])

    def test_a_word_inside_a_longer_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            self.assertEqual(slot_gate.unverified(result_with("库珀"), "cooper", directory), [])

    def test_unknown_card_is_not_a_veto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            self.assertEqual(slot_gate.unverified(result_with("股票"), "no-such-card",
                                                  directory), [])

    def test_only_user_words_are_checked(self) -> None:
        result = {"sentences": [{"glyphs": {
            "entity": {"word": "股票", "source": "vault"},
            "action": {"word": "电子", "source": "user"},
        }}]}
        self.assertEqual(slot_gate.user_slot_words(result), ["电子"])


class Responder(unittest.TestCase):
    def responder(self, lexicon) -> LeftResponder:
        return LeftResponder(enabled=True, lexicon_dir=lexicon)

    def stub_modules(self, responder: LeftResponder, paragraph_result: dict) -> None:
        request_cls = SimpleNamespace(
            OperatorWeights=lambda **kwargs: kwargs,
            ParagraphRequest=lambda **kwargs: kwargs,
        )
        weaver = SimpleNamespace(weave_paragraph=lambda vault, request: paragraph_result,
                                 ParagraphRequest=request_cls.ParagraphRequest)
        responder._weaver = weaver
        responder._request_cls = request_cls
        responder._vault = object()

    def persona(self):
        return load_persona(ROOT / "personas" / "feynman-zh" / "persona.json")

    def test_gate_is_inactive_without_coded_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self.responder(tmp).gate_active)

    def test_paragraph_with_a_stray_word_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            responder = self.responder(directory)
            self.assertTrue(responder.gate_active)
            self.stub_modules(responder, {"paragraph": "股票的配对从这里开始。",
                                          "sentences": result_with("股票")["sentences"]})
            responder.vault_path = Path(tmp) / "vault.json"
            responder.vault_path.write_text("{}", encoding="utf-8")
            out = responder.reply("股票怎么会成对", {}, self.persona(), card_id="cooper")
            self.assertIsNone(out)
            self.assertIn("股票", responder.last_error)

    def test_paragraph_with_a_supported_word_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_coded(Path(tmp))
            responder = self.responder(directory)
            self.stub_modules(responder, {"paragraph": "电子的配对从这里开始。",
                                          "sentences": result_with("电子")["sentences"]})
            responder.vault_path = Path(tmp) / "vault.json"
            responder.vault_path.write_text("{}", encoding="utf-8")
            out = responder.reply("电子怎么会成对", {}, self.persona(), card_id="cooper")
            self.assertIsNotNone(out)
            self.assertIn("电子", out.text)


class EngineWiring(unittest.TestCase):
    def test_engine_hands_the_card_to_the_gate(self) -> None:
        from omega_mind.engine import MindEngine
        import inspect
        source = inspect.getsource(MindEngine._left_reply)
        self.assertIn("gate_active", source)
        self.assertIn('"card_id": card_id', source)


if __name__ == "__main__":
    unittest.main()
