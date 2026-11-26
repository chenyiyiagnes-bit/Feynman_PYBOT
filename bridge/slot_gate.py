"""A slot may only be filled with a word the chosen card actually holds.

The typesetter can take a noun out of the user's question and put it into the
entity slot. That is fine until the word is not about the card at all: the
sentence then sounds like the answer talked about something the source never
mentioned. This gate is the machine check — every user-supplied word in a
filled slot must appear, as a token, in one of the fields of the card the
answer came from.

It is inactive when there is no coded card file (a fresh clone has none), and
inactive when the caller does not say which card was chosen. Both are honest
"cannot verify", and both mean the gate does not block anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


class CodedCards:
    """Token strings of every card, keyed by card id."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.words: Dict[str, Set[str]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            words: Set[str] = set()
            for field in row.get("fields", ()):
                for token in field.get("tokens", ()):
                    if isinstance(token, dict) and token.get("t"):
                        words.add(str(token["t"]))
            self.words[str(row["id"])] = words

    def words_of(self, card_id: str) -> Optional[Set[str]]:
        return self.words.get(str(card_id))

    def unverified(self, words: Iterable[str], card_id: str) -> List[str]:
        """Words that are not in the card; empty when the card is unknown."""
        known = self.words_of(card_id)
        if known is None:
            return []
        out: List[str] = []
        for word in words:
            word = str(word)
            if not word:
                continue
            if word in known:
                continue
            # A word may be stored as part of a longer coded token
            # ("超导磁体" inside a card that only had the phrase).
            if any(word in token for token in known):
                continue
            out.append(word)
        return out


_CACHE: Dict[str, Optional[CodedCards]] = {}


def load(lexicon_dir: Optional[str | Path]) -> Optional[CodedCards]:
    if lexicon_dir is None:
        return None
    path = Path(lexicon_dir) / "cards.coded.jsonl"
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = CodedCards(path) if path.is_file() else None
    return _CACHE[key]


def user_slot_words(result: dict) -> List[str]:
    """Words the typesetter took from the user rather than from the vault."""
    out: List[str] = []
    for sentence in (result or {}).get("sentences", ()):
        glyphs = sentence.get("glyphs") if isinstance(sentence, dict) else None
        if not isinstance(glyphs, dict):
            continue
        for slot in ("hook", "entity", "action", "consequence"):
            glyph = glyphs.get(slot)
            if isinstance(glyph, dict) and glyph.get("source") == "user":
                word = str(glyph.get("word") or "").strip()
                if word and word not in out:
                    out.append(word)
    return out


def unverified(result: dict, card_id: Optional[str], lexicon_dir) -> List[str]:
    """User words in the paragraph that the chosen card cannot account for."""
    cards = load(lexicon_dir)
    if cards is None or not card_id:
        return []
    return cards.unverified(user_slot_words(result), card_id)
