"""Characteristic magnitudes Feynman actually gives, keyed to cards.

An association edge says how two cards relate. A scale anchor says how big
the thing on the card is — in Feynman's own numbers, not in ours. It is a
separate, local overlay (``data/assoc/scales.jsonl``) for three reasons:

* a number is a claim, so every row carries the lecture it came from;
* the same card may need different anchors in different languages, and the
  English branch has to be Feynman's English, not a translation;
* anchors are spoken only when the question asks "how big / how fast / what
  order of magnitude", so an ordinary answer stays uncluttered.

Rows are ``{"card": id, "zh": ..., "en": ..., "source": "FLP I-1"}``.
``asks`` / ``asks_en`` may carry the question phrasings this anchor was
written for: retrieval alone often loses a magnitude question such as
"原子有多大" because the noun is too common to be discriminative, so the
curated phrasing is what puts the reader on the right card. A row without a
source, or with Chinese in the English line, is dropped on load.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PATH = (Path(__file__).resolve().parent / "data" / "assoc"
                / "scales.jsonl")

MAX_ZH_CHARS = 200
MAX_EN_CHARS = 700
_CJK = re.compile(r"[一-鿿]")
# Phrasings are compared without punctuation or spaces: "光速有多快？" and
# "光速有多快" are the same ask.
_NOISE = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’()（）]+")


@dataclass(frozen=True)
class Scale:
    card: str
    zh: str = ""
    en: str = ""
    source: str = ""
    asks: Sequence[str] = field(default_factory=tuple)
    asks_en: Sequence[str] = field(default_factory=tuple)

    def line(self, language: str = "zh") -> str:
        return self.en if language == "en" else self.zh

    def phrasings(self, language: str = "zh") -> Tuple[str, ...]:
        return tuple(self.asks_en if language == "en" else self.asks)


def problems(scale: Scale) -> List[str]:
    out: List[str] = []
    if not scale.card:
        out.append("scale has no card id")
    if not scale.zh:
        out.append(f"{scale.card}: no Chinese line")
    if not scale.en:
        out.append(f"{scale.card}: no English line")
    if scale.en and _CJK.search(scale.en):
        out.append(f"{scale.card}: English line contains Chinese")
    if not scale.source:
        out.append(f"{scale.card}: no source")
    if scale.zh and len(scale.zh) > MAX_ZH_CHARS:
        out.append(f"{scale.card}: Chinese line too long")
    if scale.en and len(scale.en) > MAX_EN_CHARS:
        out.append(f"{scale.card}: English line too long")
    for ask in scale.asks:
        if not _NOISE.sub("", ask):
            out.append(f"{scale.card}: empty Chinese ask")
    for ask in scale.asks_en:
        if _CJK.search(ask):
            out.append(f"{scale.card}: English ask contains Chinese")
    return out


class ScaleBook:
    def __init__(self, scales: Iterable[Scale] = (), dropped: int = 0,
                 path: Optional[Path] = None):
        self.scales: List[Scale] = list(scales)
        self.dropped = dropped
        self.path = path
        self._by_card: Dict[str, Scale] = {s.card: s for s in self.scales}

    def __len__(self) -> int:
        return len(self.scales)

    def card_ids(self) -> List[str]:
        return [s.card for s in self.scales]

    def line(self, card_id: str, language: str = "zh") -> Optional[str]:
        scale = self._by_card.get(card_id)
        if scale is None:
            return None
        text = scale.line(language)
        return text or None

    def anchor_for(self, text: str, language: str = "zh"
                   ) -> Optional[Tuple[str, str]]:
        """The card this magnitude question was written for, with the ask.

        Matching is deliberately literal: a curated phrasing must appear in
        the question after punctuation and spaces are removed. It cannot
        wander: no stemming, no synonyms, no partial characters.
        """
        query = _NOISE.sub("", text or "").casefold()
        if not query:
            return None
        for scale in self.scales:
            for ask in scale.phrasings(language):
                phrase = _NOISE.sub("", ask).casefold()
                if phrase and phrase in query:
                    return scale.card, ask
        return None


def load(path: Optional[Path] = None) -> ScaleBook:
    """Read the JSONL anchor file; invalid rows are dropped and counted."""
    file = Path(path) if path else DEFAULT_PATH
    scales: List[Scale] = []
    dropped = 0
    if not file.exists():
        return ScaleBook([], dropped=0, path=file)
    seen = set()
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        scale = Scale(
            card=str(row.get("card", "")),
            zh=str(row.get("zh", "")),
            en=str(row.get("en", "")),
            source=str(row.get("source", "")),
            asks=tuple(str(a) for a in row.get("asks") or ()),
            asks_en=tuple(str(a) for a in row.get("asks_en") or ()),
        )
        if problems(scale) or scale.card in seen:
            dropped += 1
            continue
        seen.add(scale.card)
        scales.append(scale)
    return ScaleBook(scales, dropped=dropped, path=file)
