"""Typed association edges between cards.

A retrieval score says *how much* a card matches. It cannot say *why* two
cards belong together, and it has no way to warn "this looks like the other
one but isn't". This module stores that missing layer as explicit edges:

* every edge has a kind and, once human-confirmed, a one-sentence
  ``mechanism`` (what the two share) and ``difference`` (where they part);
* machine-mined edges (``origin="cooccur"``) are candidates in quarantine.
  They are stored, inspectable and never spoken — being nearby is attention,
  not evidence;
* at most one edge is ever spoken per answer, it is chosen by kind priority,
  and it never changes which card answered (the caller keeps its source).

The data file lives with the engine, not in the public corpus: it links card
ids and encodes judgement, the same way ``data/kb_en`` overlays translations.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

# Spoken kinds, strongest first. "same_domain" is filing, not explanation;
# it stays in the data so the graph is complete, but it is never said aloud.
SPOKEN = ("same_mechanism", "boundary", "premise")
KINDS = SPOKEN + ("same_domain",)
ORIGINS = ("human", "cooccur")

# One sentence per answer. A long line stops being a remark and becomes a
# link dump. English keeps the same rule but English titles are longer, so
# the character budget differs; the branches never share a budget.
MAX_SENTENCE_CHARS = 120
MAX_SENTENCE_CHARS_EN = 420

DEFAULT_PATH = (Path(__file__).resolve().parent / "data" / "assoc"
                / "assoc_edges.jsonl")

_ZH_STOP = "。"
_CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    kind: str = "same_domain"
    mechanism: str = ""
    difference: str = ""
    origin: str = "human"
    mechanism_en: str = ""
    difference_en: str = ""

    def to_row(self) -> dict:
        row = {"a": self.a, "b": self.b, "kind": self.kind,
               "origin": self.origin}
        for key in ("mechanism", "difference", "mechanism_en",
                    "difference_en"):
            value = getattr(self, key)
            if value:
                row[key] = value
        return row

    @property
    def key(self) -> str:
        return f"{self.a}->{self.b}:{self.kind}"


def problems(edge: Edge) -> List[str]:
    """Everything wrong with one edge; empty means it may be stored."""
    out: List[str] = []
    if not edge.a or not edge.b:
        out.append("missing endpoint")
    if edge.a == edge.b:
        out.append("self loop")
    if edge.kind not in KINDS:
        out.append(f"unknown kind {edge.kind!r}")
    if edge.origin not in ORIGINS:
        out.append(f"unknown origin {edge.origin!r}")
    if edge.origin == "human" and edge.kind in SPOKEN:
        if not edge.mechanism.strip():
            out.append("spoken edge needs a mechanism")
        if edge.kind in ("same_mechanism", "boundary") \
                and not edge.difference.strip():
            out.append(f"{edge.kind} needs the strongest difference")
    for field_name in ("mechanism", "difference"):
        value = getattr(edge, field_name)
        if _ZH_STOP in value:
            out.append(f"{field_name} must be one sentence")
    # The two languages are parallel branches, not a translation pair in
    # either direction: a spoken edge carries prose for both, and the English
    # prose may never borrow Chinese characters.
    if edge.origin == "human" and edge.kind in SPOKEN:
        if not edge.mechanism_en.strip():
            out.append("spoken edge needs an English mechanism")
        if edge.kind in ("same_mechanism", "boundary") \
                and not edge.difference_en.strip():
            out.append(f"{edge.kind} needs the English difference")
    for field_name in ("mechanism_en", "difference_en"):
        value = getattr(edge, field_name)
        if _CJK.search(value):
            out.append(f"{field_name} must be English")
    return out


class AssocGraph:
    def __init__(self, edges: Iterable[Edge], dropped: int = 0,
                 path: Optional[Path] = None):
        self.edges: List[Edge] = list(edges)
        self.dropped = dropped
        self.path = path
        self._by_card: Dict[str, List[Edge]] = {}
        for edge in self.edges:
            self._by_card.setdefault(edge.a, []).append(edge)
            self._by_card.setdefault(edge.b, []).append(edge)

    def of_card(self, card_id: str) -> List[Edge]:
        return list(self._by_card.get(card_id, ()))

    def confirmed(self, card_id: str) -> List[Edge]:
        return [e for e in self.of_card(card_id) if e.origin == "human"]

    def candidates(self, card_id: str) -> List[Edge]:
        return [e for e in self.of_card(card_id) if e.origin == "cooccur"]

    def sentence(self, card_id: str, language: str = "zh",
                 titles: Optional[Mapping[str, str]] = None,
                 exclude: Iterable[str] = ()) -> Optional[str]:
        """At most one confirmed line about ``card_id``, or None.

        Kind priority is fixed (same_mechanism, boundary, premise); ties keep
        file order, so the first hand-written edge wins deterministically.
        ``exclude`` names cards the answer is already about (the caller
        passes the other top hits) — never explain what the user asked for.
        """
        skip: Set[str] = set(exclude)
        titles = titles or {}
        for kind in SPOKEN:
            for edge in self.confirmed(card_id):
                if edge.kind != kind:
                    continue
                other = edge.b if edge.a == card_id else edge.a
                # A boundary edge exists to say "this is not that". It must
                # speak even when the other card is the runner-up — that is
                # exactly the confusion it was written for.
                if other in skip and kind != "boundary":
                    continue
                if kind != "same_mechanism" and edge.a != card_id:
                    continue  # premise/boundary point from conclusion to its support
                line = _render(edge, card_id, other, kind, language, titles)
                cap = (MAX_SENTENCE_CHARS_EN if language == "en"
                       else MAX_SENTENCE_CHARS)
                if line and len(line) <= cap:
                    return line
        return None


def _text(edge: Edge, key: str, language: str) -> str:
    if language == "en":
        return getattr(edge, f"{key}_en") or ""
    return getattr(edge, key) or ""


def _title(titles: Mapping[str, str], card_id: str) -> str:
    return titles.get(card_id) or card_id


def _render(edge: Edge, card_id: str, other: str, kind: str,
            language: str, titles: Mapping[str, str]) -> Optional[str]:
    mechanism = _text(edge, "mechanism", language)
    difference = _text(edge, "difference", language)
    if language == "en":
        if not mechanism or (kind in ("same_mechanism", "boundary")
                             and not difference):
            return None
        name = _title(titles, other)
        if _CJK.search(name):
            return None  # no English name, so no English line: never mix
        if kind == "same_mechanism":
            return (f"This is the same mechanism as \"{name}\": "
                    f"{mechanism}. Where it parts: {difference}")
        if kind == "boundary":
            return f"Don't confuse it with \"{name}\": {difference}"
        return f"The premise behind this sits in \"{name}\": {mechanism}"
    name = _title(titles, other)
    if kind == "same_mechanism":
        return f"这和《{name}》是同一个机制：{mechanism}。差别是：{difference}"
    if kind == "boundary":
        return f"别和《{name}》混了：{difference}"
    return f"这条结论的前提在《{name}》：{mechanism}"


def load(path: Optional[Path] = None) -> AssocGraph:
    """Read the JSONL edge file; invalid rows are dropped and counted."""
    file = Path(path) if path else DEFAULT_PATH
    edges: List[Edge] = []
    dropped = 0
    if not file.exists():
        return AssocGraph([], dropped=0, path=file)
    for line in file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        edge = Edge(
            a=str(row.get("a", "")),
            b=str(row.get("b", "")),
            kind=str(row.get("kind", "same_domain")),
            mechanism=str(row.get("mechanism", "")),
            difference=str(row.get("difference", "")),
            origin=str(row.get("origin", "human")),
            mechanism_en=str(row.get("mechanism_en", "")),
            difference_en=str(row.get("difference_en", "")),
        )
        if problems(edge):
            dropped += 1
            continue
        edges.append(edge)
    return AssocGraph(edges, dropped=dropped, path=file)
