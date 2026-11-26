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
ids and encodes judgement; each package keeps its own file under
``personas/<id>/data/assoc``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple)

# Spoken kinds, strongest first. "same_domain" is filing, not explanation;
# it stays in the data so the graph is complete, but it is never said aloud.
# "same_equation" outranks "same_mechanism": a shared equation with a symbol
# map says more than "these two are alike". "limit_of" is a derivation move —
# one theory is another one's degenerate case — and points from the special
# case to the general one.
SPOKEN = ("same_equation", "boundary", "tension", "limit_of",
          "same_mechanism", "premise")
KINDS = SPOKEN + ("same_domain",)
ORIGINS = ("human", "cooccur")

# Kinds that may speak from either endpoint. The rest point one way on
# purpose: premise/boundary/limit_of are written from the card they explain.
SYMMETRIC = ("same_mechanism", "same_equation")

# The order a person rethinking a subject would take its faces in: the
# ground it stands on, what it is not, where it stops, then the analogy.
# Reflection walks this order and skips what has already been said.
FACET_ORDER = ("premise", "boundary", "tension", "limit_of",
               "same_equation", "same_mechanism")

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
    language: str = ""
    # same_equation: matched symbols — left belongs to a, right to b.
    # Rows may carry the pairs as "x↔q、m↔L"; load() parses both forms.
    mapping: tuple[tuple[str, str], ...] = ()
    # Optional English rendering of the same pairs, when the symbols are
    # spelled out rather than written as letters.
    mapping_en: str = ""
    # limit_of: the small parameter or regime in which b turns into a.
    condition: str = ""
    condition_en: str = ""
    relation: str = ""
    source: str = ""
    misconceptions: tuple[str, ...] = ()

    def to_row(self) -> dict:
        row = {"a": self.a, "b": self.b, "kind": self.kind,
               "origin": self.origin}
        for key in ("mechanism", "difference", "mechanism_en",
                    "difference_en", "mapping", "mapping_en",
                    "condition", "condition_en"):
            value = getattr(self, key)
            if value:
                row[key] = value
        if self.language:
            row.update(language=self.language, mapping=[list(pair) for pair in self.mapping],
                       relation=self.relation, source=self.source)
            if self.misconceptions:
                row["misconceptions"] = list(self.misconceptions)
        return row

    @property
    def key(self) -> str:
        return f"{self.a}->{self.b}:{self.kind}"


_MAPPING_SPLIT = re.compile(r"[、,;]")


def _mapping_pairs(value) -> List[Tuple[str, str]]:
    """Parse a symbol mapping into pairs; unparseable pairs are skipped.

    Rows carry the mapping either as ``"x↔q、m↔L"`` or as a list of
    ``[left, right]`` pairs; both mean the same thing.
    """
    pairs: List[Tuple[str, str]] = []
    if not value:
        return pairs
    if isinstance(value, str):
        for pair in _MAPPING_SPLIT.split(value):
            pair = pair.strip()
            if not pair or pair.count("↔") != 1:
                continue
            left, right = (side.strip() for side in pair.split("↔"))
            if left and right:
                pairs.append((left, right))
        return pairs
    for pair in value:
        try:
            left, right = pair
        except (TypeError, ValueError):
            continue
        if str(left).strip() and str(right).strip():
            pairs.append((str(left).strip(), str(right).strip()))
    return pairs


def _mapping_problems(value, name: str) -> List[str]:
    out: List[str] = []
    if not value:
        return out
    if isinstance(value, str):
        for pair in _MAPPING_SPLIT.split(value):
            pair = pair.strip()
            if not pair:
                continue
            if pair.count("↔") != 1:
                out.append(f"{name} pair {pair!r} needs exactly one ↔")
                continue
            left, right = pair.split("↔")
            if not left.strip() or not right.strip():
                out.append(f"{name} pair {pair!r} has an empty side")
        return out
    for pair in value:
        if len(pair) != 2 or not all(str(side).strip() for side in pair):
            out.append(f"{name} pair {pair!r} needs two non-empty sides")
    return out


def mapping_text(edge: Edge, language: str = "zh",
                 invert: bool = False) -> str:
    """The mapping as people write it: "x↔q、m↔L" (swapped when inverted)."""
    if language == "en" and edge.mapping_en:
        return invert_mapping(edge.mapping_en, sep=", ") if invert \
            else edge.mapping_en
    pairs = _mapping_pairs(edge.mapping)
    if invert:
        pairs = [(right, left) for left, right in pairs]
    sep = ", " if language == "en" else "、"
    return sep.join(f"{left}↔{right}" for left, right in pairs)


def invert_mapping(value: str, sep: str = "、") -> str:
    """Swap both sides of every pair, for speaking the edge from b."""
    inverted: List[str] = []
    for pair in _MAPPING_PAIRS_STR(value):
        if pair.count("↔") != 1:
            inverted.append(pair)
            continue
        left, right = pair.split("↔")
        inverted.append(f"{right.strip()}↔{left.strip()}")
    return sep.join(inverted)


def _MAPPING_PAIRS_STR(value: str) -> List[str]:
    return [pair.strip() for pair in _MAPPING_SPLIT.split(value)
            if pair.strip()]


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
    if edge.language:
        if edge.language not in ("zh", "en"):
            out.append("unknown edge language")
        if edge.origin == "human" and edge.kind in SPOKEN:
            if not (edge.source.strip() and edge.relation.strip()
                    and edge.mapping):
                out.append("typed edge needs a source, a relation and matched variables")
            if len({a for a, _ in edge.mapping}) != len(edge.mapping) or len(
                    {b for _, b in edge.mapping}) != len(edge.mapping):
                out.append("typed edge variable mapping must be one to one")
    if edge.origin == "human" and edge.kind in SPOKEN and edge.language != "en":
        if not edge.mechanism.strip():
            out.append("spoken edge needs a mechanism")
        if edge.kind in ("same_mechanism", "same_equation", "boundary",
                         "tension") and not edge.difference.strip():
            out.append(f"{edge.kind} needs the strongest difference")
        if edge.kind == "same_equation" and not edge.mapping:
            out.append("same_equation needs a symbol mapping")
        if edge.kind == "limit_of" and not edge.condition.strip():
            out.append("limit_of needs the limiting condition")
    for field_name in ("mechanism", "difference", "condition"):
        value = getattr(edge, field_name)
        if _ZH_STOP in value:
            out.append(f"{field_name} must be one sentence")
    out.extend(_mapping_problems(edge.mapping, "mapping"))
    # The two languages are parallel branches, not a translation pair in
    # either direction: a spoken edge carries prose for both, and the English
    # prose may never borrow Chinese characters.
    if edge.origin == "human" and edge.kind in SPOKEN and edge.language != "zh":
        if not edge.mechanism_en.strip():
            out.append("spoken edge needs an English mechanism")
        if edge.kind in ("same_mechanism", "same_equation", "boundary") \
                and not edge.difference_en.strip():
            out.append(f"{edge.kind} needs the English difference")
        if edge.kind == "same_equation" and not (
                edge.mapping_en or edge.mapping):
            out.append("same_equation needs an English symbol mapping")
        if edge.kind == "limit_of" and not edge.condition_en.strip():
            out.append("limit_of needs the English limiting condition")
    for field_name in ("mechanism_en", "difference_en", "condition_en"):
        value = getattr(edge, field_name)
        if _CJK.search(value):
            out.append(f"{field_name} must be English")
    if _CJK.search(edge.mapping_en):
        out.append("mapping_en must be English")
    out.extend(_mapping_problems(edge.mapping_en, "mapping_en"))
    return out


@dataclass(frozen=True)
class Match:
    text: str
    edge: Edge

    def provenance(self) -> dict:
        edge = self.edge
        return {"id": edge.key, "from": edge.a, "to": edge.b,
                "kind": edge.kind, "origin": edge.origin,
                "mapping": [list(pair) for pair in edge.mapping],
                "relation": edge.relation, "source": edge.source}


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

    def premise_step(self, card_id: str, language: str = "zh",
                     avoid: Iterable[str] = ()) -> Optional[Tuple[str, str]]:
        """The next card down the why-chain, with the link that gets there.

        Only a confirmed premise edge written from ``card_id`` counts:
        premise points one way (a is explained by b), and a mined
        co-occurrence is attention, not an explanation. ``avoid`` keeps a
        chain from walking in a circle. Returns ``(card_id, link)`` or None.
        """
        skip: Set[str] = set(avoid)
        for edge in self.confirmed(card_id):
            if edge.kind != "premise" or edge.a != card_id:
                continue
            if edge.b in skip:
                continue
            link = _text(edge, "mechanism", language)
            if not link:
                continue
            if language == "en" and _CJK.search(link):
                continue
            return edge.b, link
        return None

    def has_premise(self, card_id: str) -> bool:
        """Is there a confirmed why-step below this card?

        Used by the curiosity question: it may invite the user one step down
        only when a real, hand-confirmed premise edge will be there.
        """
        return any(e.kind == "premise" and e.a == card_id
                   and e.b != card_id
                   for e in self.confirmed(card_id))

    def recast(self, card_id: str, language: str = "zh",
               titles: Optional[Mapping[str, str]] = None
               ) -> Optional[Tuple[str, str]]:
        """Say the card a second way, through a confirmed analogy.

        Used when the user signals "I don't get it": repeating the card is
        useless, so the mind recasts it through the nearest hand-confirmed
        ``same_mechanism``/``same_equation`` edge and states where the
        analogy breaks. Returns ``(line, edge key)``, or None when the card
        has no analogy with both halves written. Never invents the analogy.
        """
        titles = titles or {}
        for kind in ("same_mechanism", "same_equation"):
            for edge in self.confirmed(card_id):
                if edge.kind != kind:
                    continue
                if edge.a != card_id and edge.b != card_id:
                    continue
                other = edge.b if edge.a == card_id else edge.a
                mechanism = _text(edge, "mechanism", language)
                difference = _text(edge, "difference", language)
                mapping = _text(edge, "mapping", language)
                if not mechanism or not difference:
                    continue
                if kind == "same_equation" and not mapping:
                    continue
                name = _title(titles, other)
                if language == "en":
                    if (_CJK.search(name) or _CJK.search(mechanism)
                            or _CJK.search(difference)):
                        continue
                    tail = (f" Match the symbols: {mapping}." if kind ==
                            "same_equation" else "")
                    line = (f"Put another way — this is the same "
                            f"{'equation' if kind == 'same_equation' else 'mechanism'} "
                            f"as \"{name}\": {mechanism}.{tail} The analogy "
                            f"breaks here: {difference}")
                else:
                    if kind == "same_equation":
                        line = (f"换个说法——它和《{name}》是同一个方程："
                                f"{mechanism}；把符号对上：{mapping}。"
                                f"这个比方到这儿就不成立了：{difference}")
                    else:
                        line = (f"换个说法——它和《{name}》是同一个机制："
                                f"{mechanism}。这个比方到这儿就不成立了："
                                f"{difference}")
                cap = (MAX_SENTENCE_CHARS_EN * 2 if language == "en"
                       else MAX_SENTENCE_CHARS * 2)
                if len(line) <= cap:
                    return line, f"{edge.key}"
        return None

    def sentence(self, card_id: str, language: str = "zh",
                 titles: Optional[Mapping[str, str]] = None,
                 exclude: Iterable[str] = (),
                 continuity_with: Optional[str] = None,
                 **selection) -> Optional[str]:
        match = self.select(card_id, language, titles, exclude,
                            continuity_with=continuity_with, **selection)
        return match.text if match is not None else None

    def select(self, card_id: str, language: str = "zh",
                 titles: Optional[Mapping[str, str]] = None,
                 exclude: Iterable[str] = (), *,
                 kinds: Sequence[str] = SPOKEN,
                 spoken: Sequence[str] = (),
                 relevant: Optional[Callable[[str], bool]] = None,
                 continuity_with: Optional[str] = None) -> Optional[Match]:
        """At most one confirmed line about ``card_id``, or None.

        Kind priority is fixed (same_equation, boundary, tension, limit_of,
        same_mechanism, premise); ties keep file order, so the first
        hand-written edge wins deterministically. A follow-up may restrict
        the kind, match a topic and skip lines already spoken.
        ``exclude`` names cards the answer is already about (the caller
        passes the other top hits) — never explain what the user asked for.

        ``continuity_with`` is the card answered on the previous turn. If an
        edge joins the two, the line says "following on from" instead of
        introducing the other card cold; that is the one place the graph may
        remember a conversation.
        """
        skip: Set[str] = set(exclude)
        titles = titles or {}
        cap = (MAX_SENTENCE_CHARS_EN if language == "en"
               else MAX_SENTENCE_CHARS)
        if continuity_with and continuity_with != card_id:
            match = self._continuation(card_id, continuity_with, language,
                                       titles, cap)
            if match is not None and (relevant is None or relevant(match.text)) \
                    and not any(match.text in text for text in spoken):
                return match
        for kind in kinds:
            for edge in self.confirmed(card_id):
                if edge.kind != kind or (edge.language and edge.language != language):
                    continue
                other = edge.b if edge.a == card_id else edge.a
                # A boundary edge exists to say "this is not that". It must
                # speak even when the other card is the runner-up — that is
                # exactly the confusion it was written for.
                if titles and other not in titles:
                    continue
                if other in skip and kind != "boundary":
                    continue
                if kind not in SYMMETRIC and edge.a != card_id:
                    continue  # premise/boundary/limit_of point one way
                line = _render(edge, card_id, other, kind, language, titles)
                if (line and len(line) <= cap
                        and (relevant is None or relevant(line))
                        and not any(line in text for text in spoken)):
                    return Match(line, edge)
        return None

    def facets(self, card_id: str, language: str = "zh",
               titles: Optional[Mapping[str, str]] = None
               ) -> List[Tuple[str, str]]:
        """Every confirmed face this card can be looked at from again.

        Reflection needs a *different* angle on the same card, so this lists
        them all rather than the one a fresh answer would show: premise
        first, then what it is not, where it stops, and the analogy last.
        Each entry is ``(edge key, line)``; file order decides ties, so the
        list is deterministic. Never invents an edge to fill a gap.
        """
        titles = titles or {}
        cap = (MAX_SENTENCE_CHARS_EN if language == "en"
               else MAX_SENTENCE_CHARS)
        out: List[Tuple[str, str]] = []
        for kind in FACET_ORDER:
            for edge in self.confirmed(card_id):
                if edge.kind != kind or (edge.language and edge.language != language):
                    continue
                other = edge.b if edge.a == card_id else edge.a
                if other == card_id:
                    continue
                if kind not in SYMMETRIC and edge.a != card_id:
                    continue  # premise/boundary/limit_of point one way
                line = _render(edge, card_id, other, kind, language, titles)
                if not line or len(line) > cap:
                    continue
                out.append((edge.key, line))
        return out

    def _continuation(self, card_id: str, previous: str, language: str,
                      titles: Mapping[str, str], cap: int) -> Optional[Match]:
        for kind in SPOKEN:
            for edge in self.confirmed(card_id):
                if edge.kind != kind or (edge.language and edge.language != language):
                    continue
                other = edge.b if edge.a == card_id else edge.a
                if other != previous:
                    continue
                if kind not in SYMMETRIC and edge.a != card_id:
                    continue
                line = _render(edge, card_id, other, kind, language, titles,
                               continuity=True)
                if line and len(line) <= cap:
                    return Match(line, edge)
        return None


def _text(edge: Edge, key: str, language: str) -> str:
    if language == "en":
        return getattr(edge, f"{key}_en") or ""
    return getattr(edge, key) or ""


def _title(titles: Mapping[str, str], card_id: str) -> str:
    return titles.get(card_id) or card_id


def _render(edge: Edge, card_id: str, other: str, kind: str,
            language: str, titles: Mapping[str, str],
            continuity: bool = False) -> Optional[str]:
    mechanism = _text(edge, "mechanism", language)
    difference = _text(edge, "difference", language)
    mapping = mapping_text(edge, language, invert=card_id != edge.a)
    condition = _text(edge, "condition", language)
    if language == "en":
        if not mechanism or (kind in ("same_mechanism", "same_equation",
                                      "boundary", "tension")
                             and not difference):
            return None
        if kind == "same_equation" and not mapping:
            return None
        if kind == "limit_of" and not condition:
            return None
        name = _title(titles, other)
        if _CJK.search(name):
            return None  # no English name, so no English line: never mix
        if continuity:
            if kind == "same_mechanism":
                return (f"Following on from \"{name}\": same mechanism, "
                        f"{mechanism}. Where it parts: {difference}")
            if kind == "same_equation":
                return (f"Following on from \"{name}\": the same equation, "
                        f"{mechanism}. Match the symbols: {mapping}. "
                        f"Where it parts: {difference}")
            if kind == "boundary":
                return (f"Following on from \"{name}\", don't mix them up: "
                        f"{difference}")
            if kind == "tension":
                return (f"Following on from \"{name}\": it looks "
                        f"contradictory — {mechanism}. It isn't once you add: "
                        f"{difference}")
            if kind == "limit_of":
                return (f"Following on from \"{name}\": this is its limit "
                        f"when {condition}: {mechanism}")
            return (f"\"{name}\", from a moment ago, is the premise: "
                    f"{mechanism}")
        if kind == "same_mechanism":
            return (f"This is the same mechanism as \"{name}\": "
                    f"{mechanism}. Where it parts: {difference}")
        if kind == "same_equation":
            return (f"This is the same equation as \"{name}\": "
                    f"{mechanism}. Match the symbols: {mapping}. "
                    f"Where it parts: {difference}")
        if kind == "boundary":
            return f"Don't confuse it with \"{name}\": {difference}"
        if kind == "tension":
            return (f"This looks like it contradicts \"{name}\": "
                    f"{mechanism}. It doesn't once you add: {difference}")
        if kind == "limit_of":
            return (f"This is the limit of \"{name}\" when {condition}: "
                    f"{mechanism}")
        return f"The premise behind this sits in \"{name}\": {mechanism}"
    name = _title(titles, other)
    if continuity:
        if kind == "same_mechanism":
            return (f"接着刚才的《{name}》：还是同一个机制：{mechanism}。"
                    f"差别是：{difference}")
        if kind == "same_equation":
            return (f"接着刚才的《{name}》——同一个方程：{mechanism}；"
                    f"符号对上是：{mapping}。差别是：{difference}")
        if kind == "boundary":
            return f"接着刚才那张《{name}》，别混了：{difference}"
        if kind == "tension":
            return (f"接着刚才的《{name}》——看着矛盾：{mechanism}；"
                    f"不矛盾，缺的条件是：{difference}")
        if kind == "limit_of":
            return (f"接着刚才的《{name}》——这是它在{condition}时的极限："
                    f"{mechanism}")
        return f"刚才那张《{name}》就是这条的前提：{mechanism}"
    if kind == "same_mechanism":
        return f"这和《{name}》是同一个机制：{mechanism}。差别是：{difference}"
    if kind == "same_equation":
        return (f"这和《{name}》是同一个方程：{mechanism}；"
                f"把符号对上：{mapping}。差别是：{difference}")
    if kind == "boundary":
        return f"别和《{name}》混了：{difference}"
    if kind == "tension":
        return (f"这和《{name}》看着矛盾：{mechanism}。"
                f"不矛盾，缺的条件是：{difference}")
    if kind == "limit_of":
        return f"这是《{name}》在{condition}时的极限：{mechanism}"
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
        if not isinstance(row, dict):
            dropped += 1
            continue
        # The mapping is accepted in either spelling: a list of
        # [left, right] pairs, or "x↔q、m↔L".
        mapping = row.get("mapping", [])
        if isinstance(mapping, str):
            raw_pairs = [pair.strip() for pair in _MAPPING_SPLIT.split(mapping)
                         if pair.strip()]
            if any(pair.count("↔") != 1 for pair in raw_pairs):
                dropped += 1
                continue
        elif not isinstance(mapping, list) or any(
                not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(not isinstance(name, str) or not name.strip()
                       for name in pair) for pair in mapping):
            dropped += 1
            continue
        misconceptions = row.get("misconceptions", [])
        if not isinstance(misconceptions, list) or any(
                not isinstance(item, str) or not item.strip() for item in misconceptions):
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
            language=str(row.get("language", "")),
            mapping=tuple(_mapping_pairs(mapping)),
            mapping_en=str(row.get("mapping_en", "")),
            condition=str(row.get("condition", "")),
            condition_en=str(row.get("condition_en", "")),
            relation=str(row.get("relation", "")),
            source=str(row.get("source", "")),
            misconceptions=tuple(misconceptions),
        )
        if problems(edge):
            dropped += 1
            continue
        edges.append(edge)
    return AssocGraph(edges, dropped=dropped, path=file)
