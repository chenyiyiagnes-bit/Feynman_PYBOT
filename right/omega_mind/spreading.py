"""Opt-in query spreading over the document co-occurrence graph.

The corpus already answers by lexical overlap. Two cards about the same
subject often share neighbours without sharing the user's exact words
("阻尼" and "摩擦" may each sit near "振动" without either title mentioning
the other). This module builds the neighbour graph with normalised pointwise
mutual information and lets a query borrow a little energy from those
neighbours.

It is deliberately off by default. Two rules keep it honest:

* the bonus is small (``SPREAD_ENERGY``) — it can break a near tie, never
  invent an answer;
* a spread candidate must still open the card's *own* door: a query term has
  to hit the card's title / tags / aliases. A card matched only through its
  body is not allowed to be lifted over the threshold by a neighbour.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Set, Tuple

SPREAD_ENERGY = 0.35
MAX_NEIGHBORS = 5
MIN_COOCCUR = 2
# Same floor the scorer uses for "this token names the card's subject".
ANCHOR_WEIGHT = 2.5


def _tokens_of(doc: Any) -> Set[str]:
    return set(getattr(doc, "weights", {}) or {})


def npmi_neighbors(docs: Iterable[Any], top: int = MAX_NEIGHBORS,
                   min_cooccur: int = MIN_COOCCUR
                   ) -> Dict[str, List[Tuple[str, float]]]:
    """Top NPMI neighbours per token, built once per load."""
    documents = [_tokens_of(doc) for doc in docs]
    n = len(documents)
    if n < 2:
        return {}
    df: Dict[str, int] = {}
    pairs: Dict[Tuple[str, str], int] = {}
    for tokens in documents:
        for token in tokens:
            df[token] = df.get(token, 0) + 1
        ordered = sorted(tokens)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                pairs[(first, second)] = pairs.get((first, second), 0) + 1
    neighbours: Dict[str, List[Tuple[str, float]]] = {}
    for (first, second), count in pairs.items():
        if count < min_cooccur:
            continue
        p_pair = count / n
        p_first = df[first] / n
        p_second = df[second] / n
        denominator = -math.log(p_pair)
        if denominator <= 0:
            continue
        npmi = math.log(p_pair / (p_first * p_second)) / denominator
        if npmi <= 0:
            continue
        neighbours.setdefault(first, []).append((second, npmi))
        neighbours.setdefault(second, []).append((first, npmi))
    return {
        token: sorted(values, key=lambda pair: pair[1], reverse=True)[:top]
        for token, values in neighbours.items()
    }


def precision_open(query_tokens: Iterable[str], doc: Any,
                   floor: float = ANCHOR_WEIGHT) -> bool:
    """A spread card must name the subject, not merely mention it."""
    weights = getattr(doc, "weights", {}) or {}
    return any(weights.get(token, 0.0) >= floor for token in query_tokens)


def spread_bonus(query_tokens: Iterable[str], doc: Any,
                 neighbours: Dict[str, List[Tuple[str, float]]],
                 energy: float = SPREAD_ENERGY) -> float:
    """Energy borrowed from the strongest neighbour the card contains."""
    weights = getattr(doc, "weights", {}) or {}
    best = 0.0
    for token in query_tokens:
        for neighbour, npmi in neighbours.get(token, ()):
            if neighbour in weights:
                best = max(best, npmi)
    return energy * best if best > 0 else 0.0
