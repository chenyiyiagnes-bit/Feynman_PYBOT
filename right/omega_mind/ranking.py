"""Package-local BM25 and bounded reciprocal-rank fusion."""

from __future__ import annotations

from collections import Counter
from math import log


class BM25:
    def __init__(self, docs, core):
        self.documents = {}
        self.df = Counter()
        for doc in docs:
            # Index title/aliases twice, and the answer once.
            aliases = (doc.row or {}).get("aliases", [])
            aliases = [aliases] if isinstance(aliases, str) else aliases
            head = " ".join([doc.title, *(alias for alias in aliases if isinstance(alias, str))])
            terms = Counter(core.tokens(" ".join((head, head, doc.answer))))
            self.df.update(terms.keys())
            self.documents[doc.id] = doc, terms, sum(terms.values())
        total = sum(length for _, _, length in self.documents.values())
        self.average = total / max(1, len(self.documents)) or 1

    def rank(self, query, limit, allowed=None):
        weights = {word: log(1 + (len(self.documents) - self.df[word] + .5)
                             / (self.df[word] + .5))
                   for word in sorted(set(query)) if word in self.df}
        scores = []
        for cid, (doc, terms, length) in self.documents.items():
            if allowed is not None and cid not in allowed:
                continue
            length_norm = 1.2 * (.25 + .75 * length / self.average)
            score = sum(weight * (terms[word] * 2.2) / (terms[word] + length_norm)
                        for word, weight in weights.items() if word in terms)
            if score:
                scores.append((score, doc))
        return sorted(scores, key=lambda x: (-x[0], x[1].id))[:limit]


def fuse(legacy, bm25, limit):
    scores = Counter()
    for hits in (legacy, bm25):
        for rank, (_, doc) in enumerate(hits, 1):
            scores[doc.id] += 1 / (60 + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:limit]]
