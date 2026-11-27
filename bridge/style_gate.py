"""Measured style band and the check a reply is held to.

The band is built from what the package's own bot says — the composed
answer of every card (``Doc.answer_for``), not the raw card text, because a
reply appends invariants and evidence and is twice as long as the card.
Register bands (per source file) are kept for diagnosis; the gate itself
uses the package band, since register bands are too narrow for composed
answers.

Nothing here rewrites a reply. ``check`` measures and reports.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

# Sentences shorter than this are fragments ("好。" / "Yes.") and would
# pull the mean down without saying anything about rhythm.
MIN_SENTENCE_CHARS = 4

ZH_SENT_SPLIT = re.compile(r"[。！？!?…]+")
EN_SENT_SPLIT = re.compile(r"[.!?]+(?=\s|$)")

CLICHES = {
    "zh": [
        "首先", "其次", "综上所述", "总而言之", "总之", "换句话说",
        "值得注意的是", "希望对你有帮助", "作为一个", "总的来说",
        "不难看出", "由此可见",
    ],
    "en": [
        "as an ai", "it's important to note", "in conclusion", "firstly",
        "secondly", "overall,", "i hope this helps", "delve",
        "it is worth noting", "in summary",
    ],
}

# A register needs this many cards before its own band means anything.
MIN_CARDS_FOR_REGISTER = 8


def sentences(text: str, lang: str) -> list[str]:
    splitter = ZH_SENT_SPLIT if lang == "zh" else EN_SENT_SPLIT
    parts = [p.strip() for p in splitter.split(text or "")]
    return [p for p in parts if len(p) >= MIN_SENTENCE_CHARS]


def pct(values: Iterable[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[idx])


def is_language(text: str, lang: str) -> bool:
    """True when ``text`` is predominantly written in ``lang``."""
    if not text:
        return False
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cjk + letters
    if not total:
        return False
    return (cjk / total) >= 0.5 if lang == "zh" else (letters / total) > 0.5


def count_cliches(text: str, lang: str) -> dict[str, int]:
    lowered = text.lower() if lang == "en" else text
    return {c: lowered.count(c) for c in CLICHES[lang] if lowered.count(c)}


def band(samples: list[tuple[str, str]], lang: str,
         cliche_max: int = 0) -> dict:
    """Build a band from ``(register, text)`` samples."""
    means_all: list[float] = []
    max_all: list[int] = []
    per: dict[str, tuple[list[float], list[int]]] = {}
    for register, text in samples:
        lens = [len(s) for s in sentences(text, lang)]
        if not lens:
            continue
        mean = sum(lens) / len(lens)
        means_all.append(mean)
        max_all.append(max(lens))
        m, x = per.setdefault(register, ([], []))
        m.append(mean)
        x.append(max(lens))
    overall = {
        "language": lang,
        "mean_min": round(pct(means_all, 0.05), 1),
        "mean_max": round(pct(means_all, 0.95), 1),
        "sentence_cap": round(pct(max_all, 0.98)),
        "cliche_max": cliche_max,
        "n_samples": len(means_all),
    }
    by_source = {}
    for register, (m, x) in per.items():
        if len(m) < MIN_CARDS_FOR_REGISTER:
            continue
        by_source[register] = {
            "language": lang,
            "mean_min": round(pct(m, 0.05), 1),
            "mean_max": round(pct(m, 0.95), 1),
            "sentence_cap": round(pct(x, 0.98)),
            "cliche_max": cliche_max,
            "n_samples": len(m),
        }
    return {"overall": overall, "by_source": by_source}


def split_appendix(text: str) -> tuple[str, str]:
    """Separate the reply body from its one trailing aside.

    The right brain appends a single line — an association remark, a
    magnitude anchor or a curiosity question — after a plain ``"\\n"``
    (``engine._with_association``), while the answer's own parts are joined
    with ``"\\n\\n"``. The aside is scaffolding, not voice, so the band is
    measured on the body alone; when the text is a single block, nothing is
    split.

    Card text that itself holds one newline (one meta card in
    ``feynman-zh``) would read as an aside here. That is the cost of a
    structural rule with no parser; the band is built from
    ``answer_for()``, where no split happens, so the two sides stay
    consistent.
    """
    text = text or ""
    head, sep, tail = text.rpartition("\n")
    if not sep or not head.strip() or not tail.strip():
        return text, ""
    if head.endswith("\n"):  # the separator was part of "\n\n": all body
        return text, ""
    return head, tail


def check(text: str, band_dict: dict) -> dict:
    """Measure one reply against a band. Returns metrics and a verdict."""
    text = (text or "").strip()
    lang = band_dict["language"]
    if not text:
        return {"ok": False, "metrics": {}, "checks": {"non_empty": False}}
    parts = sentences(text, lang)
    lengths = [len(s) for s in parts]
    hits = count_cliches(text, lang)
    checks = {
        "non_empty": True,
        "language": is_language(text, lang),
    }
    if lengths:
        mean = sum(lengths) / len(lengths)
        checks["mean_in_band"] = band_dict["mean_min"] <= mean <= band_dict["mean_max"]
        checks["no_runaway_sentence"] = max(lengths) <= band_dict["sentence_cap"]
    else:
        mean = 0.0
        checks["mean_in_band"] = False
        checks["no_runaway_sentence"] = False
    checks["no_ai_cliche"] = len(hits) <= band_dict["cliche_max"]
    return {
        "ok": all(checks.values()),
        "metrics": {
            "n_sentences": len(parts),
            "mean_sentence_chars": round(mean, 1),
            "max_sentence_chars": max(lengths) if lengths else 0,
            "cliche_hits": hits,
        },
        "checks": checks,
    }


def band_for_docs(lang: str, docs: list, stem_by_id: Optional[dict] = None) -> dict:
    """Band for a KB package from its composed card answers.

    ``docs`` are ``Doc`` objects (anything with ``id`` and ``answer_for()``).
    """
    stem_by_id = stem_by_id or {}
    samples = [(stem_by_id.get(doc.id, "?"), doc.answer_for()) for doc in docs]
    return band(samples, lang)
