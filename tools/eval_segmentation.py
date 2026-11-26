"""Compare segmentation schemes on the golden retrieval set.

A change to how text is cut into words can quietly move retrieval: the index
and the query are both built from tokens, so a different boundary is a
different vector. This tool scores each scheme on two things that matter —

* the golden questions (``omega_mind.golden_queries``): does the right card
  still win, do the traps stay unanswered;
* the hand-segmented reference set (``tests/data/segmentation_reference.json``):
  boundary precision/recall against words a human wrote down.

Usage::

    python3 tools/eval_segmentation.py                 # table, no failure
    python3 tools/eval_segmentation.py --strict        # CI: fail on a regression
    python3 tools/eval_segmentation.py --json out.json # machine-readable

Schemes: ``bigram`` is the shipped baseline (character bigrams), ``words`` is
``LanguageCore.segment()`` from the language core, ``jieba`` is the left
brain's third-party segmenter when it is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "right"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from omega_mind import feynman_kb  # noqa: E402
from omega_mind import lang  # noqa: E402
from omega_mind.golden_queries import GOLDEN, NEGATIVES  # noqa: E402

REFERENCE = ROOT / "tests" / "data" / "segmentation_reference.json"
BASELINE = "bigram"


# ---------------------------------------------------------------------------
# segmenters: text -> tokens, the shape ``feynman_kb`` indexes with
# ---------------------------------------------------------------------------

_ORIGINAL_TOKENS = feynman_kb._tokens


def _bigram(text: str) -> List[str]:
    return _ORIGINAL_TOKENS(text)


def _words(text: str) -> List[str]:
    return lang.load("zh").segment(text)


def _jieba(text: str) -> List[str]:
    import jieba  # noqa: PLC0415 - optional dependency, probed at run time
    return [tok for tok in jieba.lcut(text) if tok.strip()]


SCHEMES: Dict[str, Callable[[str], List[str]]] = {
    "bigram": _bigram,
    "words": _words,
    "jieba": _jieba,
}


class _ProxyCore:
    """A language core whose only difference is how it cuts Chinese text."""

    def __init__(self, core, scheme: Callable[[str], List[str]]) -> None:
        self._core = core
        self._scheme = scheme

    def __getattr__(self, name):
        return getattr(self._core, name)

    def tokens(self, text: str) -> List[str]:
        return self._scheme(text)


@contextlib.contextmanager
def patched(scheme: Callable[[str], List[str]]):
    """Route the KB's Chinese tokenizer through ``scheme`` for one build.

    The knowledge base reads through its own language core, so the scheme is
    installed where the core is loaded rather than by swapping a module-level
    function nothing calls any more.
    """
    original = lang.load

    def load_with_scheme(language, profile=None):
        core = original(language, profile)
        return _ProxyCore(core, scheme) if language == "zh" else core

    lang.load = load_with_scheme
    try:
        yield
    finally:
        lang.load = original


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def boundaries(words: Iterable[str]) -> List[int]:
    """Character offsets where one word ends and the next begins."""
    out: List[int] = []
    total = 0
    for word in words:
        total += len(word)
        out.append(total)
    return out[:-1]


def boundary_scores(gold: List[str], predicted: List[str]) -> Tuple[float, float, float]:
    """(precision, recall, f1) over boundary positions."""
    want = set(boundaries(gold))
    got = set(boundaries(predicted))
    if not want and not got:
        return 1.0, 1.0, 1.0
    hit = len(want & got)
    precision = hit / len(got) if got else 0.0
    recall = hit / len(want) if want else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_reference(scheme: Callable[[str], List[str]]) -> Dict[str, object]:
    data = json.loads(REFERENCE.read_text(encoding="utf-8"))
    rows = []
    for case in data["cases"]:
        gold = case["words"]
        predicted = scheme(case["text"])
        precision, recall, f1 = boundary_scores(gold, predicted)
        rows.append({"text": case["text"], "gold": gold, "predicted": predicted,
                     "precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4)})
    mean_f1 = sum(row["f1"] for row in rows) / len(rows) if rows else 0.0
    exact = sum(1 for row in rows if row["gold"] == row["predicted"])
    return {"cases": rows, "mean_f1": round(mean_f1, 4),
            "exact": exact, "total": len(rows)}


def evaluate_retrieval(scheme: Callable[[str], List[str]]) -> Dict[str, object]:
    with patched(scheme):
        kb = feynman_kb.FeynmanKB()
        if not kb.available():
            raise SystemExit("feynman knowledge is not on this machine; "
                             "set QBOT_FEYNMAN_CORPUS or run where it exists")
        hits, misses = 0, []
        for query, language, acceptable in GOLDEN:
            _, source = kb.answer(query, language=language)
            card = source.split(":", 1)[1] if ":" in source else ""
            if card in acceptable:
                hits += 1
            else:
                misses.append({"query": query, "language": language,
                               "got": card, "expected": sorted(acceptable)})
        rejected, leaks = 0, []
        for query, language in NEGATIVES:
            text, source = kb.answer(query, language=language)
            if text is None and source.endswith(":no_match"):
                rejected += 1
            else:
                leaks.append({"query": query, "language": language, "source": source})
    return {"hits": hits, "total": len(GOLDEN), "misses": misses,
            "rejected": rejected, "negatives": len(NEGATIVES), "leaks": leaks}


def run(schemes: Dict[str, Callable[[str], List[str]]]) -> Dict[str, object]:
    report: Dict[str, object] = {"baseline": BASELINE, "schemes": {}}
    for name, scheme in schemes.items():
        try:
            retrieval = evaluate_retrieval(scheme)
        except ImportError as error:
            report["schemes"][name] = {"unavailable": str(error)}
            continue
        reference = evaluate_reference(scheme)
        report["schemes"][name] = {"retrieval": retrieval, "reference": reference}
    return report


def regressions(report: Dict[str, object]) -> List[str]:
    """Schemes that are worse than the baseline and must not be merged."""
    schemes = report["schemes"]
    base = schemes.get(BASELINE) or {}
    if "retrieval" not in base:
        raise SystemExit("baseline scheme unavailable; cannot judge candidates")
    base_retrieval = base["retrieval"]
    problems: List[str] = []
    for name, row in schemes.items():
        if name == BASELINE or "retrieval" not in row:
            continue
        retrieval = row["retrieval"]
        if retrieval["hits"] < base_retrieval["hits"]:
            problems.append(f"{name}: golden hits {retrieval['hits']} < "
                            f"{base_retrieval['hits']}")
        if retrieval["rejected"] < base_retrieval["rejected"]:
            problems.append(f"{name}: traps rejected {retrieval['rejected']} < "
                            f"{base_retrieval['rejected']}")
    return problems


def render(report: Dict[str, object]) -> str:
    lines = ["scheme    golden     traps      F1     exact   note"]
    for name, row in report["schemes"].items():
        if "unavailable" in row:
            lines.append(f"{name:<9} -          -          -      -       {row['unavailable']}")
            continue
        retrieval, reference = row["retrieval"], row["reference"]
        note = ""
        if name == report["baseline"]:
            note = "baseline"
        else:
            base = report["schemes"][report["baseline"]]["retrieval"]
            delta = retrieval["hits"] - base["hits"]
            note = "same as baseline" if delta == 0 else f"{delta:+d} vs baseline"
        lines.append(
            f"{name:<9} {retrieval['hits']:>3}/{retrieval['total']:<3} "
            f" {retrieval['rejected']:>3}/{retrieval['negatives']:<3} "
            f" {reference['mean_f1']:>6.3f}  {reference['exact']:>3}/{reference['total']:<3}  {note}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scheme", action="append", choices=sorted(SCHEMES),
                        help="scheme to score (repeatable); default all")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 when a candidate is worse than the baseline")
    parser.add_argument("--json", metavar="PATH", help="also write the full report")
    args = parser.parse_args(argv)

    names = args.scheme or list(SCHEMES)
    if BASELINE not in names:
        names.insert(0, BASELINE)
    report = run({name: SCHEMES[name] for name in names})

    print(render(report))
    problems = regressions(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    if problems:
        print("\nregressions:")
        for problem in problems:
            print(f"  - {problem}")
        return 1 if args.strict else 0
    print("\nno scheme is worse than the baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
