"""Compare segmentation schemes on each package's own golden set.

A change to how text is cut into words can quietly move retrieval: the index
and the query are both built from tokens, so a different boundary is a
different vector. This tool scores each scheme on two things that matter —

* the package's golden questions (``personas/<id>/evals/golden.json``): does
  the right card still win, do the traps stay unanswered;
* the hand-segmented reference set (``tests/data/segmentation_reference.json``):
  boundary precision/recall against words a human wrote down.

Each package is scored with its own cards and its own net; a package without
``evals/golden.json`` is skipped, not judged against another package's net.

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
from omega_mind.golden_queries import for_package, nets_by_package  # noqa: E402

REFERENCE = ROOT / "tests" / "data" / "segmentation_reference.json"
PACKAGES_ROOT = ROOT / "personas"
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


def _package_language(package: Path) -> str:
    try:
        data = json.loads((package / "persona.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "zh"
    return str(data.get("language") or "zh")


def packages_with_nets(root: Path = PACKAGES_ROOT) -> List[Path]:
    """Packages that ship an eval net; one language, one index, one score."""
    return [root / name for name in sorted(nets_by_package(root))]


def evaluate_retrieval(scheme: Callable[[str], List[str]],
                       package: Path) -> Dict[str, object]:
    golden, negatives = for_package(package.name)
    language = _package_language(package)
    with patched(scheme):
        kb = feynman_kb.FeynmanKB(
            knowledge_dir=str(package / "knowledge"),
            language=language, source_prefix=package.name)
        if not kb.available():
            return {"unavailable": f"no cards in {package / 'knowledge'}"}
        prefix = kb.source_root + ":"
        hits, misses = 0, []
        for query, _, acceptable in golden:
            _, source = kb.answer(query)
            card = source[len(prefix):] if source.startswith(prefix) else ""
            if card in acceptable:
                hits += 1
            else:
                misses.append({"query": query, "got": card,
                               "expected": sorted(acceptable)})
        rejected, leaks = 0, []
        for query, _ in negatives:
            text, source = kb.answer(query)
            if text is None and source.endswith(":no_match"):
                rejected += 1
            else:
                leaks.append({"query": query, "source": source})
    return {"hits": hits, "total": len(golden), "misses": misses,
            "rejected": rejected, "negatives": len(negatives), "leaks": leaks}


def _merge_totals(rows: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    totals = {"hits": 0, "total": 0, "rejected": 0, "negatives": 0}
    misses, leaks = [], []
    for package, row in rows.items():
        for key in totals:
            totals[key] += int(row.get(key) or 0)
        misses.extend(dict(miss, package=package) for miss in row.get("misses") or [])
        leaks.extend(dict(leak, package=package) for leak in row.get("leaks") or [])
    return dict(totals, misses=misses, leaks=leaks)


def run(schemes: Dict[str, Callable[[str], List[str]]]) -> Dict[str, object]:
    packages = packages_with_nets()
    report: Dict[str, object] = {"baseline": BASELINE,
                                 "packages": [p.name for p in packages],
                                 "schemes": {}}
    for name, scheme in schemes.items():
        rows: Dict[str, Dict[str, object]] = {}
        try:
            for package in packages:
                rows[package.name] = evaluate_retrieval(scheme, package)
        except ImportError as error:
            report["schemes"][name] = {"unavailable": str(error)}
            continue
        report["schemes"][name] = {"packages": rows,
                                   "retrieval": _merge_totals(rows),
                                   "reference": evaluate_reference(scheme)}
    return report


def regressions(report: Dict[str, object]) -> List[str]:
    """Schemes that are worse than the baseline and must not be merged."""
    schemes = report["schemes"]
    base = schemes.get(BASELINE) or {}
    if "retrieval" not in base:
        raise SystemExit("baseline scheme unavailable; cannot judge candidates")
    base_retrieval = base["retrieval"]
    problems: List[str] = []
    # The shipped scheme must still pass its own packages' nets: a corpus,
    # index or scoring change that breaks the baseline is a regression too,
    # not just a candidate that scores lower.
    total = base_retrieval.get("total")
    if total is not None and base_retrieval["hits"] < total:
        problems.append(f"{BASELINE}: golden hits {base_retrieval['hits']} < "
                        f"{total} (shipped scheme fails its own net)")
    negatives = base_retrieval.get("negatives")
    if negatives is not None and base_retrieval["rejected"] < negatives:
        problems.append(f"{BASELINE}: traps rejected {base_retrieval['rejected']} < "
                        f"{negatives} (a trap got answered)")
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
    lines = ["package       scheme    golden     traps      F1     exact   note"]
    base_scheme = report["schemes"].get(report["baseline"]) or {}
    for name, row in report["schemes"].items():
        if "unavailable" in row:
            lines.append(f"{'-':<13} {name:<9} -          -          -      -       "
                         f"{row['unavailable']}")
            continue
        reference = row["reference"]
        for package, got in row["packages"].items():
            note = "baseline" if name == report["baseline"] else ""
            if "unavailable" in got:
                lines.append(f"{package:<13} {name:<9} -          -          -      "
                             f"-       {got['unavailable']}")
                continue
            lines.append(
                f"{package:<13} {name:<9} {got['hits']:>3}/{got['total']:<3} "
                f" {got['rejected']:>3}/{got['negatives']:<3} "
                f" {reference['mean_f1']:>6.3f}  "
                f"{reference['exact']:>3}/{reference['total']:<3}  {note}")
        retrieval = row["retrieval"]
        if name == report["baseline"]:
            note = "baseline total"
        elif "retrieval" in base_scheme:
            delta = retrieval["hits"] - base_scheme["retrieval"]["hits"]
            note = "same as baseline" if delta == 0 else f"{delta:+d} vs baseline"
        else:
            note = ""
        lines.append(
            f"{'ALL':<13} {name:<9} {retrieval['hits']:>3}/{retrieval['total']:<3} "
            f" {retrieval['rejected']:>3}/{retrieval['negatives']:<3} "
            f" {reference['mean_f1']:>6.3f}  "
            f"{reference['exact']:>3}/{reference['total']:<3}  {note}")
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
