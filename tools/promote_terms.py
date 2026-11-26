"""Promote a compound the cards keep saying together into the word list.

A candidate comes from the cards themselves: two known terms that stand next
to each other in many cards, tightly associated (high normalised PMI). Before
it is accepted it has to earn its place — adding it to the segmenter must not
lose any exact match on the hand-segmented reference set, and must raise the
mean boundary F1 by at least ``MIN_F1_GAIN``. Every decision is written down
with its provenance so it can be reviewed and replayed; a candidate that fails
is recorded as discarded and not retried unless ``--retry`` says so.

The lexicon rebuild (``tools/build_lexicon.py``) reads ``promotions.jsonl`` and
keeps whatever was promoted, so a full rebuild does not silently undo a
decision — it is a decision, not a patch to be overwritten.

Usage:
    python3 tools/build_lexicon.py personas/feynman-zh
    python3 tools/promote_terms.py personas/feynman-zh            # report only
    python3 tools/promote_terms.py personas/feynman-zh --apply    # write the log
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "right", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from omega_mind import lang  # noqa: E402

import build_lexicon  # noqa: E402
import eval_segmentation  # noqa: E402

TOOL_VERSION = 1
MIN_NPMI = 0.5        # tight enough that the pairing is not a coincidence
MIN_CARDS = 3         # has to recur, not appear once
MIN_LEN = 3           # two-character pairs are already solid vocabulary
MAX_LEN = 6           # longer than a term is a phrase the cards repeat
MIN_F1_GAIN = 0.0005  # the provisional threshold from HANDOFF §5
PROMOTIONS = "promotions.jsonl"


def adjacency(core, cards: Sequence[Dict]) -> Tuple[Counter, Dict, Dict]:
    """Adjacent term pairs across cards, plus how many cards each term is in."""
    pairs: Counter = Counter()
    pair_cards: Dict[Tuple[str, str], set] = defaultdict(set)
    term_cards: Dict[str, set] = defaultdict(set)
    for card in cards:
        text = build_lexicon.card_text(card)
        tokens = [tok for tok in core.segment(text) if build_lexicon.keep(core, tok)]
        for token in set(tokens):
            term_cards[token].add(str(card["id"]))
        for left, right in zip(tokens, tokens[1:]):
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            pairs[pair] += 1
            pair_cards[pair].add(str(card["id"]))
    return pairs, pair_cards, term_cards


def npmi(pair: Tuple[str, str], pairs: Counter, pair_cards, term_cards,
         total: int) -> float:
    joint = len(pair_cards[pair]) / total
    left = len(term_cards[pair[0]]) / total
    right = len(term_cards[pair[1]]) / total
    if joint <= 0 or left <= 0 or right <= 0 or joint >= 1:
        return 0.0
    pmi = math.log(joint / (left * right))
    return pmi / -math.log(joint)


def candidates(core, cards: Sequence[Dict], known: Iterable[str]) -> List[Dict]:
    known = set(known)
    pairs, pair_cards, term_cards = adjacency(core, cards)
    total = len(cards) or 1
    out = []
    for pair, count in pairs.items():
        if count < MIN_CARDS or len(pair_cards[pair]) < MIN_CARDS:
            continue
        term = pair[0] + pair[1]
        if term in known or term in core.data.get("lexicon", ()):
            continue
        if len(term) < MIN_LEN or len(term) > MAX_LEN:
            continue
        if not all(len(part) >= 2 and build_lexicon._is_cjk(part[0]) for part in pair):
            continue
        if not all(build_lexicon._is_cjk(char) for char in term):
            continue
        score = npmi(pair, pairs, pair_cards, term_cards, total)
        if score < MIN_NPMI:
            continue
        out.append({"term": term, "parts": list(pair),
                    "npmi": round(score, 6), "cards": count,
                    "sources": sorted(pair_cards[pair])[:8]})
    out.sort(key=lambda row: (-row["npmi"], -row["cards"], row["term"]))
    return out


def with_term(core, term: str):
    """A copy of the core that knows one more word."""
    data = copy.deepcopy(core.data)
    data["lexicon"] = sorted(set(data.get("lexicon", ())) | {term})
    return lang.LanguageCore(data)


def evaluate(core, term: str) -> Dict:
    before = eval_segmentation.evaluate_reference(core.segment)
    after = eval_segmentation.evaluate_reference(with_term(core, term).segment)
    return {
        "f1_before": before["mean_f1"], "f1_after": after["mean_f1"],
        "exact_before": before["exact"], "exact_after": after["exact"],
        "total": before["total"],
        "gain": round(after["mean_f1"] - before["mean_f1"], 6),
    }


def decide(core, candidate: Dict) -> Dict:
    measured = evaluate(core, candidate["term"])
    keeps_exact = measured["exact_after"] >= measured["exact_before"]
    gains = measured["gain"] >= MIN_F1_GAIN
    status = "promoted" if keeps_exact and gains else "discarded"
    if not keeps_exact:
        reason = f"lost an exact match ({measured['exact_before']} -> {measured['exact_after']})"
    elif not gains:
        reason = f"gain {measured['gain']} below {MIN_F1_GAIN}"
    else:
        reason = f"gain +{measured['gain']} on the reference set"
    record = dict(candidate)
    record.update(measured)
    record.update({"status": status, "reason": reason,
                   "threshold": MIN_F1_GAIN})
    return record


def load_log(path: Path) -> Dict[str, Dict]:
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["term"]] = row
    return rows


def write_log(path: Path, rows: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[term], ensure_ascii=False, sort_keys=True) + "\n"
                            for term in sorted(rows)), encoding="utf-8")


def run(package: Path, apply: bool = False, retry: bool = False,
        built: str | None = None) -> Dict:
    package = package.resolve()
    meta = json.loads((package / "persona.json").read_text(encoding="utf-8")) \
        if (package / "persona.json").exists() else {}
    core = lang.load(meta.get("language") or "zh",
                     meta.get("language_profile") or "default")
    cards = build_lexicon.load_cards(package / "knowledge")
    terms_path = package / "data" / "lexicon" / "terms.json"
    known = {row["term"] for row in
             json.loads(terms_path.read_text(encoding="utf-8"))["terms"]} \
        if terms_path.exists() else set()

    log_path = package / "data" / "lexicon" / PROMOTIONS
    log = load_log(log_path)
    examined = []
    for candidate in candidates(core, cards, known):
        previous = log.get(candidate["term"])
        if previous and previous.get("status") == "discarded" and not retry:
            continue
        record = decide(core, candidate)
        record["evaluated"] = built or datetime.date.today().isoformat()
        record["tool_version"] = TOOL_VERSION
        log[candidate["term"]] = record
        examined.append(record)

    if apply and examined:
        write_log(log_path, log)
    promoted = [row["term"] for row in examined if row["status"] == "promoted"]
    return {"examined": examined, "promoted": promoted,
            "log": str(log_path), "written": bool(apply and examined)}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("persona")
    parser.add_argument("--apply", action="store_true",
                        help="write decisions to data/lexicon/promotions.jsonl")
    parser.add_argument("--retry", action="store_true",
                        help="re-evaluate candidates that were discarded before")
    parser.add_argument("--date", help="evaluation date (default today)")
    args = parser.parse_args(argv)
    report = run(Path(args.persona), apply=args.apply, retry=args.retry,
                 built=args.date)
    for row in report["examined"]:
        print(f"{row['status']:<9} {row['term']}  npmi {row['npmi']}  "
              f"cards {row['cards']}  {row['reason']}")
    if not report["examined"]:
        print("no new candidates")
    print(f"{len(report['promoted'])} promoted"
          f"{'' if report['written'] else ' (dry run; --apply to record)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
