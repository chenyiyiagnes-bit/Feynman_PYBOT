"""Derive a term / co-occurrence lexicon from a persona's cards.

The cards are the only source of truth; this tool only measures them. Every
string field is mined — structured cards keep their body in ``phenomenon``,
``force_invariant`` or ``pattern`` and have no ``text`` field — and the text is
cut with the package's own language rules (``LanguageCore.segment``), so a
package that reads classical Chinese is measured with classical boundaries.

Output (all derived, never committed):

    personas/<id>/data/lexicon/terms.json      term, df, tf, idf, category, cards
    personas/<id>/data/lexicon/edges.jsonl     term pairs with PMI and a relation
    personas/<id>/data/lexicon/manifest.json   source digest, counts, versions

Two builds from the same cards are byte-identical, so a change in the files is
a change in the cards, not in run order.

Usage:
    python3 tools/build_lexicon.py personas/feynman-zh
    python3 tools/build_lexicon.py personas/laozhuang-zh --date 2026-11-26
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind import lang  # noqa: E402

TOOL_VERSION = 1
MIN_DF = 2            # a term seen in one card is a typo or a proper name
MIN_PAIR = 2          # an edge needs to happen in more than one card
MAX_DF_RATIO = 0.6    # a term in most cards says nothing about which card
MIN_CARDS_FOR_BREADTH = 20
MAX_CARDS_PER_TERM = 8
MIN_ASCII_LEN = 3
RELATION = "co_occurs_in_card"

# Fields that say what a card teaches. A whitelist rather than a blocklist:
# provenance fields (``source``, ``source_note``, ``name_en``) are what editors
# wrote to each other, and indexing them would put "编译" and "source" at the
# top of every package. Structured cards keep their body in one of these, so
# missing ``text`` does not mean an empty card.
CONTENT_FIELDS = (
    "title", "text", "quote", "aliases",
    # teachable units / physics edges
    "phenomenon", "pattern", "invariant", "force_invariant", "operator",
    "transfer", "boundary", "check_yourself", "entry_question",
    "reality_meaning", "novel_use", "guardrail", "game_instance",
    # persona core / taste / relations
    "move", "contempt", "respect", "word", "expand",
)


# ---------------------------------------------------------------------------
# reading the cards
# ---------------------------------------------------------------------------

def load_cards(knowledge_dir: Path) -> List[Dict]:
    cards: List[Dict] = []
    seen: set = set()
    for path in sorted(knowledge_dir.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path.name}:{number}: card is not an object")
            if "id" not in row:
                raise SystemExit(f"{path.name}:{number}: card has no id")
            if row["id"] in seen:
                raise SystemExit(f"{path.name}:{number}: duplicate card id {row['id']}")
            seen.add(row["id"])
            cards.append(row)
    return cards


def card_strings(value, key: str = "") -> Iterator[str]:
    """Every content string under a card, whatever field it hides in.

    Only ``CONTENT_FIELDS`` are read: a card that has none of them teaches
    nothing this tool can measure, and provenance or translation fields are
    left alone on purpose.
    """
    if isinstance(value, str):
        if key in CONTENT_FIELDS:
            yield value
    elif isinstance(value, dict):
        for sub_key, sub in value.items():
            yield from card_strings(sub, sub_key)
    elif isinstance(value, list):
        for item in value:
            yield from card_strings(item, key)


def card_text(card: Dict) -> str:
    return "\n".join(text for text in card_strings(card) if text.strip())


def card_label(card: Dict) -> str:
    """What kind of card a term was found in — the term's category hint."""
    for field in ("type", "partition"):
        value = card.get(field)
        if isinstance(value, str) and value:
            return value
    return "card"


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------

def _is_cjk(char: str) -> bool:
    return "一" <= char <= "鿿"


def keep(core, token: str) -> bool:
    """Is this token a word the lexicon should carry?

    Function words are not vocabulary: a token whose every Chinese character
    is a stop character ("不是", "为什么") never names anything, and a single
    Chinese character is a fragment rather than a word.
    """
    if not token:
        return False
    if token.isdigit():
        return False
    stop_chars = getattr(core, "_unigram_stop", frozenset())
    cjk = [ch for ch in token if _is_cjk(ch)]
    if cjk:
        if all(ch in stop_chars for ch in cjk):
            return False
        if getattr(core, "language", "zh") == "zh" and len(cjk) == 1 and len(token) == 1:
            return False
    if token.isascii():
        # "g-2" names a quantity; "dq" is a fragment
        if len(token) < MIN_ASCII_LEN and not any(ch.isdigit() for ch in token):
            return False
        if token in getattr(core, "_stop_words", ()):
            return False
    return True


def term_counts(core, cards: Sequence[Dict]) -> Tuple[Counter, Dict[str, int], Dict[str, Counter], Dict[str, List[str]]]:
    """df, tf, per-term category counts and the cards a term comes from."""
    df: Counter = Counter()
    tf: Counter = Counter()
    categories: Dict[str, Counter] = defaultdict(Counter)
    sources: Dict[str, List[str]] = defaultdict(list)
    for card in cards:
        tokens = [tok for tok in core.segment(card_text(card)) if keep(core, tok)]
        uniq = sorted(set(tokens))
        tf.update(tokens)
        for token in uniq:
            df[token] += 1
            categories[token][card_label(card)] += 1
        for token in uniq:
            sources[token].append(str(card["id"]))
    return df, tf, categories, sources


def sorted_terms(core, cards: Sequence[Dict]) -> List[Dict]:
    df, tf, categories, sources = term_counts(core, cards)
    total = len(cards) or 1
    rows = []
    for term in sorted(df):
        if df[term] < MIN_DF:
            continue
        # A ratio means nothing in a five-card package; only judge breadth
        # once there are enough cards for "most of them" to be a fact.
        if total >= MIN_CARDS_FOR_BREADTH and df[term] / total > MAX_DF_RATIO:
            continue
        category = sorted(categories[term].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        rows.append({
            "term": term,
            "df": df[term],
            "tf": tf[term],
            # smoothed inverse document frequency; rounded so two builds agree
            "idf": round(math.log(1 + total / df[term]), 6),
            "category": category,
            "cards": sorted(sources[term])[:MAX_CARDS_PER_TERM],
            "card_count": len(sources[term]),
        })
    rows.sort(key=lambda row: (-row["df"], row["term"]))
    return rows


# ---------------------------------------------------------------------------
# edges
# ---------------------------------------------------------------------------

def sorted_edges(core, cards: Sequence[Dict], terms: Sequence[Dict]) -> List[Dict]:
    wanted = {row["term"] for row in terms}
    pair_cards: Dict[Tuple[str, str], int] = Counter()
    for card in cards:
        tokens = sorted({tok for tok in core.segment(card_text(card)) if tok in wanted})
        for i, left in enumerate(tokens):
            for right in tokens[i + 1:]:
                pair_cards[(left, right)] += 1
    total = len(cards) or 1
    df = {row["term"]: row["df"] for row in terms}
    rows = []
    for (left, right), count in sorted(pair_cards.items()):
        if count < MIN_PAIR:
            continue
        p_left = df[left] / total
        p_right = df[right] / total
        p_pair = count / total
        pmi = round(math.log(p_pair / (p_left * p_right)), 6)
        if pmi <= 0:
            continue
        npmi = round(pmi / -math.log(p_pair), 6)
        rows.append({
            "source": left,
            "target": right,
            "cards": count,
            "pmi": pmi,
            "npmi": npmi,
            "relation": RELATION,
        })
    rows.sort(key=lambda row: (-row["npmi"], -row["cards"], row["source"], row["target"]))
    return rows


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def digest(knowledge_dir: Path) -> str:
    """One hash over every card file, so the manifest names its input."""
    overall = hashlib.sha256()
    for path in sorted(knowledge_dir.glob("*.jsonl")):
        overall.update(path.name.encode("utf-8"))
        overall.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
    return overall.hexdigest()[:16]


def dump_lines(rows: Iterable[Dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows)


def build(persona_dir: Path, out_dir: Path | None = None,
          language: str | None = None, profile: str | None = None,
          built: str | None = None) -> Dict:
    persona_dir = persona_dir.resolve()
    meta_path = persona_dir / "persona.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    language = language or meta.get("language") or "zh"
    profile = profile or meta.get("language_profile") or "default"
    core = lang.load(language, profile)

    knowledge_dir = persona_dir / "knowledge"
    if not knowledge_dir.is_dir():
        raise SystemExit(f"no knowledge directory in {persona_dir}")
    cards = load_cards(knowledge_dir)
    if not cards:
        raise SystemExit(f"{knowledge_dir} has no cards")

    terms = sorted_terms(core, cards)
    edges = sorted_edges(core, cards, terms)

    out_dir = Path(out_dir) if out_dir else persona_dir / "data" / "lexicon"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "terms.json").write_text(
        json.dumps({"version": TOOL_VERSION, "language": language,
                    "profile": profile, "terms": terms},
                   ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    (out_dir / "edges.jsonl").write_text(dump_lines(edges), encoding="utf-8")
    manifest = {
        "tool": "tools/build_lexicon.py",
        "tool_version": TOOL_VERSION,
        "built": built or datetime.date.today().isoformat(),
        "persona": persona_dir.name,
        "language": language,
        "profile": profile,
        "cards": len(cards),
        "terms": len(terms),
        "edges": len(edges),
        "min_df": MIN_DF,
        "min_pair": MIN_PAIR,
        "source_digest": digest(knowledge_dir),
        "note": "derived from knowledge/*.jsonl; never committed",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("persona", help="package directory, e.g. personas/feynman-zh")
    parser.add_argument("--out", help="output directory (default <persona>/data/lexicon)")
    parser.add_argument("--language", help="override the package language")
    parser.add_argument("--profile", help="override the language profile")
    parser.add_argument("--date", help="manifest build date (default today)")
    args = parser.parse_args(argv)

    manifest = build(Path(args.persona), out_dir=Path(args.out) if args.out else None,
                     language=args.language, profile=args.profile, built=args.date)
    print(f"{manifest['persona']}: {manifest['cards']} cards -> "
          f"{manifest['terms']} terms, {manifest['edges']} edges "
          f"({manifest['language']}/{manifest['profile']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
