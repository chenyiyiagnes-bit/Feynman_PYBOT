"""Code every card into token ids with byte offsets.

``build_lexicon.py`` says which words a package knows; this tool says where
each word sits in a card. A token is only useful for slot filling if it can be
pointed at: ``text.encode()[start:end].decode()`` must give the token back,
which the tests assert for every token in every card.

Input:  personas/<id>/knowledge/*.jsonl and data/lexicon/terms.json
Output: personas/<id>/data/lexicon/cards.coded.jsonl (derived, never committed)

Token ids are positions in ``terms.json``. They are deliberately *not*
authoritative: every coded token also carries its string, so a rebuilt lexicon
that reorders terms shows up as a mismatch instead of silently decoding to the
wrong word.

Usage:
    python3 tools/build_lexicon.py personas/feynman-zh
    python3 tools/code_cards.py personas/feynman-zh
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "right", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from omega_mind import lang  # noqa: E402

import build_lexicon  # noqa: E402

TOOL_VERSION = 1
OOV = -1


def load_terms(path: Path) -> Tuple[List[Dict], Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    terms = data["terms"]
    index = {row["term"]: position for position, row in enumerate(terms)}
    if len(index) != len(terms):
        raise SystemExit(f"{path}: duplicate terms in the token table")
    return terms, index


def byte_offsets(text: str) -> List[int]:
    """Byte position of every character boundary, plus the end."""
    out = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        out.append(total)
    return out


def code_field(text: str, core, index: Dict[str, int]) -> List[Dict]:
    offsets = byte_offsets(text)
    tokens = []
    for word, start, end in core.segment_spans(text):
        tid = index.get(word, OOV)
        if tid == OOV and core.language == "en":
            # English segments to stems; the table holds stems too.
            tid = index.get(core.stem(word.lower()), OOV)
        tokens.append({"id": tid, "t": word,
                       "start": offsets[start], "end": offsets[end]})
    return tokens


def code_card(card: Dict, core, index: Dict[str, int]) -> Dict:
    fields = []
    for field, value in card.items():
        if field not in build_lexicon.CONTENT_FIELDS:
            continue
        for text in _strings(value):
            if not text.strip():
                continue
            fields.append({"field": field, "tokens": code_field(text, core, index)})
    return {"id": str(card["id"]), "fields": fields}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def build(persona_dir: Path, lexicon_dir: Path | None = None,
          built: str | None = None) -> Dict:
    persona_dir = persona_dir.resolve()
    meta_path = persona_dir / "persona.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    language = meta.get("language") or "zh"
    core = lang.load(language, meta.get("language_profile") or "default")

    lexicon_dir = lexicon_dir or persona_dir / "data" / "lexicon"
    terms_path = lexicon_dir / "terms.json"
    if not terms_path.exists():
        raise SystemExit(f"{terms_path} is missing; run tools/build_lexicon.py first")
    terms, index = load_terms(terms_path)

    cards = build_lexicon.load_cards(persona_dir / "knowledge")
    coded = [code_card(card, core, index) for card in cards]

    total = sum(len(field["tokens"]) for row in coded for field in row["fields"])
    known = sum(1 for row in coded for field in row["fields"]
                for token in field["tokens"] if token["id"] != OOV)

    out = lexicon_dir / "cards.coded.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                           for row in coded), encoding="utf-8")
    manifest = {
        "tool": "tools/code_cards.py",
        "tool_version": TOOL_VERSION,
        "built": built or datetime.date.today().isoformat(),
        "persona": persona_dir.name,
        "language": language,
        "cards": len(coded),
        "tokens": total,
        "in_vocabulary": known,
        "coverage": round(known / total, 6) if total else 0.0,
        "terms_digest": hashlib.sha256(terms_path.read_bytes()).hexdigest()[:16],
        "note": "derived from knowledge/*.jsonl and terms.json; never committed",
    }
    (lexicon_dir / "cards.coded.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8")
    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("persona", help="package directory, e.g. personas/feynman-zh")
    parser.add_argument("--lexicon", help="directory holding terms.json")
    parser.add_argument("--date", help="manifest build date (default today)")
    args = parser.parse_args(argv)
    manifest = build(Path(args.persona),
                     lexicon_dir=Path(args.lexicon) if args.lexicon else None,
                     built=args.date)
    print(f"{manifest['persona']}: {manifest['cards']} cards, "
          f"{manifest['tokens']} tokens, {manifest['coverage']:.1%} in vocabulary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
