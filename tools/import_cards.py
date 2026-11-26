"""Turn your own notes into knowledge cards.

    python3 tools/import_cards.py notes.md -o personas/mybot/knowledge/notes.jsonl
    python3 tools/import_cards.py notes.md --tags 物理,入门 --print

Markdown ``#``/``##`` headings start a card; a plain text file splits on blank
lines. The first short line of each block becomes the card title — the title
and tags are what retrieval scores, so write them with the words someone would
actually search for. No model is involved: your text goes in, cards come out,
nothing is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]

TITLE_MAX = 40


def split_markdown(text: str) -> List[tuple[str, str]]:
    """[(title, body)] from heading-delimited sections."""
    sections: List[tuple[str, str]] = []
    title, body = "", []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^#{1,3}\s+", stripped):
            if title or body:
                sections.append((title, "\n".join(body).strip()))
            title, body = re.sub(r"^#{1,3}\s+", "", stripped).strip(), []
        else:
            body.append(line)
    if title or body:
        sections.append((title, "\n".join(body).strip()))
    return [(t, b) for t, b in sections if b]


def split_paragraphs(text: str) -> List[tuple[str, str]]:
    """[(title, body)] from blank-line separated blocks."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        first = lines[0]
        if len(first) <= TITLE_MAX and len(lines) > 1:
            out.append((first, " ".join(lines[1:])))
        else:
            title = first[:TITLE_MAX].rstrip("，。,.、 ")
            out.append((title, " ".join(lines)))
    return out


def latin_tokens(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9+-]{2,}", text)]


def cards_from(text: str, prefix: str, extra_tags: Iterable[str] = ()) -> List[dict]:
    sections = split_markdown(text)
    blocks = sections if any(t for t, _ in sections) else split_paragraphs(text)
    cards = []
    for title, body in blocks:
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
        tags = list(dict.fromkeys([*extra_tags, *latin_tokens(title)]))
        card = {
            "id": f"{prefix}-{digest}",
            "type": "fact",
            "title": title,
            "text": body,
            "tags": tags,
        }
        cards.append(card)
    return cards


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a .md/.txt file with your notes")
    ap.add_argument("-o", "--out", help="write cards to this .jsonl file")
    ap.add_argument("--tags", default="", help="comma-separated tags added to every card")
    ap.add_argument("--prefix", default=None, help="card id prefix (default: file stem)")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print cards instead of writing them")
    args = ap.parse_args(argv)

    src = Path(args.source).expanduser()
    if not src.is_file():
        print(f"not a file: {src}", file=sys.stderr)
        return 2
    text = src.read_text(encoding="utf-8")
    prefix = args.prefix or re.sub(r"[^a-z0-9]+", "-", src.stem.lower()).strip("-") or "note"
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    cards = cards_from(text, prefix, tags)
    if not cards:
        print("no cards found — split the file into headings or blank-line blocks",
              file=sys.stderr)
        return 1

    payload = "\n".join(json.dumps(c, ensure_ascii=False) for c in cards) + "\n"
    if args.show or not args.out:
        print(payload, end="")
        return 0
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload, encoding="utf-8")
    print(f"{len(cards)} cards → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
