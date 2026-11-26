# Feynman · English package

One package, one language. The Chinese package is `personas/feynman-zh/`; run
them separately — the runtime never mixes the two.

- Cards: `knowledge/*.jsonl`. The 293 rows in the thirteen teaching files
  are meant to come from Feynman's English (lectures, books, talks), not
  from a translation of `feynman-zh`. Where a card was rewritten that way,
  `translation_of` and `source_language` are gone and a `support` note names
  the English source. Cards with no English source checked were left as they
  were, and those still carry the old translation marks. The nine `web-*`
  rows are a separate file.
- `knowledge/verified_facts.jsonl` holds the nine `web-*` quotes already
  stored in `feynman-zh` (same ids, titles, text, and URLs). Those rows were
  English to begin with. The Chinese audit note on them is rewritten in
  English so this package stays one language.
- Corpus files stay on the author's machine and are **not** committed.
- Derived state (`data/assoc/`, `data/lexicon/`, `data/kb_learned/`,
  `snapshots/`) is per-package, machine-local, and git-ignored; rebuild with
  `tools/build_assoc.py` / `tools/build_lexicon.py`.

```sh
python3 tools/validate_persona.py personas/feynman-en
python3 scripts/chat.py --persona feynman-en
python3 -m unittest discover -s tests
```
