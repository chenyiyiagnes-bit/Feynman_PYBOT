# Feynman · English package

One package, one language. The Chinese package is `personas/feynman-zh/`; run
them separately — the runtime never mixes the two.

- Cards: `knowledge/*.jsonl`. The 293 rows in the thirteen teaching files
  are English. A card has a `layer`:
  - `base` — Feynman's own life, words, lectures, and books. This is the
    foundation. A missing `layer` means `base`, so older rewritten cards are
    not retagged one by one.
  - `learned` — modern knowledge the mental model has acquired, not a passage
    from his corpus. Every such card sets `"layer": "learned"` and a `support`
    note that says so. It is not written as if he had said it.
  Where a base card was rewritten from an English source, `translation_of`
  and `source_language` are gone and `support` names the source. A card that
  still cannot be sourced or classified would keep the old translation marks.
  The nine `web-*` rows are a separate file. The loader does not branch on
  `layer`; it keeps the row. Runtime cards a user teaches later live in
  `data/kb_learned/` and are a different mechanism.
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
