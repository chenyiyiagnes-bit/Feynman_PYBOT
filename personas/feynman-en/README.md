# Feynman · English package

One package, one language. The Chinese package is `personas/feynman-zh/`; run
them separately — the runtime never mixes the two.

- Cards: `knowledge/*.jsonl`. The 293 rows in the thirteen teaching files
  are English. Two labels are independent:
  - `layer` — where the knowledge comes from.
    - `base` — his life, lectures, and books, or a second-hand account of
      them. A missing `layer` means `base`.
    - `learned` — modern knowledge the mental model has acquired. Every such
      card sets `"layer": "learned"` and a `support` note that says so.
  - `wording` — whether the card text is his speech.
    - `rewrite` — every teaching card. The text was checked against his
      English (or against a named second-hand or modern source) and then
      written in our words. It is not a quotation. Do not cite the card body
      as something he said.
    - `verbatim` — a sentence copied from his English, only in
      `verified_facts.jsonl`, ids `quote-*`. `text` and `quote` are the same
      string. The related teaching card stays a rewrite.
    - `fetched_page` — the nine `web-*` rows. Each is a stored page excerpt.
      The page was checked against itself. That does not make the excerpt his
      speech. One of them (`web-a7b3020dd7`) opens with a sentence from his
      1946 letter and then continues in the article's own words.
  The loader does not branch on `layer` or `wording`; it keeps the row.
  Runtime cards a user teaches later live in `data/kb_learned/` and are a
  different mechanism.
- `knowledge/verified_facts.jsonl` keeps the nine `web-*` rows already stored
  in `feynman-zh` (same ids, titles, text, quotes, and URLs) and adds only
  the `quote-*` verbatim sentences. Do not treat a `rewrite` card as a
  member of this file.
- Corpus files stay on the author's machine and are **not** committed.
- Derived state (`data/assoc/`, `data/lexicon/`, `data/kb_learned/`,
  `snapshots/`) is per-package, machine-local, and git-ignored; rebuild with
  `tools/build_assoc.py` / `tools/build_lexicon.py`.

```sh
python3 tools/validate_persona.py personas/feynman-en
python3 scripts/chat.py --persona feynman-en
python3 -m unittest discover -s tests
```
