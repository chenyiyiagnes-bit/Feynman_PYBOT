# Feynman · English package

One package, one language. The Chinese package is `personas/feynman-zh/`; run
them separately — the runtime never mixes the two.

- Cards: `knowledge/*.jsonl`, 302 cards. Lecture and interview cards quote
  English sources; they are not back-translated from the Chinese cards.
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
