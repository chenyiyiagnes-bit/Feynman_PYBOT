# Feynman · English package

One package, one language. The Chinese package is `personas/feynman-zh/`; run
them separately — the runtime never mixes the two.

- Cards: `knowledge/*.jsonl`, 293 cards, quoted from Feynman's own English
  (lectures, interviews, talks). English text is never back-translated from
  the Chinese cards; each language quotes its own source.
- Corpus files stay on the author's machine and are **not** committed.
- Derived state (`data/assoc/`, `data/lexicon/`, `data/kb_learned/`,
  `snapshots/`) is per-package, machine-local, and git-ignored; rebuild with
  `tools/build_assoc.py` / `tools/build_lexicon.py`.

```sh
python3 tools/validate_persona.py personas/feynman-en
python3 scripts/chat.py --persona feynman-en
python3 -m unittest discover -s tests
```
