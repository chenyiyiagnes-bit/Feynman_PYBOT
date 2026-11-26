# Lao-Zhuang · English package

The English half of the Lao-Zhuang persona. One package, one language — the
Chinese package is `personas/laozhuang-zh/`, run it separately.

- Texts: the *Daodejing* and *Zhuangzi* are public domain. Translations must be
  public domain too (e.g. James Legge, 1891) or our own; no modern in-copyright
  translations, and no back-translating the Chinese cards. English is written
  from the source text.
- Source text and readings stay on the author's machine. **Nothing under
  `corpus/` or `readings/` is committed** — the repository ships cards only.
- Cards live in `knowledge/cards.jsonl` and pair with the Chinese set by id:
  `lz-en-NNN` ↔ `lz-zh-NNN`. Add both sides or neither; a test enforces the
  pairing and the language purity of every card.

## Voice

`voice.prefixes_en` / `voice.endings_en` are placeholders for now. Keep them
working (they are language-pure and the runtime uses them); rewrite them from
the source texts once the Chinese cards are final. Do not copy the Chinese
prefixes over.

## Working on it

```sh
python3 tools/validate_persona.py personas/laozhuang-en
python3 tools/doctor.py --persona laozhuang-en
python3 scripts/chat.py --persona laozhuang-en
python3 -m unittest tests.test_laozhuang_package
```

Tasks and red lines: `personas/laozhuang-zh/TASK.md` applies to both packages.
