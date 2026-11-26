# English card overlay

This directory is a **translation overlay**, not a second corpus.

- Every row has the same `id` as a Chinese row in
  `~/Projects/feynman-open/knowledge/*.jsonl`, plus `translation_of` and
  `source_language: "zh"`.
- The Chinese row remains the source of truth. This overlay replaces only the
  reply/retrieval fields (`title`, `text`, `tags`, `aliases`, unit fields, ...);
  it never modifies the upstream corpus.
- Rows that were already English (`verified_facts`) are not copied here and are
  used as-is. They are the only safe source for a Feynman quotation.
- These translations are machine-produced and are **not** Feynman's original
  words. Do not present them as quotations.

Automated checks cover id alignment, provenance fields, field types, and the
absence of CJK characters in every value. There has not been a line-by-line
human semantic review. If a translation is corrected, edit it here and rerun
the test suite.
