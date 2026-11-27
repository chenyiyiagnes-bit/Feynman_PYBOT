"""Measured surface style per persona package — the numbers behind "speaks like X".

The 活字印刷 note proposes a ``surface_fingerprint`` (sentence length,
punctuation habits, high-frequency vocabulary) but computes one global
average per book. A persona package is not a book: lectures, anecdotes and
Q&A speak differently, and the reply voice we care about is narrower than
the whole card corpus. This script measures the same quantities, but per
source file and per package, so a reply can be checked against a band
instead of a vibe.

Read-only: nothing is written unless ``--out`` is given, and the profile
holds only numbers and short tokens — never card text — so the output is
safe to keep next to the package.

Usage:
  python3 tools/style_profile.py --package feynman-zh
  python3 tools/style_profile.py --all
  python3 tools/style_profile.py --all --out personas
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bridge.style_gate import (  # noqa: E402
    CLICHES, MIN_SENTENCE_CHARS, is_language as _is_language,
    pct as _pct, sentences as _sentences,
)

# Sources whose text is the persona's own speaking voice. Files that only
# carry structure (relations, people, edges) have no ``text`` and are
# skipped by the loader anyway; this list decides the rows that count
# toward the package-level band.
VOICE_FILES = {
    "zh": {
        "facts", "verified_facts", "feynman_biography", "feynman_lectures",
        "feynman_physics", "feynman_math_comp", "feynman_life_art",
        "persona_core", "persona_memories", "physics_guardrails",
        "thinking_moves", "cards",
    },
    "en": {
        "facts", "verified_facts", "feynman_biography", "feynman_lectures",
        "feynman_physics", "feynman_math_comp", "feynman_life_art",
        "persona_core", "persona_memories", "physics_guardrails",
        "thinking_moves", "cards",
    },
}


EN_STOPWORDS = set("""
the a an and or but if then than that this these those there here
of in on at to from by for with without into over under about after
before between through during as is are was were be been being am
do does did have has had having it its he she they them his her their
we you your our i me my not no so such can could may might must
will would should just also very more most much many few some any all
what when where who whom which why how one two three
""".split())

TERMINALS = {"zh": ("。", "！", "？", "…"), "en": (".", "!", "?", "…")}

PUNCT = {
    "zh": {
        "comma": "，",
        "period": "。",
        "question": "？",
        "exclaim": "！",
        "dash": "——",
        "ellipsis": "……",
        "colon": "：",
        "semicolon": "；",
    },
    "en": {
        "comma": ",",
        "period": ".",
        "question": "?",
        "exclaim": "!",
        "dash": "—",
        "ellipsis": "...",
        "colon": ":",
        "semicolon": ";",
    },
}

# Characters that carry no topic on their own. High-frequency n-grams made
# only of these are discarded, which keeps the vocabulary list meaningful
# without a segmenter.
ZH_FUNCTION_CHARS = set(
    "的了着过是在有和与或就都也还又很更最太不没无别被把给对向从到为以于而且但"
    "如果因为所以这那哪什么怎样吗呢吧啊呀嘛哦嗯他她它我你们个之其所以然地得"
    "一二三四五六七八九十百千万些许才再呢吗吧"
)

def _load_cards(package_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((package_dir / "knowledge").glob("*.jsonl")):
        stem = path.stem
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("answer") or ""
            if isinstance(text, str) and text.strip():
                rows.append({"source": stem, "text": text.strip(),
                             "title": row.get("title", ""),
                             "lang": row.get("lang")})
    return rows


def _package_language(package_dir: Path) -> str:
    """The package's language, from its golden net (same rule the KB uses)."""
    golden = package_dir / "evals" / "golden.json"
    if golden.exists():
        try:
            lang = json.loads(golden.read_text(encoding="utf-8")).get("language")
            if lang in ("zh", "en"):
                return lang
        except json.JSONDecodeError:
            pass
    # Fall back to the densest script in the text.
    for row in _load_cards(package_dir):
        cjk = sum(1 for ch in row["text"] if "一" <= ch <= "鿿")
        if len(row["text"]) and cjk / len(row["text"]) > 0.3:
            return "zh"
    return "en"


def _vocabulary(texts: list[str], lang: str, top: int = 12) -> list[str]:
    if lang == "zh":
        counts: dict[str, int] = {}
        for text in texts:
            clean = re.sub(r"[^一-鿿]", "", text)
            for n in (2, 3):
                for i in range(len(clean) - n + 1):
                    gram = clean[i:i + n]
                    if all(ch in ZH_FUNCTION_CHARS for ch in gram):
                        continue
                    if len(set(gram)) == 1:
                        continue
                    counts[gram] = counts.get(gram, 0) + 1
        # Drop an n-gram whose longer extension is at least as frequent:
        # "费曼先生" explains "费曼" better than "费曼" alone.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0])))
        kept: list[str] = []
        for gram, count in ranked:
            if count < 3:
                break
            if any(gram in longer and longer != gram and counts[longer] >= count
                   for longer in kept):
                continue
            kept.append(gram)
            if len(kept) >= top:
                break
        return kept
    counts = {}
    for text in texts:
        for word in re.findall(r"[A-Za-z']+", text.lower()):
            if len(word) < 4 or word in EN_STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
    return [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if counts[w] >= 3][:top]


def profile_texts(texts: list[str], lang: str) -> dict:
    sentences: list[str] = []
    for text in texts:
        sentences.extend(_sentences(text, lang))
    lengths = [len(s) for s in sentences]
    joined = "\n".join(texts)
    n_chars = len(joined) or 1
    punct = PUNCT[lang]
    habit = {name: round(joined.count(sym) * 1000.0 / n_chars, 1)
             for name, sym in punct.items()}
    terminals = sum(joined.count(mark) for mark in TERMINALS[lang]) or 1
    q = joined.count("？" if lang == "zh" else "?")
    ex = joined.count("！" if lang == "zh" else "!")
    clichés = CLICHES[lang]
    lowered = joined.lower() if lang == "en" else joined
    cliché_hits = {c: lowered.count(c) for c in clichés if lowered.count(c)}
    return {
        "unit": "chars",
        "n_texts": len(texts),
        "n_sentences": len(sentences),
        "sentence_len": {
            "mean": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
            "p50": _pct(lengths, 0.50),
            "p90": _pct(lengths, 0.90),
            "max": max(lengths) if lengths else 0,
        },
        "question_ratio": round(q / terminals, 3),
        "exclaim_ratio": round(ex / terminals, 3),
        "punct_per_1000_chars": habit,
        "high_freq_terms": _vocabulary(texts, lang),
        "ai_cliches_per_text": round(sum(cliché_hits.values()) / max(len(texts), 1), 3),
        "ai_cliches_found": cliché_hits,
    }


def profile_package(package_dir: Path) -> dict:
    lang = _package_language(package_dir)
    rows = _load_cards(package_dir)
    # A zh package carries a few English citations (and vice versa); they
    # are real content but they describe another voice, so they stay out of
    # the band. They are still counted in ``n_dropped`` so the loss is
    # visible rather than silent.
    kept = [r for r in rows if _is_language(r["text"], lang)]
    by_source: dict[str, list[str]] = {}
    for row in kept:
        by_source.setdefault(row["source"], []).append(row["text"])
    voice = VOICE_FILES[lang]
    voice_texts = [row["text"] for row in kept if row["source"] in voice]
    fallback = [r["text"] for r in kept]
    return {
        "schema": 1,
        "language": lang,
        "package": package_dir.name,
        "n_rows": len(rows),
        "n_dropped_wrong_language": len(rows) - len(kept),
        "overall": profile_texts(fallback, lang),
        "voice": profile_texts(voice_texts or fallback, lang),
        "by_source": {name: profile_texts(texts, lang)
                      for name, texts in sorted(by_source.items())},
    }


def _print_summary(profile: dict) -> None:
    for key in ("overall", "voice"):
        p = profile[key]
        s = p["sentence_len"]
        print(f"  {key:8s} texts={p['n_texts']:3d} sent={p['n_sentences']:4d} "
              f"len mean={s['mean']:5.1f} p50={s['p50']:4.0f} p90={s['p90']:4.0f} "
              f"?={p['question_ratio']:.2f} !={p['exclaim_ratio']:.2f} "
              f"cliché={p['ai_cliches_per_text']:.2f}")
        print(f"           terms: {', '.join(p['high_freq_terms'][:8])}")
    print("  by source:")
    for name, p in profile["by_source"].items():
        s = p["sentence_len"]
        print(f"    {name:28s} mean={s['mean']:5.1f} p90={s['p90']:4.0f} "
              f"?={p['question_ratio']:.2f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", action="append", default=[],
                    help="package name under personas/ (repeatable)")
    ap.add_argument("--all", action="store_true", help="every package")
    ap.add_argument("--out", help="directory to write <package>.json into")
    ap.add_argument("--json", action="store_true", help="print the profile as JSON")
    args = ap.parse_args(argv)

    names = args.package
    if args.all:
        names = [p.name for p in sorted((ROOT / "personas").iterdir()) if p.is_dir()]
    if not names:
        ap.error("give --package NAME or --all")

    for name in names:
        package_dir = ROOT / "personas" / name
        if not (package_dir / "knowledge").is_dir():
            print(f"skip {name}: no knowledge/", file=sys.stderr)
            continue
        profile = profile_package(package_dir)
        if args.json:
            print(json.dumps(profile, ensure_ascii=False, indent=2))
        else:
            print(f"== {name} ({profile['language']})")
            _print_summary(profile)
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / name / "evals" / "style_profile.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
