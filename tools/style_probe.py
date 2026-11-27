"""A/B probes for the 活字印刷 note's own methods, on this repo's data.

The note proposes five mechanisms that can be measured without any model
call:

  A. retrieval by keyword overlap (+ a randomly injected golden sentence)
  B. a banned-word list pushed into the decoder (logit_bias)
  C. rhetorical-anchor skeleton mining with slot abstraction
  D. second-order Markov text generation
  E. one global style average per book

Each is run against the persona packages and the golden nets that the repo
already ships, and compared with what the repo does today. Read-only; the
only output is a measurement table.

Usage:
  python3 tools/style_probe.py
  python3 tools/style_probe.py --package feynman-zh
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "right"))

from style_profile import (  # noqa: E402
    _is_language, _load_cards, _package_language, _sentences,
)

# ---------------------------------------------------------------- corpora

def load_package(name: str) -> dict:
    pkg = ROOT / "personas" / name
    lang = _package_language(pkg)
    # ``_load_cards`` keeps only the measurement fields; the retrieval probe
    # also needs each card's id to score against the golden net, so rows are
    # read once more here with the full record.
    rows = []
    for path in sorted((pkg / "knowledge").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text") or row.get("answer") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if not _is_language(text, lang):
                continue
            row["text"] = text
            row["_stem"] = path.stem
            rows.append(row)
    golden = json.loads((pkg / "evals" / "golden.json").read_text(encoding="utf-8"))
    return {"name": name, "lang": lang, "rows": rows, "golden": golden}


# ------------------------------------------------------- A. retrieval

ZH_BIGRAM = re.compile(r"[一-鿿]")
EN_WORD = re.compile(r"[a-z0-9]+")


def _doc_tokens(text: str, lang: str) -> set[str]:
    if lang == "zh":
        clean = "".join(ch for ch in text if ZH_BIGRAM.match(ch))
        return {clean[i:i + 2] for i in range(len(clean) - 1)}
    return set(EN_WORD.findall(text.lower()))


def doc_overlap_rank(query: str, rows: list[dict], lang: str) -> list[tuple[float, dict]]:
    """The note's ``_dynamic_pick``: keyword overlap, best first.

    jieba is not installed system-wide, so words are replaced by character
    bigrams (zh) / lowercase words (en). Overlap is normalised by the query,
    which is how the note's ``len(intersection) / len(query_words)`` behaves.
    """
    q = _doc_tokens(query, lang)
    if not q:
        return []
    scored = []
    for row in rows:
        d = _doc_tokens(row["text"], lang)
        if not d:
            continue
        scored.append((len(q & d) / len(q), row))
    scored.sort(key=lambda kv: -kv[0])
    return scored


def kb_rank(package: str, lang: str, query: str) -> list[tuple[float, str]]:
    from omega_mind.feynman_kb import FeynmanKB
    kb = FeynmanKB(knowledge_dir=str(ROOT / "personas" / package / "knowledge"),
                   language=lang)
    hits = kb.search(query, limit=3)
    return [(score, doc.id) for score, doc in hits]


def probe_retrieval(pkg: dict) -> dict:
    lang, rows, golden = pkg["lang"], pkg["rows"], pkg["golden"]
    by_id = {}
    for row in rows:
        by_id.setdefault(row.get("id"), row)
    ok_doc = ok_kb = total = 0
    misses = []
    zero_overlap = 0
    rows_with_id = [r for r in rows if r.get("id")]
    for item in golden["golden"]:
        q, accept = item["query"], set(item["acceptable"])
        total += 1
        ranked = doc_overlap_rank(q, rows_with_id, lang)
        if not ranked or ranked[0][0] == 0:
            zero_overlap += 1
        elif ranked[0][1].get("id") in accept:
            ok_doc += 1
        try:
            hits = kb_rank(pkg["name"], lang, q)
        except Exception as exc:  # KB not importable / package broken
            hits = []
            if total == 1:
                misses.append(f"KB error: {exc}")
        if hits and hits[0][1] in accept:
            ok_kb += 1
        elif not any(h and h[1] in accept for h in hits[:3]):
            misses.append(q)
    return {
        "package": pkg["name"], "language": lang, "n_queries": total,
        "doc_overlap_top1": round(ok_doc / total, 3) if total else 0,
        "kb_top1": round(ok_kb / total, 3) if total else 0,
        "zero_overlap_queries": zero_overlap,
        "kb_misses": misses[:6],
    }


# --------------------------------------------------- B. banned words

# Verbatim from the note's ``token_bias_builder.py`` (multi-token words get
# -4.5; the note calls these "AI 客服腔").
BANNED = ["首先", "其次", "最后", "综上所述", "总而言之", "总之", "赋能",
          "抓手", "闭环", "维度", "痛点", "底层逻辑", "希望", "帮助",
          "抱歉", "作为", "助手", "此外", "然而"]


def probe_banned(pkg: dict) -> dict:
    texts = [r["text"] for r in pkg["rows"]]
    joined = "\n".join(texts)
    hits = {w: joined.count(w) for w in BANNED if joined.count(w)}
    affected = sum(1 for t in texts if any(w in t for w in BANNED))
    return {
        "package": pkg["name"],
        "banned_hits_in_corpus": sum(hits.values()),
        "banned_words_hit": hits,
        "texts_touching_a_banned_word": affected,
        "fraction_of_texts": round(affected / max(len(texts), 1), 3),
    }


# ---------------------------------------------------- C. skeleton miner

RHETORICAL = [
    r"你看[，,].*?其实.*?", r"我讨厌.*?因为.*?", r"如果你.*?那说明.*?",
    r"有意思的是[，,].*?结果.*?", r"人们总以为.*?但实际上.*?",
    r"别去.*?不代表.*?", r"本质上.*?不过是.*?罢了",
]
PROTECTED = {"你看", "其实", "讨厌", "因为", "说明", "有意思", "结果",
             "本质", "不过", "罢了", "小孩", "大自然", "上帝", "科学", "玩意"}


def _slot_abstract(sentence: str) -> str:
    """A jieba-free stand-in for the note's POS slotting.

    Without a segmenter we can only slot 2+char runs that are *not* in the
    protected list — and we cannot tell a noun from a modifier, which is
    exactly the information the note's ``{n}/{v}/{a}`` boxes need. The
    probe measures what the anchors alone buy.
    """
    out, slots, i = [], 0, 0
    for m in re.finditer(r"[一-鿿]{2,}", sentence):
        chunk = m.group()
        if chunk in PROTECTED or any(p in chunk for p in PROTECTED):
            out.append(sentence[i:m.end()])
        else:
            out.append(sentence[i:m.start()])
            out.append("{x}")
            slots += 1
        i = m.end()
    out.append(sentence[i:])
    return "".join(out) if slots else ""


def probe_skeleton(pkg: dict) -> dict:
    texts = [r["text"] for r in pkg["rows"]]
    sentences = []
    for t in texts:
        sentences.extend(s for s in _sentences(t, pkg["lang"])
                         if 10 <= len(s) <= 45)
    anchored = [s for s in sentences
                if any(re.search(p, s) for p in RHETORICAL)]
    templates = {_slot_abstract(s) for s in anchored}
    templates.discard("")
    multi_slot = sum(1 for t in templates if t.count("{x}") >= 2)
    return {
        "package": pkg["name"],
        "n_sentences_10_45": len(sentences),
        "anchored_sentences": len(anchored),
        "anchor_recall": round(len(anchored) / max(len(sentences), 1), 3),
        "templates": len(templates),
        "templates_with_2plus_slots": multi_slot,
        "slot_density": round(sum(t.count("{x}") for t in templates)
                              / max(len(templates), 1), 2),
    }


# ------------------------------------------------------- D. Markov

def probe_markov(pkg: dict, samples: int = 200, seed: int = 7) -> dict:
    if pkg["lang"] != "zh":
        return {"package": pkg["name"], "skipped": "zh only in this probe"}
    chain: dict[tuple, list[str]] = {}
    starts = []
    corpus_grams = set()
    for row in pkg["rows"]:
        for sent in _sentences(row["text"], "zh"):
            chars = [c for c in sent if "一" <= c <= "鿿"]
            for i in range(len(chars) - 1):
                chain.setdefault((chars[i],), []).append(chars[i + 1])
                if i + 1 < len(chars):
                    chain.setdefault((chars[i], chars[i + 1]), []).append(
                        chars[i + 2] if i + 2 < len(chars) else "。")
            for i in range(len(chars) - 3):
                corpus_grams.add("".join(chars[i:i + 4]))
            if len(chars) >= 2:
                starts.append(chars[0])
    rng = random.Random(seed)
    seen, novel, lengths = 0, 0, []
    for _ in range(samples):
        cur = [rng.choice(starts)]
        for _ in range(60):
            key = (cur[-1],)
            nxt = chain.get(key) or chain.get((cur[-2], cur[-1])) if len(cur) > 1 else chain.get(key)
            if not nxt:
                break
            ch = rng.choice(nxt)
            if ch == "。":
                break
            cur.append(ch)
        if len(cur) < 4:
            continue
        lengths.append(len(cur))
        grams = ["".join(cur[i:i + 4]) for i in range(len(cur) - 3)]
        seen += len(grams)
        novel += sum(1 for g in grams if g not in corpus_grams)
    return {
        "package": pkg["name"],
        "samples": len(lengths),
        "mean_generated_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "novel_4gram_rate": round(novel / max(seen, 1), 3),
        "note": "novel_4gram_rate high = glue never appeared in the corpus",
    }


# ------------------------------------------------------ E. variance

def probe_variance(pkg: dict) -> dict:
    by: dict[str, list[int]] = {}
    for row in pkg["rows"]:
        by.setdefault(row.get("_stem", "?"), []).extend(
            len(s) for s in _sentences(row["text"], pkg["lang"]))
    means = {k: sum(v) / len(v) for k, v in by.items() if v}
    if len(means) < 2:
        return {"package": pkg["name"], "registers": len(means)}
    mu = sum(means.values()) / len(means)
    var = sum((v - mu) ** 2 for v in means.values()) / len(means)
    return {
        "package": pkg["name"],
        "registers": len(means),
        "min_mean": round(min(means.values()), 1),
        "max_mean": round(max(means.values()), 1),
        "ratio": round(max(means.values()) / max(min(means.values()), 1e-9), 2),
        "cv": round((var ** 0.5) / mu, 3),
        "worst": sorted(means.items(), key=lambda kv: kv[1])[:1],
        "longest": sorted(means.items(), key=lambda kv: -kv[1])[:1],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", action="append", default=[])
    args = ap.parse_args(argv)
    names = args.package or ["feynman-zh", "feynman-en",
                             "laozhuang-zh", "laozhuang-en"]
    for name in names:
        pkg = load_package(name)
        print(f"\n===== {name} ({pkg['lang']}, {len(pkg['rows'])} rows)")
        for probe in (probe_retrieval, probe_banned, probe_skeleton,
                      probe_markov, probe_variance):
            try:
                result = probe(pkg)
            except Exception as exc:
                result = {"probe": probe.__name__, "error": repr(exc)}
            print(f"  [{probe.__name__[6:]}] "
                  + json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
