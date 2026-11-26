"""Corpus profiler: what is actually in the card files.

Answers questions the retrieval layer cannot: how many rows per file, which
fields are missing, how much of it is Chinese, duplicate ids, verification
status, tag frequency. Read-only over JSONL.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_CJK = re.compile(r"[一-鿿]")
_ASCII_WORD = re.compile(r"[A-Za-z]{2,}")

# Fields every card should have, by role rather than by file name.
REQUIRED = ("id", "title")
ROLE_REQUIRED = {
    "teachable_unit": ("entry_question", "invariant", "phenomenon"),
    "physics_guardrail": ("text",),
    "lecture_card": ("text",),
    "biography": ("text",),
}


def _lang(text: str) -> str:
    cjk = len(_CJK.findall(text))
    words = len(_ASCII_WORD.findall(text))
    if cjk and words > cjk / 2:
        return "mixed"
    return "zh" if cjk else ("en" if words else "other")


def profile_file(path: Path) -> Dict[str, Any]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"__invalid__": line[:40]})
    ids = [r.get("id") for r in rows if r.get("id")]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]

    kinds = Counter(str(r.get("type") or path.stem) for r in rows)
    status = Counter(str(r.get("verification_status") or "unstated") for r in rows)
    body_fields = [k for k in ("text", "move", "pattern", "force_invariant", "invariant")
                   if any(k in r for r in rows)]
    missing = Counter()
    for row in rows:
        for field in REQUIRED:
            if not row.get(field):
                missing[field] += 1
        kind = str(row.get("type") or path.stem)
        for field in ROLE_REQUIRED.get(kind, ()):
            if not row.get(field):
                missing[field] += 1

    texts = [
        " ".join(str(row.get(k) or "") for k in body_fields + ["title", "tags"])
        for row in rows
    ]
    langs = Counter(_lang(t) for t in texts)
    lengths = [len(t) for t in texts] or [0]

    return {
        "file": path.name,
        "rows": len(rows),
        "invalid_json": sum(1 for r in rows if "__invalid__" in r),
        "kinds": dict(kinds),
        "body_fields": body_fields,
        "missing_required": dict(missing),
        "duplicate_ids": duplicates,
        "verification": dict(status),
        "language": dict(langs),
        "chars": {"min": min(lengths), "max": max(lengths),
                  "avg": round(sum(lengths) / len(lengths), 1)},
    }


def profile_corpus(directory: str | Path) -> Dict[str, Any]:
    path = Path(directory)
    files = [profile_file(f) for f in sorted(path.glob("*.jsonl"))]
    tags = Counter()
    total_rows = 0
    for file in sorted(path.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_rows += 1
            raw = row.get("tags")
            if isinstance(raw, str):
                raw = [t.strip() for t in raw.strip("[]").replace("'", "").split(",")]
            for tag in raw or []:
                if tag:
                    tags[str(tag)] += 1
    return {
        "directory": str(path),
        "files": files,
        "totals": {
            "rows": total_rows,
            "files": len(files),
            "duplicate_ids": sum(len(f["duplicate_ids"]) for f in files),
            "missing_required": sum(sum(f["missing_required"].values()) for f in files),
            "invalid_json": sum(f["invalid_json"] for f in files),
        },
        "top_tags": tags.most_common(15),
    }


def format_profile(profile: Dict[str, Any]) -> str:
    lines = [f"corpus: {profile['directory']}"]
    totals = profile["totals"]
    lines.append(
        f"  {totals['rows']} rows · {totals['files']} files · "
        f"{totals['duplicate_ids']} duplicate ids · "
        f"{totals['missing_required']} missing fields · "
        f"{totals['invalid_json']} invalid lines"
    )
    for f in profile["files"]:
        bits = [f"{f['rows']} rows", "/".join(f"{k}:{v}" for k, v in f["kinds"].items())]
        if f["missing_required"]:
            bits.append("missing " + ",".join(f["missing_required"]))
        if f["duplicate_ids"]:
            bits.append("dup " + ",".join(f["duplicate_ids"]))
        lines.append(f"  - {f['file']}: " + " · ".join(bits))
    if profile["top_tags"]:
        lines.append("  top tags: " + ", ".join(f"{t}({n})" for t, n in profile["top_tags"]))
    return "\n".join(lines)
