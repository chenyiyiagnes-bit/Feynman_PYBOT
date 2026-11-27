"""Per-turn review: what one exchange taught, judged at once, promoted never.

The dream-time audit (``reflect.py``) only sees turns that survived to a
dream. A standalone stretch of use may never dream — the state directory can
be cleared, the process can be killed, the user can leave — and turns where
the cards could not answer are exactly the ones worth keeping.

So every turn is judged when it ends, by rules only:

* **abstain** — the reply was the fallback line: nothing answered.
* **unverified_followup** — a follow-up could not be grounded in the current
  topic; the weak card elsewhere was refused on purpose.
* **weak** — a card answered, but barely: a soft hit, a score under the
  loader's floor, or a near-tie.

Each judgement is one append-only line in ``turn_gaps.jsonl`` next to the
learned store. That file is a trail, not a curriculum: nothing here is
promoted. Repetition is what earns a change, exactly as in ``reflect.py``:
the second sighting of the same question queues a ``gap`` candidate in the
review journal (``review.py``), and even then the candidate waits for a human
decision. A single odd question stays a line in the trail.

The same restraint the Hermes Agent issue taught: an unverified attempt must
never become a confident note. Here the note is the honest record of what the
cards could not answer, and the machinery that could turn it into a card is
the same one that was already gated.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

FILE = "turn_gaps.jsonl"
REASONS = ("abstain", "unverified_followup", "weak")
# The second sighting queues a candidate; one odd question never does.
REPEAT_TO_QUEUE = 2
# The trail is capped by rewrite, not by silence: the newest lines win.
MAX_LINES = 5000


def _key(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def judge(source: str, retrieval: Optional[Dict[str, Any]],
          min_score: Optional[float] = None,
          soft_margin: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """One turn's outcome, or ``None`` when the turn went well."""
    retrieval = retrieval or {}
    if source.startswith("fallback") or source.endswith(":no_match"):
        return {"reason": "abstain"}
    if retrieval.get("unverified_followup"):
        return {"reason": "unverified_followup"}
    if ":" not in source:
        return None
    if "_soft:" in source:
        return {"reason": "weak"}
    score = retrieval.get("score")
    margin = retrieval.get("margin")
    if score is not None and min_score is not None and score < min_score:
        return {"reason": "weak"}
    if margin is not None and soft_margin is not None and margin < soft_margin:
        return {"reason": "weak"}
    return None


def path_for(store) -> Path:
    return Path(store.dir) / FILE


def record(store, row: Dict[str, Any],
           now: Optional[float] = None) -> Dict[str, Any]:
    """Append one judgement; return it with the question's running count."""
    entry = dict(row)
    entry.setdefault("key", _key(str(entry.get("text", ""))))
    entry.setdefault("ts", now if now is not None else time.time())
    path = path_for(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    entry["count"] = count_of(store, str(entry.get("text", "")))
    _trim(path)
    return entry


def _trim(path: Path) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= MAX_LINES:
        return
    keep = lines[-MAX_LINES:]
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(line + "\n" for line in keep), encoding="utf-8")
    tmp.replace(path)


def events(store) -> List[Dict[str, Any]]:
    path = path_for(store)
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("reason") in REASONS:
            out.append(row)
    return out


def count_of(store, text: str) -> int:
    key = _key(text)
    return sum(1 for row in events(store) if row.get("key") == key)


def summary(store) -> List[Dict[str, Any]]:
    """Group the trail by question, newest first among equal counts."""
    groups: Dict[str, Dict[str, Any]] = {}
    for row in events(store):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        key = row.get("key") or _key(text)
        item = groups.setdefault(key, {
            "text": text,
            "language": row.get("language", ""),
            "count": 0,
            "reasons": [],
            "near": None,
            "first_turn": row.get("turn"),
            "last_ts": row.get("ts"),
        })
        item["count"] += 1
        reason = str(row.get("reason"))
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        if row.get("near") and not item["near"]:
            item["near"] = row["near"]
        item["first_turn"] = min(item["first_turn"] or row.get("turn") or 0,
                                 row.get("turn") or item["first_turn"] or 0)
        if (row.get("ts") or 0) > (item["last_ts"] or 0):
            item["last_ts"] = row.get("ts")
    return sorted(groups.values(),
                  key=lambda r: (r["count"], r["last_ts"] or 0), reverse=True)


def queue_repeats(store, min_count: int = REPEAT_TO_QUEUE) -> List[str]:
    """Queue reviewed repeats as ``gap`` candidates; return the identities."""
    from . import review

    queued: List[str] = []
    for item in summary(store):
        if item["count"] < min_count:
            continue
        identity = review.queue_candidate(store, "gap", {
            "question": item["text"],
            "language": item["language"],
            "reason": item["reasons"][0],
            "seen": item["count"],
            "near": item["near"],
        })
        if identity:
            queued.append(identity)
    return queued
