"""Package-local candidate journal and review decisions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

FILE = "review.json"
MAX_PENDING = 200


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(store) -> dict:
    if store is not None:
        path = store.dir / FILE
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"candidates": {}, "decisions": []}


@contextmanager
def edit(store):
    store.dir.mkdir(parents=True, exist_ok=True)
    with (store.dir / "review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read(store)
        original = json.dumps(data, ensure_ascii=False, sort_keys=True)
        yield data
        if json.dumps(data, ensure_ascii=False, sort_keys=True) != original:
            write(store.dir, data)


def write(directory: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix=".review-", delete=False) as handle:
        try:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(handle.name, directory / FILE)
        finally:
            Path(handle.name).unlink(missing_ok=True)


def queue_candidate(store, kind: str, payload: dict) -> str | None:
    if store is None:
        return None
    identity = digest(json.dumps([kind, payload], sort_keys=True, ensure_ascii=False))
    try:
        with edit(store) as data:
            candidates = data["candidates"]
            candidates.setdefault(identity, {"kind": kind, **payload})
            decided = {row["id"] for row in data["decisions"]}
            pending = [cid for cid in candidates if cid not in decided]
            for cid in pending[:-MAX_PENDING]:
                del candidates[cid]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return identity


def active(data: dict) -> list[dict]:
    decisions = {}
    for row in data["decisions"]:
        if row["action"] == "approve":
            decisions[row["id"]] = row
        elif row["action"] == "revoke":
            decisions.pop(row["id"], None)
    return list(decisions.values())
