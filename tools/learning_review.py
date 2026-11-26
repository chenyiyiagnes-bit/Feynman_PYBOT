"""Inspect, rehearse, approve or revoke a package's local learning decisions."""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.runtime import build
from omega_mind import learner, reflect, review
from tools.card_support import _check_entry


def prepare(kb, candidate: dict, proof: dict, path: Path, identity: str) -> dict:
    if not all(isinstance(proof.get(key), str) and proof[key].strip()
               for key in ("reviewer", "source", "quote", "quote_sha256", "source_file")):
        raise ValueError("proof needs reviewer, source, source_file, quote and quote_sha256")
    status = _check_entry(proof, [path])
    if status["status"] != "valid":
        raise ValueError(status["problem"])
    decision = {"id": identity, "action": "approve", "reviewer": proof["reviewer"],
                "source": proof["source"], "quote_sha256": proof["quote_sha256"],
                "recorded_at": datetime.now(timezone.utc).isoformat()}
    kind = candidate["kind"]
    if kind == "alias":
        cid, phrase = candidate["card_id"], candidate["phrase"]
        if cid not in kb.by_id or not kb.public_doc(kb.by_id[cid]) or not learner.valid_alias(phrase):
            raise ValueError("alias needs a live public card and a valid phrase")
        decision["alias"] = [cid, phrase]
    elif kind in ("correction", "analogy"):
        answer = candidate["text"]
        title = candidate.get("question", proof.get("title", ""))
        if not isinstance(title, str) or not title.strip():
            raise ValueError("analogy proof needs a title that retrieves this card")
        decision["row"] = {
            "id": "learned-review-" + identity[:20], "type": "learned_card",
            "title": title, "aliases": [title], "body": answer, "text": answer,
            "source": proof["source"], "source_class": "user_claim",
            "verification_status": "reviewed_local", "origin": "review",
        }
    else:
        raise ValueError("knowledge gaps require editing a candidate with evidence before review")
    return decision


def _actual(kb, question: str) -> str | None:
    result = kb.lookup(question, public_only=True, structured=True)
    return result.source.partition(":")[2] if result.text else None


def rehearse(kb, data: dict, decision: dict, cases: list[dict]) -> None:
    if not cases or any(not isinstance(c.get("question"), str) or "card" not in c for c in cases):
        raise ValueError("holdout must contain questions and expected card IDs or null")
    row = decision.get("row")
    query = row["title"] if row else decision["alias"][1]
    expected = row["id"] if row else decision["alias"][0]
    if any(case["question"].casefold().strip() == query.casefold().strip()
           or case["card"] == expected for case in cases):
        raise ValueError("holdout overlaps the candidate question or card")
    baseline = [_actual(kb, case["question"]) for case in cases]
    if not any(expected == actual for expected, actual in
               zip((case["card"] for case in cases), baseline)):
        raise ValueError("holdout has no passing baseline case to protect")
    original_failures = set(reflect.golden_ok(kb)[1])
    trial, scratch = reflect._trial_copy(kb, kb.learned)
    if trial is None or scratch is None:
        raise ValueError("could not construct an isolated trial")
    try:
        trial_data = {**data, "decisions": [*data["decisions"], decision]}
        review.write(scratch, trial_data)
        trial.refresh_review()
        changed = [_actual(trial, case["question"]) for case in cases]
        if _actual(trial, query) != expected:
            raise ValueError("candidate question does not reach its reviewed answer")
        if any(before == case["card"] and after != before
               for case, before, after in zip(cases, baseline, changed)):
            raise ValueError("holdout regression")
        if set(reflect.golden_ok(trial)[1]) - original_failures:
            raise ValueError("golden retrieval regression")
    finally:
        shutil.rmtree(scratch)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package")
    parser.add_argument("action", choices=("inspect", "replay", "approve", "revoke"))
    parser.add_argument("--id")
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--holdout", type=Path)
    args = parser.parse_args(argv)
    try:
        engine = build(args.package)
        store = engine.kb.learned
        if store is None:
            raise ValueError("this package has no learned store")
        if args.action == "inspect":
            print(json.dumps(review.read(store), ensure_ascii=False, indent=2))
            return 0
        if not args.id:
            raise ValueError("--id required")
        with review.edit(store) as data:
            if args.action == "revoke":
                if args.id not in {d["id"] for d in review.active(data)}:
                    raise ValueError("no active review with this ID")
                data["decisions"].append({"id": args.id, "action": "revoke",
                                          "recorded_at": datetime.now(timezone.utc).isoformat()})
                return 0
            if not args.proof or not args.holdout:
                raise ValueError("--proof and --holdout required")
            proof = json.loads(args.proof.read_text(encoding="utf-8"))
            cases = json.loads(args.holdout.read_text(encoding="utf-8"))
            engine.kb.refresh_review()
            if args.action == "approve" and args.id in {d["id"] for d in review.active(data)}:
                raise ValueError("this candidate is already active")
            decision = prepare(engine.kb, data["candidates"][args.id], proof,
                               args.proof.parent, args.id)
            rehearse(engine.kb, data, decision, cases)
            if args.action == "approve":
                data["decisions"].append(decision)
        print(f"{args.action}: PASS {args.id}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
