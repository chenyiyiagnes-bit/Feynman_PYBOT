"""Dream-time self-audit: notice what did not work, repair a little, stay safe.

The dreamer consolidates questions into themes; this pass asks the harder
question afterwards — *which turns went badly, and does the same trouble keep
coming back?* Only repetition earns a change (a question asked once may just be
oddly worded), and every change is one of two bounded, event-sourced moves:

* a **gap card** that records the question the cards could not answer, so the
  next dream sees it and the curriculum can teach it later;
* an **alias** attached to the card the question nearly reached, after a dry
  run proves the change against the same golden questions the test suite uses.

Nothing here is timed and nothing grows without bound. A night of heavy
cortisol is read-only: the findings are still written, the store is not.
Deleting ``data/kb_learned`` rolls every repair back, as it always could.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .feynman_kb import FeynmanKB
from .golden_queries import GOLDEN, NEGATIVES
from .learner import (
    CORRECTION_MARKERS,
    LearnedStore,
    card_id_for,
    normalize,
    similarity,
    valid_alias,
)

# Same line the learner uses before promoting anything under stress.
MAX_CORTISOL = 0.40
# A single bad turn is noise; the same trouble twice is a pattern.
REPEAT_MIN = 2
# Near-identical wordings count as the same question.
SAME_QUESTION = 0.60
# A near-tie corrected by the user is the same failure seen from the other side.
SAME_FOLLOWUP = 0.75
# Per-dream caps: a dream may tidy, never rewrite the shelf.
MAX_ACTIONS = 3
MAX_GAP_CARDS = 2
MAX_ALIASES = 3
# An alias is only worth trying when the target card was actually close.
ALIAS_MIN_SCORE = 0.22
# A context bridge leans on the previous turn, not on this question's own
# words, so it must be closer before it may attach: same subject, just a
# phrasing the cards have not seen.
BRIDGE_MIN_SCORE = 0.26
# Corrections and bridges are written the first time they are seen (the user
# asked for that), but still bounded per night.
MAX_BRIDGES = 2
MAX_CORRECTIONS = 1
CORRECTION_MIN_TEXT = 8
CORRECTION_MAX_TEXT = 600
# History is diagnostic state, not memory: keep it bounded.
MAX_HISTORY = 300
HISTORY_FILE = "reflect_history.json"
QUEUE_FILE = "self_evolution_queue.json"


@dataclass
class Finding:
    kind: str                      # unanswered | weak | broken_followup
    question: str
    count: int
    turns: List[int] = field(default_factory=list)
    source: str = ""
    alternatives: List[List[Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Action:
    kind: str                      # gap_card | alias
    target: str
    detail: str
    applied: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    findings: List[Finding] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    read_only: bool = False
    gate_ok: bool = True
    gate_failures: List[str] = field(default_factory=list)

    @property
    def repaired(self) -> int:
        return sum(1 for a in self.actions if a.applied)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "actions": [a.to_dict() for a in self.actions],
            "read_only": self.read_only,
            "gate_ok": self.gate_ok,
            "gate_failures": self.gate_failures,
            "repaired": self.repaired,
        }

    def summary(self, language: str = "zh") -> str:
        """One line for the CLI; the dream already has the full dict."""
        if language == "en":
            if not self.findings:
                return "dream audit: nothing went wrong"
            if self.read_only:
                return (f"dream audit: {len(self.findings)} trouble(s) noted, "
                        f"read-only (cortisol too high)")
            return (f"dream audit: {len(self.findings)} trouble(s) noted, "
                    f"{self.repaired} repair(s)")
        if not self.findings:
            return "梦后复盘：没有发现明显不对劲的地方"
        if self.read_only:
            return f"梦后复盘：发现 {len(self.findings)} 处问题，情绪太高只记录不改"
        return (f"梦后复盘：发现 {len(self.findings)} 处问题，"
                f"自动修了 {self.repaired} 处")


def _is_miss(source: str) -> bool:
    return source.endswith(":no_match") or source.startswith("fallback")


def _is_weak(episode: Any, source_root: str = "feynman_kb") -> bool:
    """Answered, but by a card that barely qualified or barely won.

    ``source_root`` is this package's prefix. A firm hit is ``<root>:<card>``
    and a barely-cleared one is ``<root>_soft:<card>``. The default root is
    the unnamed KB's ``feynman_kb``; a packaged mind passes its own.
    """
    retrieval = getattr(episode, "retrieval", None) or {}
    if not retrieval:
        return False
    source = getattr(episode, "source", "")
    firm = source.startswith(source_root + ":")
    soft = source.startswith(source_root + "_soft:")
    if not firm and not soft:
        return False
    if soft:
        return True
    score = retrieval.get("score")
    margin = retrieval.get("margin")
    if score is not None and score < FeynmanKB.MIN_SCORE:
        return True
    if margin is not None and margin < FeynmanKB.SOFT_MARGIN:
        return True
    return False


def _alternatives(episode: Any) -> List[List[Any]]:
    retrieval = getattr(episode, "retrieval", None) or {}
    raw = retrieval.get("alternatives") or []
    out: List[List[Any]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([str(item[0]), float(item[1])])
    return out


# -- persisted repeat counters ---------------------------------------------

def _history_path(store: LearnedStore) -> Path:
    return store.dir / HISTORY_FILE


def _load_history(store: LearnedStore) -> Dict[str, Any]:
    path = _history_path(store)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    misses = data.get("misses") if isinstance(data, dict) else None
    return misses if isinstance(misses, dict) else {}


def _save_history(store: LearnedStore, history: Dict[str, Any]) -> None:
    # Keep the most recent entries only; old trouble the user stopped asking
    # about should age out instead of growing a file forever.
    if len(history) > MAX_HISTORY:
        ordered = sorted(history.items(),
                         key=lambda kv: kv[1].get("last", 0), reverse=True)
        history = dict(ordered[:MAX_HISTORY])
    path = _history_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"misses": history}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def _history_key(history: Dict[str, Any], text: str) -> str:
    key = normalize(text)
    if key in history:
        return key
    for other in history:
        if similarity(other, key) >= SAME_QUESTION:
            return other
    return key


def _record(history: Dict[str, Any], text: str, kind: str, turn: int,
            source: str, alternatives: List[List[Any]],
            increment: bool = True) -> Finding:
    key = _history_key(history, text)
    entry = history.get(key)
    now = time.time()
    if entry is None:
        entry = {"text": text, "kind": kind, "count": 0, "turns": [],
                 "source": source, "alternatives": alternatives, "last": now}
        history[key] = entry
    # A broken follow-up is the same failure seen twice in one night, not a
    # second night of it: it relabels the finding without inflating the count.
    if increment:
        entry["count"] = int(entry.get("count", 0)) + 1
    entry["last"] = now
    entry["source"] = source or entry.get("source", "")
    if alternatives:
        entry["alternatives"] = alternatives
    turns = list(entry.get("turns") or [])
    if increment or not turns:
        turns.append(turn)
    entry["turns"] = turns[-10:]
    # A later, worse sighting outranks an earlier mild one.
    if kind == "unanswered" or entry.get("kind") not in ("unanswered", "broken_followup"):
        entry["kind"] = kind
    return Finding(kind=kind, question=entry.get("text") or text,
                   count=entry["count"], turns=entry["turns"][-5:],
                   source=source, alternatives=alternatives)


# -- golden gate ------------------------------------------------------------

def golden_ok(kb: FeynmanKB) -> Tuple[bool, List[str]]:
    """The same net the tests use: no repair may flip a known answer."""
    failures: List[str] = []
    prefix = kb.source_root
    for query, language, acceptable in GOLDEN:
        if language != kb.language:
            continue  # the other package's cards are not in this index
        _, source = kb.answer(query)
        if not source.startswith(prefix + ":"):
            failures.append(f"{query} -> {source}")
            continue
        card = source[len(prefix) + 1:]
        if card not in acceptable:
            failures.append(f"{query} -> {card}")
    for query, language in NEGATIVES:
        if language != kb.language:
            continue
        text, _ = kb.answer(query)
        if text is not None:
            failures.append(f"{query} -> answered")
    return (not failures, failures)


def _trial_kb(kb: FeynmanKB, store: LearnedStore) -> Optional[FeynmanKB]:
    """A throwaway copy so a bad alias can be tested without undoing it."""
    if kb.dir is None:
        return None
    try:
        trial = FeynmanKB(knowledge_dir=str(kb.dir),
                          learned_dir=str(store.dir),
                          assoc_path=(str(kb.assoc_path)
                                      if kb.assoc_path else None),
                          language=kb.language,
                          profile=kb.profile,
                          lexicon_dir=(str(kb.lexicon_dir)
                                       if kb.lexicon_dir else None),
                          source_prefix=kb.source_prefix,
                          spreading=kb.spreading)
    except Exception:
        return None
    return trial if trial.available() else None


# -- repairs ----------------------------------------------------------------

def _gap_body(question: str) -> str:
    return f"你问过我「{question}」，我到现在还没弄懂。"


def _gap_exists(store: LearnedStore, question: str) -> bool:
    body = _gap_body(question)
    card_id = card_id_for(body)
    for card in store.cards:
        if card.get("id") == card_id:
            return True
        if card.get("origin") == "dream_gap":
            text = card.get("text") or card.get("body") or ""
            if similarity(question, text) >= SAME_QUESTION:
                return True
    return False


def _mint_gap(kb: FeynmanKB, store: LearnedStore, question: str,
              today: str) -> Action:
    body = _gap_body(question)
    row = {
        "id": card_id_for(body),
        "type": "learned_card",
        "kind": "learned_card",
        "title": question[:40],
        "text": body,
        "body": body,
        "tags": ["learned", "gap", "dream"],
        "aliases": [question],
        "topic": "gap",
        "origin": "dream_gap",
        "learned_at": today,
        "revisions": [],
    }
    if store.add_card(row):
        kb.reload_learned()
        return Action(kind="gap_card", target=row["id"],
                      detail=question, applied=True)
    return Action(kind="gap_card", target=row["id"], detail=question,
                  applied=False, reason="unchanged")


def _usable_alias(text: str) -> bool:
    return bool(valid_alias(text)) and len(text) <= 60


def _card_score(kb: FeynmanKB, question: str, card_id: str) -> float:
    """What the card would score for this question before any alias exists."""
    language = "en" if question.isascii() else "zh"
    try:
        hits = kb.search(question, language=language, limit=50,
                         min_score=0.0)
    except Exception:
        return 0.0
    return next((float(score) for score, doc in hits if doc.id == card_id),
                0.0)


def _corpus_card_id(source: str) -> Optional[str]:
    """The card behind a confident corpus hit, if it was one."""
    if "soft" in source or source.endswith(":no_match"):
        return None
    # Sources are "<package>:[<language>:]<card id>"; the package and the
    # language tag are the oracle's, the last field is the card.
    parts = source.split(":")
    if len(parts) < 2:
        return None
    card_id = parts[-1]
    # A gap or correction card is the mind's own note, not a source.
    return None if card_id.startswith("learned-") else card_id


def _apply_alias(kb: FeynmanKB, store: LearnedStore, question: str,
                 card_id: str, min_score: float, kind: str) -> Action:
    if not _usable_alias(question):
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False, reason="not a usable alias")
    if not card_id:
        return Action(kind=kind, target="", detail=question,
                      applied=False, reason="no card to attach to")
    if not any(d.id == card_id for d in kb.docs):
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False, reason="card no longer loaded")
    score = _card_score(kb, question, card_id)
    if score < min_score:
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False,
                      reason=f"score {score:.2f} below {min_score}")
    trial = _trial_kb(kb, store)
    if trial is None:
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False, reason="could not build trial retrieval")
    trial.learn_alias(card_id, question, persist=False)
    ok, failures = golden_ok(trial)
    if not ok:
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False,
                      reason="golden gate: " + "; ".join(failures[:2]))
    # The alias must actually make the question land on that card.
    _, source = trial.answer(question)
    prefix = trial.source_root + ":"
    if not source.startswith(prefix) or source[len(prefix):] != card_id:
        return Action(kind=kind, target=card_id, detail=question,
                      applied=False, reason="alias did not change the answer")
    kb.learn_alias(card_id, question)
    return Action(kind=kind, target=card_id, detail=question, applied=True)


def _try_alias(kb: FeynmanKB, store: LearnedStore, question: str,
               alternatives: List[List[Any]]) -> Action:
    """Attach the wording to the card it almost reached, if that is safe."""
    best_id, best_score = "", 0.0
    for card_id, score in alternatives:
        if not isinstance(card_id, str) or card_id.startswith("learned-"):
            # Never teach a gap card a new question: it has no answer yet.
            continue
        best_id, best_score = card_id, float(score)
        break
    if not best_id:
        return Action(kind="alias", target="", detail=question,
                      applied=False, reason="no corpus card was close")
    return _apply_alias(kb, store, question, best_id, ALIAS_MIN_SCORE,
                        "alias")


def _try_bridge(kb: FeynmanKB, store: LearnedStore, question: str,
                card_id: str) -> Action:
    """A follow-up question missed; the previous answer was about it."""
    return _apply_alias(kb, store, question, card_id, BRIDGE_MIN_SCORE,
                        "bridge_alias")


def _trial_copy(kb: FeynmanKB, store: LearnedStore
                ) -> Tuple[Optional[FeynmanKB], Optional[Path]]:
    """A full throwaway copy of the learned store, for testing a new card."""
    if kb.dir is None:
        return None, None
    tmp = Path(tempfile.mkdtemp(prefix="omega-reflect-card-"))
    if store.dir.exists():
        for item in store.dir.iterdir():
            if item.is_file():
                shutil.copy2(item, tmp / item.name)
    try:
        trial = FeynmanKB(
            knowledge_dir=str(kb.dir),
            learned_dir=str(tmp),
            assoc_path=(str(kb.assoc_path) if kb.assoc_path else None),
            language=kb.language,
            profile=kb.profile,
            lexicon_dir=(str(kb.lexicon_dir) if kb.lexicon_dir else None),
            source_prefix=kb.source_prefix,
            spreading=kb.spreading)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None
    if not trial.available():
        shutil.rmtree(tmp, ignore_errors=True)
        return None, None
    return trial, tmp


def _card_by_body(store: LearnedStore, body: str) -> Optional[Dict[str, Any]]:
    for card in store.cards:
        if (card.get("body") or card.get("text")) == body:
            return card
    return None


def _clean_correction(text: str) -> str:
    """Drop the "不对，其实…" wrapper so the stored answer reads cleanly."""
    body = text.strip()
    for marker in sorted(CORRECTION_MARKERS, key=len, reverse=True):
        if body.startswith(marker):
            body = body[len(marker):].lstrip("，,。.！!？? 　")
    for lead in ("其实", "实际上", "我是说", "我说的是"):
        if body.startswith(lead):
            body = body[len(lead):].lstrip("，,。.！!？? 　")
    return body.strip() if len(body.strip()) >= CORRECTION_MIN_TEXT \
        else text.strip()


def _mint_correction(kb: FeynmanKB, store: LearnedStore, question: str,
                     correction: str, today: str) -> Action:
    """Store the user's own correction as a card the first time it is given.

    It is written immediately (the user asked for that), but it is marked as
    an unverified user claim and carries a lower retrieval boost: a single
    correction is evidence, not an established fact. The golden net still
    has to pass, and the corrected question must really land on the new card.
    """
    body = _clean_correction(correction)
    if not question.strip() or len(body) < CORRECTION_MIN_TEXT \
            or len(body) > CORRECTION_MAX_TEXT:
        return Action(kind="correction_card", target="", detail=question,
                      applied=False, reason="correction too short to store")
    aliases = [question.strip()] if _usable_alias(question.strip()) else []
    row = {
        "id": card_id_for(body),
        "type": "learned_card",
        "kind": "learned_card",
        "title": question.strip()[:40],
        "text": body,
        "body": body,
        "tags": ["learned", "correction", "dream"],
        "aliases": aliases,
        "topic": "correction",
        "origin": "dream_correction",
        "source": "dream audit: user correction",
        "source_class": "user_claim",
        "verification_status": "user_claim",
        "source_note": ("captured from the user's correction during a dream "
                        "audit; not independently verified"),
        "retrieval_boost": 0.8,
        "learned_at": today,
        "revisions": [],
    }
    trial, tmp = _trial_copy(kb, store)
    if trial is None or tmp is None:
        return Action(kind="correction_card", target=row["id"],
                      detail=question, applied=False,
                      reason="could not build trial retrieval")
    try:
        if not trial.learned.add_card(row):
            return Action(kind="correction_card", target=row["id"],
                          detail=question, applied=False, reason="unchanged")
        trial.reload_learned()
        card = _card_by_body(trial.learned, body)
        if card is None:
            return Action(kind="correction_card", target=row["id"],
                          detail=question, applied=False,
                          reason="card did not load")
        ok, failures = golden_ok(trial)
        if not ok:
            return Action(kind="correction_card", target=str(card["id"]),
                          detail=question, applied=False,
                          reason="golden gate: " + "; ".join(failures[:2]))
        _, source = trial.answer(question.strip())
        if not source.endswith(":" + str(card["id"])):
            return Action(kind="correction_card", target=str(card["id"]),
                          detail=question, applied=False,
                          reason="corrected question still misses the card")
        card_id = str(card["id"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not store.add_card(row):
        return Action(kind="correction_card", target=card_id,
                      detail=question, applied=False, reason="unchanged")
    kb.reload_learned()
    return Action(kind="correction_card", target=card_id,
                  detail=question, applied=True)


# -- the pass ---------------------------------------------------------------

def audit(kb: FeynmanKB, store: LearnedStore, batch: List[Any],
          cortisol: float = 0.0, today: Optional[str] = None,
          prior: Any = None) -> AuditReport:
    """Look over one dream's episodes and repair what went wrong.

    ``batch`` is captured before consolidation flips ``dreamed``; the caller
    passes the episodes exactly as they were. ``prior`` is the last episode
    before the batch, so a follow-up at the start of the night still has the
    turn it follows.

    Two failures are written the first time they are seen (the user chose
    that): a context bridge from a miss back to the previous answer, and a
    correction card built from the user's own words. Everything else still
    waits for repetition.
    """
    today = today or time.strftime("%Y-%m-%d")
    report = AuditReport()
    history = _load_history(store)
    ordered = sorted(batch, key=lambda e: getattr(e, "turn", 0))
    sequence = ([prior] if prior is not None else []) + ordered
    batch_ids = {id(e) for e in ordered}
    # A gap card says "I still don't know"; being answered by one is not an
    # answer, so it keeps counting as the same unanswered question.
    gap_ids = {str(c.get("id")) for c in store.cards
               if c.get("origin") == "dream_gap"}
    seen: Dict[str, Finding] = {}

    for episode in ordered:
        text = str(getattr(episode, "text", "") or "").strip()
        if not text:
            continue
        source = str(getattr(episode, "source", "") or "")
        answered_by_gap = source.rsplit(":", 1)[-1] in gap_ids
        if _is_miss(source) or answered_by_gap:
            finding = _record(history, text, "unanswered",
                              getattr(episode, "turn", 0), source,
                              _alternatives(episode))
        elif _is_weak(episode, kb.source_root):
            finding = _record(history, text, "weak",
                              getattr(episode, "turn", 0), source,
                              _alternatives(episode))
        else:
            continue
        seen[finding.question] = finding

    # -- first-sight repairs: a correction, then a context bridge -----------
    corrections: List[Tuple[Any, Any]] = []
    previous_of: Dict[int, Any] = {}
    for prev, nxt in zip(sequence, sequence[1:]):
        previous_of[id(nxt)] = prev
        if id(nxt) not in batch_ids:
            continue
        follow = str(getattr(nxt, "text", "") or "").strip()
        if not follow:
            continue
        prev_source = str(getattr(prev, "source", "") or "")
        prev_text = str(getattr(prev, "text", "") or "").strip()
        if _is_miss(prev_source) or _is_weak(prev, kb.source_root):
            if any(mark in follow for mark in CORRECTION_MARKERS):
                corrections.append((prev, nxt))
            if prev_text and (
                    any(mark in follow for mark in CORRECTION_MARKERS)
                    or similarity(normalize(follow),
                                  normalize(prev_text)) >= SAME_FOLLOWUP):
                # A miss followed by "不对" or by the same question again is
                # one failure with two turns, not two nights of it.
                finding = _record(history, prev_text, "broken_followup",
                                  getattr(prev, "turn", 0), prev_source,
                                  _alternatives(prev), increment=False)
                seen[finding.question] = finding

    _save_history(store, history)

    # The queue is written every night, read-only or not: it is the honest
    # list of what this mind cannot yet answer, food for the next curriculum.
    report.findings = sorted(seen.values(), key=lambda f: f.count, reverse=True)
    _write_queue(store, report.findings, today)

    report.read_only = cortisol > MAX_CORTISOL
    # Baseline: the shelf must already pass before any repair may touch it.
    gate_ok, gate_failures = golden_ok(kb)
    report.gate_ok = gate_ok
    report.gate_failures = gate_failures
    if not gate_ok:
        return report

    budget = MAX_ACTIONS
    # Corrections come first: the user's own words are the best evidence of
    # what the question meant. Then bridges, then repairs that wait for
    # repetition.
    for count, (prev, nxt) in enumerate(corrections):
        if budget <= 0 or count >= MAX_CORRECTIONS:
            break
        question = str(getattr(prev, "text", "") or "").strip()
        if report.read_only:
            report.actions.append(Action(
                kind="correction_card", target="", detail=question,
                applied=False, reason="read-only: cortisol above limit"))
        else:
            report.actions.append(_mint_correction(
                kb, store, question,
                str(getattr(nxt, "text", "") or ""), today))
        budget -= 1

    bridges = 0
    if not report.read_only:
        for episode in ordered:
            if budget <= 0 or bridges >= MAX_BRIDGES:
                break
            source = str(getattr(episode, "source", "") or "")
            text = str(getattr(episode, "text", "") or "").strip()
            if not text or not _is_miss(source):
                continue
            if any(mark in text for mark in CORRECTION_MARKERS):
                continue  # that is a follow-up, not a new question
            prev = previous_of.get(id(episode))
            if prev is None:
                continue
            card_id = _corpus_card_id(str(getattr(prev, "source", "") or ""))
            if card_id is None:
                continue
            action = _try_bridge(kb, store, text, card_id)
            report.actions.append(action)
            bridges += 1
            budget -= 1

    gaps = aliases = 0
    for finding in report.findings:
        if budget <= 0:
            break
        if finding.count < REPEAT_MIN:
            continue
        if report.read_only:
            report.actions.append(Action(
                kind="gap_card", target="", detail=finding.question,
                applied=False, reason="read-only: cortisol above limit"))
            budget -= 1
            continue
        action: Optional[Action] = None
        if gaps < MAX_GAP_CARDS and not _gap_exists(store, finding.question):
            action = _mint_gap(kb, store, finding.question, today)
            gaps += 1
        elif aliases < MAX_ALIASES and finding.alternatives:
            action = _try_alias(kb, store, finding.question,
                                finding.alternatives)
            aliases += 1
        if action is not None:
            report.actions.append(action)
            budget -= 1
    return report


def _write_queue(store: LearnedStore, findings: List[Finding],
                 today: str) -> None:
    path = store.dir / QUEUE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [
        {"question": f.question, "kind": f.kind, "count": f.count,
         "turns": f.turns, "alternatives": f.alternatives}
        for f in findings
    ]
    try:
        path.write_text(json.dumps({"updated": today, "items": items},
                                   ensure_ascii=False, indent=1),
                        encoding="utf-8")
    except OSError:
        pass


def load_queue(store: Optional[LearnedStore] = None) -> List[Dict[str, Any]]:
    """Read the last queue back (the curriculum script prints it)."""
    directory = store.dir if store is not None else \
        Path(__file__).resolve().parent / "data" / "kb_learned"
    path = Path(directory) / QUEUE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = data.get("items") if isinstance(data, dict) else None
    return items if isinstance(items, list) else []
