"""What the mind keeps from a conversation, and how it forgets.

Three ideas, all plain JSON and plain arithmetic — no weights:

  * one append-only event log (``experience.jsonl``) is the source of truth;
    cards and aliases are projections replayed from it, so any rule change or
    mistake can be re-derived or rolled back by editing events;
  * a similar new metaphor is a *revision* of the card it resembles, not a
    second card — the old wording stays in ``revisions`` for audit;
  * the card cap is soft. At the limit the least useful card is archived
    (utility = hits + 2*praise + aliases - a little age decay), never silently
    dropped, and it can be restored.

Deleting the learned directory is still a complete rollback.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LEARNED_DIR = Path(__file__).resolve().parent / "data" / "kb_learned"
EVENTS_FILE = "experience.jsonl"
CARDS_FILE = "learned_cards.jsonl"
ALIASES_FILE = "learned_aliases.json"
STATS_FILE = "card_stats.json"

MAX_CARDS = 200
MAX_ALIASES_PER_CARD = 20
MAX_REVISIONS = 10
MIN_TEXT = 8
MAX_TEXT = 240
MERGE_SIMILARITY = 0.55
PROMOTION_TTL = 3

# Explicit teaching: the user says "think of it as / it is like". These mint
# a card immediately.
STRICT_MARKERS = ("像", "好比", "想象成", "想象为", "可以想成", "就像")
# A looser comparison. These wait for confirmation (a reference or praise on
# a later turn), because they are also how people hedge and correct.
LOOSE_MARKERS = ("相当于", "等于", "可以看作", "看作", "比作",
                 "打个比方", "打比方", "仿佛", "好像")
# Phrases that point at an earlier comparison instead of making a new one.
REFERENCE_GUARDS = ("像你说的", "像刚才", "就像刚才", "不是像", "不像",
                    "打个比方说过了")
CORRECTION_MARKERS = ("不是这个意思", "不对", "说错了", "我不是说",
                      "别再", "不要这么说", "理解错", "你误会")

# "像" in these constructions points at a person or a demonstrative, not at
# an analogy: "像你这么聪明的人" is praise. Cardinality and quotative uses are
# stripped before the marker check, so praise can never mint a card.
ADDRESS_PATTERNS = (
    re.compile(r"像(?:你|您|我|他|她|它|我们|你们|他们|她们|它们|大家|别人|人家|这|那)"),
    re.compile(r"像[^，。！？,!?]{0,8}的人"),
)

# A confirmation must quote the comparison, not merely mention it in a long
# reply. Short turns that contain the whole candidate, or a quoted run of it,
# count; an essay that happens to include the words does not.
CONFIRM_MAX_LEN = 80

# Implicit promotion (praise, paraphrasing) is blocked while the mind is
# suspicious: at that level a repeated phrase is being challenged, not
# endorsed. Same 0.40 the left brain uses for its own guard.
PROMOTION_CORTISOL_MAX = 0.40

# Sleep-time absorption band: similar enough to be one idea, too similar for
# the daytime merge (>= MERGE_SIMILARITY revises on sight).
SLEEP_MERGE_FLOOR = 0.35
MAX_MERGES_PER_NIGHT = 5
# Absorption is destructive, so the host must already be more useful than the
# victim. Two never-used cards at ~0 utility are left alone: the newer wording
# is all either of them has.
MIN_MERGE_UTILITY_GAP = 1.0

# At least this many cards per topic survive an eviction round, so a burst of
# new physics metaphors cannot wipe art out of memory.
TOPIC_FLOOR_CAP = 5


def normalize(text: str) -> str:
    """Case-, width- and punctuation-insensitive form used for dedupe."""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s\W_]+", "", text)


def card_id_for(text: str) -> str:
    return "learned-" + hashlib.sha1(
        normalize(text).encode("utf-8")).hexdigest()[:10]


# Words that merely contain a comparison character ("形象", "图像") are not
# comparisons; strip them before looking for markers.
FALSE_FRIENDS = ("形象", "图像", "影像", "想像", "偶像", "画像", "肖像",
                 "像素", "镜像", "成像")


def _marker_text(text: str, strict: bool = False) -> str:
    for word in FALSE_FRIENDS:
        text = text.replace(word, " ")
    for pattern in ADDRESS_PATTERNS:
        text = pattern.sub(" ", text)
    if strict:
        # "好像" is a hedge, not an explicit comparison; "就像" still counts.
        text = text.replace("好像", " ")
    return text


def _is_question(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith(("？", "?", "吗", "呢")) or "是不是" in stripped


def _guarded(text: str) -> bool:
    return any(g in text for g in REFERENCE_GUARDS)


def looks_like_metaphor(text: str) -> bool:
    """An explicit comparison the user made, not a question or a fact."""
    body = text.strip()
    if not (MIN_TEXT <= len(body) <= MAX_TEXT):
        return False
    if _is_question(body) or _guarded(body):
        return False
    marked = _marker_text(body, strict=True)
    return any(marker in marked for marker in STRICT_MARKERS)


def metaphor_candidate(text: str) -> bool:
    """A possible comparison, loose markers included. Needs confirmation."""
    body = text.strip()
    if not (MIN_TEXT <= len(body) <= MAX_TEXT):
        return False
    if _is_question(body) or _guarded(body):
        return False
    marked = _marker_text(body)
    return any(marker in marked for marker in STRICT_MARKERS + LOOSE_MARKERS)


def _body_of(row: Dict[str, Any]) -> str:
    """The user's own words, without the learned-card template around them."""
    body = row.get("body")
    if isinstance(body, str) and body.strip():
        return body.strip()
    text = str(row.get("text") or "")
    marker = "："
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _bigrams(text: str) -> set:
    flat = normalize(text)
    if len(flat) < 2:
        return set()
    return {flat[i:i + 2] for i in range(len(flat) - 1)}


def similarity(a: str, b: str) -> float:
    """Cosine over character bigrams — the same evidence retrieval uses."""
    left, right = _bigrams(a), _bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def shared_pairs(a: str, b: str) -> int:
    """How many character bigrams two utterances have in common."""
    return len(_bigrams(a) & _bigrams(b))


def explicit_confirmation(text: str, candidate: str) -> bool:
    """The user repeats a pending comparison whole, or quotes it.

    Containment in a short turn ("对，就是那条首尾相扣的蜡封信件…") is the
    strongest signal a rule-based system can have. A long turn that merely
    happens to include the same words is not: it is usually the topic being
    discussed, not the comparison being endorsed.
    """
    body = text.strip()
    if not body or len(body) > max(CONFIRM_MAX_LEN, 2 * len(candidate.strip())):
        return False
    want = normalize(candidate)
    have = normalize(body)
    if len(want) < 10 or not have:
        return False
    if want in have or have in want:
        return True
    # A short turn that restates nearly all of the comparison is a repeat,
    # even when it drops the subject at the front ("对，就是…").
    if similarity(body, candidate) >= 0.70:
        return True
    for open_q, close_q in (("“", "”"), ("「", "」"), ("『", "』")):
        start = body.find(open_q)
        while start != -1:
            end = body.find(close_q, start + 1)
            if end == -1:
                break
            span = body[start + 1:end]
            if len(normalize(span)) >= 10 and similarity(span, candidate) >= 0.6:
                return True
            start = body.find(open_q, end + 1)
    return False


def _clauses(body: str) -> List[str]:
    """Content clauses of a comparison, each a possible way back to it.

    The title is only the first clause; a user who remembers the *end* of the
    metaphor ("改一封就得重盖章") has no subject word to match on. Every
    clause becomes an alias at mint time, so either half can find the card.
    """
    parts = re.split(r"[，。；！？,;!?]", body)
    out: List[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 4:
            continue
        if any(normalize(part) == normalize(seen) for seen in out):
            continue
        out.append(part)
    return out[:4]


def card_row(text: str, origin: str = "conversation",
             topic: str = "general") -> Optional[Dict[str, Any]]:
    """Build a knowledge row in the same schema the loader already reads."""
    body = text.strip()
    if not metaphor_candidate(body):
        return None
    title = body.split("，", 1)[0].split("。", 1)[0][:40]
    aliases = [title]
    for clause in _clauses(body):
        if normalize(clause) != normalize(title):
            aliases.append(clause)
    return {
        "id": card_id_for(body),
        "type": "learned_card",
        "title": title,
        "body": body,
        "text": f"你上次教过我一个比喻，我一直记着：{body}",
        "aliases": aliases,
        "tags": ["learned", "metaphor", origin],
        "topic": topic or "general",
        "learned_at": date.today().isoformat(),
        "origin": origin,
        "revisions": [],
    }


def valid_alias(phrase: str) -> bool:
    phrase = phrase.strip()
    return 4 <= len(phrase) <= 60 and bool(re.search(r"[一-鿿a-zA-Z]", phrase))


class LearnedStore:
    """Event log on disk, projections in memory; tolerates a missing dir."""

    def __init__(self, directory: Optional[Path] = None):
        self.dir = Path(directory) if directory else LEARNED_DIR
        self.cards: List[Dict[str, Any]] = []
        self.aliases: Dict[str, List[str]] = {}
        self.archived: List[Dict[str, Any]] = []
        self.stats: Dict[str, Dict[str, int]] = {}
        # Cards minted this session. A newly taught comparison is never the
        # first thing evicted: the user just told us it matters.
        self._fresh: set = set()
        self.load()

    # -- disk -------------------------------------------------------------

    def load(self) -> None:
        self.cards = []
        self.aliases = {}
        self.archived = []
        events = self._read_events()
        if events is None:
            events = self._migrate_legacy()
            if events:
                self._write_events(events)
        self._reduce(events)
        self._load_stats()

    def _read_events(self) -> Optional[List[Dict[str, Any]]]:
        path = self.dir / EVENTS_FILE
        if not path.exists():
            return None
        events: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("kind"):
                events.append(event)
        return events

    def _write_events(self, events: List[Dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / EVENTS_FILE
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events),
            encoding="utf-8")
        tmp.replace(path)

    def _migrate_legacy(self) -> List[Dict[str, Any]]:
        """Old two-file format becomes the first events of the log."""
        events: List[Dict[str, Any]] = []
        cards_path = self.dir / CARDS_FILE
        if cards_path.exists():
            for line in cards_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("id") and row.get("text"):
                    events.append({"kind": "card", "row": row})
        aliases_path = self.dir / ALIASES_FILE
        if aliases_path.exists():
            try:
                data = json.loads(aliases_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            if isinstance(data, dict):
                for cid, phrases in data.items():
                    if not isinstance(phrases, list):
                        continue
                    for phrase in phrases:
                        if isinstance(phrase, str) and valid_alias(phrase):
                            events.append({
                                "kind": "alias",
                                "card_id": str(cid),
                                "phrase": phrase,
                            })
        return events

    def _load_stats(self) -> None:
        self.stats = {}
        path = self.dir / STATS_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(data, dict):
            for cid, counts in data.items():
                if isinstance(counts, dict):
                    self.stats[str(cid)] = {
                        k: int(v) for k, v in counts.items()
                        if isinstance(v, (int, float))
                    }

    def _write_stats(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / STATS_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.stats, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

    # -- projection -------------------------------------------------------

    def _reduce(self, events: List[Dict[str, Any]]) -> None:
        by_id: Dict[str, Dict[str, Any]] = {}
        aliases: Dict[str, List[str]] = {}
        archived: Dict[str, Dict[str, Any]] = {}
        for event in events:
            kind = event.get("kind")
            if kind == "card":
                row = event.get("row") or {}
                cid = row.get("id")
                if not cid or not row.get("text"):
                    continue
                row = dict(row)
                row.setdefault("revisions", [])
                by_id[str(cid)] = row
                archived.pop(str(cid), None)
            elif kind == "revision":
                cid = str(event.get("card_id") or "")
                card = by_id.get(cid)
                if card is None:
                    continue
                revisions = list(card.get("revisions") or [])
                revisions.append({
                    "text": card.get("text"),
                    "body": _body_of(card),
                    "title": card.get("title"),
                })
                card["revisions"] = revisions[-MAX_REVISIONS:]
                for field in ("text", "body", "title"):
                    if event.get(field):
                        card[field] = event[field]
                card["updated_at"] = event.get("ts") or date.today().isoformat()
                tags = set(card.get("tags") or []) | set(event.get("tags") or [])
                card["tags"] = sorted(tags)
            elif kind == "alias":
                cid = str(event.get("card_id") or "")
                phrase = str(event.get("phrase") or "").strip()
                if not valid_alias(phrase):
                    continue
                phrases = aliases.setdefault(cid, [])
                key = normalize(phrase)
                if any(normalize(p) == key for p in phrases):
                    continue
                if len(phrases) < MAX_ALIASES_PER_CARD:
                    phrases.append(phrase)
            elif kind == "archive":
                cid = str(event.get("card_id") or "")
                card = by_id.pop(cid, None)
                if card is not None:
                    card["archived_at"] = event.get("ts")
                    card["archive_reason"] = event.get("reason", "utility")
                    if event.get("merged_into"):
                        card["merged_into"] = event["merged_into"]
                    archived[cid] = card
            elif kind == "restore":
                cid = str(event.get("card_id") or "")
                card = archived.pop(cid, None)
                if card is not None:
                    card.pop("archived_at", None)
                    card.pop("archive_reason", None)
                    by_id[cid] = card
        self.cards = list(by_id.values())
        self.aliases = aliases
        self.archived = list(archived.values())

    # -- materialized views (for humans and legacy tools) -----------------

    def _write_cards(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / CARDS_FILE
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.cards),
            encoding="utf-8")
        tmp.replace(path)

    def _write_aliases(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / ALIASES_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.aliases, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)

    def _append(self, event: Dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / EVENTS_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # -- writes -----------------------------------------------------------

    def add_card(self, row: Dict[str, Any]) -> bool:
        """Add a card, revise a resembling one, or do nothing if unchanged."""
        if not row.get("id") or not row.get("text"):
            return False
        body = _body_of(row)
        existing = next((c for c in self.cards if c.get("id") == row["id"]),
                        None)
        if existing is not None:
            if _body_of(existing) == body:
                return False
            return self._revise(existing, row)
        best = max(self.cards, key=lambda c: similarity(body, _body_of(c)),
                   default=None)
        if best is not None and similarity(body, _body_of(best)) >= MERGE_SIMILARITY:
            return self._revise(best, row)
        row = dict(row)
        row.setdefault("revisions", [])
        self._append({"kind": "card", "row": row})
        self.cards.append(row)
        self._fresh.add(str(row["id"]))
        self._enforce_capacity()
        self._write_cards()
        return True

    def _revise(self, card: Dict[str, Any], row: Dict[str, Any]) -> bool:
        row = dict(row)
        self._append({
            "kind": "revision",
            "card_id": card["id"],
            "text": row.get("text"),
            "body": _body_of(row),
            "title": row.get("title"),
            "tags": row.get("tags") or [],
            "ts": date.today().isoformat(),
        })
        old_body = _body_of(card)
        revisions = list(card.get("revisions") or [])
        revisions.append({"text": card.get("text"), "body": old_body,
                          "title": card.get("title")})
        card["revisions"] = revisions[-MAX_REVISIONS:]
        card["text"] = row.get("text")
        card["body"] = _body_of(row)
        card["title"] = row.get("title")
        card["updated_at"] = date.today().isoformat()
        card["tags"] = sorted(set(card.get("tags") or []) | set(row.get("tags") or []))
        # The old phrasing stays findable; both spellings point at one card.
        # So do the revised row's own clause aliases: a user may paraphrase
        # either half of the new wording on a later turn.
        phrases = [old_body, row.get("title")]
        phrases.extend(row.get("aliases") or [])
        for phrase in phrases:
            if isinstance(phrase, str):
                self.add_alias(card["id"], phrase)
        self._write_cards()
        return True

    def utility(self, card: Dict[str, Any]) -> float:
        counts = self.stats.get(str(card.get("id")), {})
        hits = counts.get("hits", 0)
        praise = counts.get("praise", 0)
        aliases = len(self.aliases.get(str(card.get("id")), []))
        days = 0.0
        raw = card.get("learned_at")
        if isinstance(raw, str):
            try:
                days = max(0.0, (date.today() - date.fromisoformat(raw)).days)
            except ValueError:
                days = 0.0
        return hits + 2.0 * praise + 0.2 * aliases - 0.05 * days

    def _enforce_capacity(self) -> None:
        """Keep at most MAX_CARDS, with per-topic floors and a mint grace.

        Eviction is soft: the victim is archived with its reason, never
        dropped. A topic that still has more cards than its floor can lose
        one; a single-card topic is only evicted when nothing else is left.
        Cards minted this session sit out the round.
        """
        while len(self.cards) > MAX_CARDS:
            victim = self._pick_victim()
            if victim is None:
                return
            cid = str(victim.get("id"))
            self._append({"kind": "archive", "card_id": cid,
                          "reason": "utility", "ts": date.today().isoformat()})
            self.cards.remove(victim)
            self.archived.append({**victim,
                                  "archive_reason": "utility",
                                  "archived_at": date.today().isoformat()})

    @staticmethod
    def topic_of(card: Dict[str, Any]) -> str:
        topic = card.get("topic")
        return topic if isinstance(topic, str) and topic else "general"

    def _pick_victim(self) -> Optional[Dict[str, Any]]:
        pool = [c for c in self.cards if c.get("id") not in self._fresh]
        if not pool:
            pool = list(self.cards)  # everything is new; something must go
        by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for card in pool:
            by_topic.setdefault(self.topic_of(card), []).append(card)
        quota = max(1, min(TOPIC_FLOOR_CAP,
                           MAX_CARDS // max(1, 2 * len(by_topic))))
        protected = set()
        for cards in by_topic.values():
            keep = max(1, min(quota, len(cards)))
            for card in sorted(cards, key=self.utility, reverse=True)[:keep]:
                protected.add(str(card.get("id")))
        evictable = [c for c in pool if str(c.get("id")) not in protected]
        if not evictable:
            evictable = pool
        return min(evictable, key=self.utility, default=None)

    def sleep_merge(self, limit: int = MAX_MERGES_PER_NIGHT) -> List[Dict[str, Any]]:
        """Absorb near-duplicate cards of the same topic while the mind sleeps.

        The daytime path revises at >= MERGE_SIMILARITY; this is the band just
        below it, where two cards say nearly the same thing in different
        words. The victim's body becomes an alias of the host and the victim
        is archived with ``merged_into`` — the event log keeps it auditable and
        restorable. No ``revision`` is emitted: the host's own text did not
        change, and pretending it did would corrupt ``revisions``.
        """
        candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for i, first in enumerate(self.cards):
            for second in self.cards[i + 1:]:
                if self.topic_of(first) != self.topic_of(second):
                    continue
                sim = similarity(_body_of(first), _body_of(second))
                if not (SLEEP_MERGE_FLOOR <= sim < MERGE_SIMILARITY):
                    continue
                if self.utility(first) >= self.utility(second):
                    host, victim = first, second
                else:
                    host, victim = second, first
                if self.utility(host) - self.utility(victim) < MIN_MERGE_UTILITY_GAP:
                    continue
                candidates.append((sim, host, victim))
        candidates.sort(key=lambda item: -item[0])
        merged: List[Dict[str, Any]] = []
        used: set = set()
        for sim, host, victim in candidates:
            if len(merged) >= limit:
                break
            host_id, victim_id = str(host.get("id")), str(victim.get("id"))
            if host_id in used or victim_id in used:
                continue
            used.add(host_id)
            used.add(victim_id)
            body = _body_of(victim)
            self.add_alias(host_id, body)
            self._append({
                "kind": "archive", "card_id": victim_id,
                "reason": "sleep_cluster_merge", "merged_into": host_id,
                "similarity": round(sim, 4),
                "ts": date.today().isoformat(),
            })
            self.cards.remove(victim)
            self.archived.append({**victim,
                                  "archive_reason": "sleep_cluster_merge",
                                  "merged_into": host_id,
                                  "archived_at": date.today().isoformat()})
            merged.append({"host": host_id, "victim": victim_id,
                           "similarity": round(sim, 4)})
        if merged:
            # Merge first, then re-check capacity; never the reverse, or the
            # round would evict a card that was about to be absorbed.
            self._enforce_capacity()
            self._write_cards()
        return merged

    def archive_card(self, card_id: str, reason: str = "manual") -> bool:
        card = next((c for c in self.cards if c.get("id") == card_id), None)
        if card is None:
            return False
        self._append({"kind": "archive", "card_id": card_id,
                      "reason": reason, "ts": date.today().isoformat()})
        self.cards.remove(card)
        self.archived.append({**card, "archive_reason": reason,
                              "archived_at": date.today().isoformat()})
        self._write_cards()
        return True

    def restore_card(self, card_id: str) -> bool:
        card = next((c for c in self.archived if c.get("id") == card_id), None)
        if card is None:
            return False
        self._append({"kind": "restore", "card_id": card_id,
                      "ts": date.today().isoformat()})
        self.archived.remove(card)
        card = dict(card)
        card.pop("archive_reason", None)
        card.pop("archived_at", None)
        self.cards.append(card)
        self._enforce_capacity()
        self._write_cards()
        return True

    def add_alias(self, card_id: str, phrase: str) -> bool:
        phrase = phrase.strip()
        if not valid_alias(phrase):
            return False
        phrases = self.aliases.setdefault(card_id, [])
        key = normalize(phrase)
        if any(normalize(p) == key for p in phrases):
            return False
        if len(phrases) >= MAX_ALIASES_PER_CARD:
            return False
        self._append({"kind": "alias", "card_id": card_id, "phrase": phrase,
                      "ts": date.today().isoformat()})
        phrases.append(phrase)
        self._write_aliases()
        return True

    def record_hit(self, card_id: str) -> None:
        counts = self.stats.setdefault(card_id, {})
        counts["hits"] = counts.get("hits", 0) + 1
        self._write_stats()

    def record_praise(self, card_id: str) -> None:
        counts = self.stats.setdefault(card_id, {})
        counts["praise"] = counts.get("praise", 0) + 1
        self._write_stats()

    def alias_items(self) -> Iterable[Tuple[str, str]]:
        for cid, phrases in self.aliases.items():
            for phrase in phrases:
                yield cid, phrase
