"""Teach-back challenge mode.

Feynman asks a question from a teachable unit, the user answers in their own
words, and the unit's own invariant is used as the answer key. Grading is
word-coverage over the key terms plus a numbers check — deterministic, no
model calls. Getting it wrong costs nothing socially: curiosity goes up
either way, which is the persona rule "被纠错是礼物" turned around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .feynman_kb import Doc, FeynmanKB, _stem_en, _tokens


@dataclass
class Grade:
    verdict: str            # "strong" | "partial" | "weak"
    score: float
    matched: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    feedback: str = ""
    revealed: Optional[str] = None   # model answer, only after giving up


_STOP_CHARS = set("的了是在我你他她它吗呢啊吧就不么这那个有为和与及很太都也还即被把给对从向于以及把被一个一些")

_SUBJECT_STOP = {"时间", "常数", "同一个", "一个", "无关", "多大", "比例"}

_EN_STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "with",
    "by", "for", "from", "that", "this", "it", "as", "at", "on", "not",
    "no", "same", "every", "any", "its", "their", "there", "which", "when",
    "how", "what", "why", "does", "do", "can", "be", "has", "have", "was",
    "were", "will", "would", "should", "you", "your", "we", "our", "they",
    "them", "if", "than", "then", "so", "but", "one", "two", "all", "only",
    "more", "less", "per", "single", "fixed", "regardless", "matter",
    "over", "under", "time", "constant", "large", "small", "say", "says",
    "said", "must", "where", "went", "now", "take", "takes", "taken",
    "three", "slow", "fast", "fastest", "big", "get", "gets", "start",
    "starting", "same", "different", "thing", "things", "way",
}


def _stem(word: str) -> str:
    """Cheap English stemming: enough to match decay/decays/decaying."""
    return _stem_en(word)


def _en_terms(text: str) -> List[str]:
    seen, terms = set(), []
    for tok in _tokens(text):
        if len(tok) < 3 or tok.isdigit() or not any(c.isalpha() for c in tok):
            continue
        if tok in _EN_STOP:
            continue
        stem = _stem(tok)
        if stem not in seen:
            seen.add(stem)
            terms.append(stem)
    return terms[:12]


def _term_in(term: str, tokens: set) -> bool:
    """Match a key term, tolerating one side being a stem of the other.

    The two stemmers are intentionally light, so "devi" (from deviation) and
    "deviat" (from deviates) must still count as the same concept. Only
    tokens of at least four characters may use prefix matching.
    """
    if term in tokens:
        return True
    if len(term) < 4:
        return False
    return any(len(tok) >= 4 and (term.startswith(tok) or tok.startswith(term))
               for tok in tokens)


def _key_terms(unit: dict, language: str = "zh") -> List[str]:
    """Concept terms of the invariant, without formula fragments."""
    if language == "en":
        return _en_terms(str(unit.get("invariant") or ""))
    text = str(unit.get("invariant") or "")
    seen, terms = set(), []
    for tok in _tokens(text):
        if len(tok) < 2 or tok.isdigit():
            continue
        if any(ch.isascii() for ch in tok):
            continue  # ln2, dx, dt: formula pieces, not answer requirements
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)
    return [t for t in terms if t not in _SUBJECT_STOP][:10]


def _content_chars(text: str) -> set:
    return {ch for ch in text if "一" <= ch <= "鿿" and ch not in _STOP_CHARS}


def _numbers(unit: dict) -> List[str]:
    import re

    text = str(unit.get("operator") or "") + str(unit.get("invariant") or "")
    # A digit glued to letters (ln2, t2) is part of a symbol, not a number the
    # learner must reproduce; counting it would punish correct prose answers.
    return re.findall(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?![A-Za-z0-9])", text)


class ChallengeSession:
    def __init__(self, kb: FeynmanKB, engine, language: Optional[str] = None):
        self.kb = kb
        self.engine = engine
        self.language = language or getattr(engine, "language", "zh")
        self.units: List[Doc] = kb.teachable_units(language=self.language)
        self.index = -1
        self.unit: Optional[Doc] = None
        self.tries = 0
        self.done = 0
        # unit id -> {"asked": int, "strong": int}. A unit answered strongly is
        # mastered; next_unit walks past those until everything is seen once.
        self.mastery: dict[str, dict[str, int]] = {}

    # -- session flow ------------------------------------------------------

    def available(self) -> bool:
        return bool(self.units)

    def snapshot(self) -> dict:
        return {
            "index": self.index,
            "tries": self.tries,
            "done": self.done,
            "active": self.unit.id if self.unit is not None else None,
            "language": self.language,
            "mastery": {uid: dict(counts) for uid, counts in self.mastery.items()},
        }

    def restore(self, data: dict) -> None:
        """Resume a saved session. A missing unit id simply means the next
        ``next_unit`` call starts over instead of pointing at a stale card."""
        self.index = int(data.get("index", -1))
        self.tries = int(data.get("tries", 0))
        self.done = int(data.get("done", 0))
        raw = data.get("mastery") or {}
        self.mastery = {
            str(uid): {"asked": int(v.get("asked", 0)),
                       "strong": int(v.get("strong", 0))}
            for uid, v in raw.items() if isinstance(v, dict)
        }
        active = data.get("active")
        self.unit = next((d for d in self.units if d.id == active), None) \
            if active else None

    def status(self) -> dict:
        question = None
        if self.unit is not None:
            question = self.unit.unit.get("entry_question") or self.unit.title
        return {
            "available": self.available(),
            "active": self.unit.id if self.unit is not None else None,
            "question": question,
            "tries": self.tries,
            "done": self.done,
            "mastered": sum(1 for d in self.units if self._mastered(d.id)),
            "total": len(self.units),
        }

    def _mastered(self, unit_id: str) -> bool:
        return self.mastery.get(unit_id, {}).get("strong", 0) > 0

    def next_unit(self) -> Optional[str]:
        if not self.units:
            return None
        # Serve an unmastered card first; once all are mastered, keep cycling
        # so practice can continue.
        for offset in range(1, len(self.units) + 1):
            idx = (self.index + offset) % len(self.units)
            if not self._mastered(self.units[idx].id):
                self.index = idx
                break
        else:
            self.index = (self.index + 1) % len(self.units)
        self.unit = self.units[self.index]
        self.tries = 0
        counts = self.mastery.setdefault(self.unit.id, {"asked": 0, "strong": 0})
        counts["asked"] += 1
        question = self.unit.unit.get("entry_question") or self.unit.title
        lead = "Here's one for you. " if self.language == "en" else "来，我问你一个。"
        return self.engine.say(f"{lead}{question}")

    def submit(self, answer: str) -> Grade:
        if self.unit is None:
            self.next_unit()
        assert self.unit is not None
        unit = self.unit.unit
        self.tries += 1

        terms = _key_terms(unit, self.language)
        if self.language == "en":
            answer_tokens = {_stem(t) for t in _tokens(answer)}
            matched = [t for t in terms if _term_in(t, answer_tokens)]
            missing = [t for t in terms if not _term_in(t, answer_tokens)]
            term_cover = len(matched) / len(terms) if terms else 0.0
            score = term_cover
        else:
            answer_tokens = set(_tokens(answer))
            matched = [t for t in terms if t in answer_tokens]
            missing = [t for t in terms if t not in answer_tokens]

            # Paraphrase in Chinese rarely reuses the same bigrams, but it does
            # reuse the content characters ("凉得更慢" vs "衰减更慢"). Character
            # coverage carries most of the score; exact terms are a bonus.
            key_chars = _content_chars(str(unit.get("invariant") or ""))
            hit_chars = key_chars & _content_chars(answer)
            char_cover = len(hit_chars) / len(key_chars) if key_chars else 0.0
            term_cover = len(matched) / len(terms) if terms else 0.0
            score = 0.6 * char_cover + 0.4 * term_cover

        # Only enforce the number check when the key contains a real
        # quantity (a decimal or a multi-digit number). "first-order" and
        # "ln2" are symbols, and a correct prose answer need not repeat them.
        nums = [n for n in _numbers(unit)
                if "." in n or (n.isdigit() and int(n) >= 10)]
        if nums:
            hit = [n for n in nums if n in answer]
            score *= 0.7 + 0.3 * (len(hit) / len(nums))

        if self.language == "en":
            strong, partial = 0.55, 0.3
        else:
            strong, partial = 0.5, 0.28
        if score >= strong:
            verdict = "strong"
        elif score >= partial:
            verdict = "partial"
        else:
            verdict = "weak"
        self._affect(verdict)

        feedback = self._feedback(verdict, unit, matched, missing)
        grade = Grade(verdict=verdict, score=round(score, 2),
                      matched=matched, missing=missing[:4], feedback=feedback)

        if verdict == "strong":
            self.done += 1
            counts = self.mastery.setdefault(self.unit.id,
                                             {"asked": 0, "strong": 0})
            counts["strong"] += 1
            self.unit = None
        elif self.tries >= 2:
            grade.revealed = unit.get("force_invariant", "")
            lead = ("All right, here's how I think about it — "
                    if self.language == "en" else "算了，我直接给你看我是怎么想的 —— ")
            grade.feedback += "\n\n" + self.engine.say(lead + grade.revealed)
            self.unit = None
        return grade

    # -- internals ---------------------------------------------------------

    def _affect(self, verdict: str) -> None:
        boost = {"strong": 0.18, "partial": 0.10, "weak": 0.05}[verdict]
        self.engine.state.apply({"dopamine": boost, "valence": boost / 2})

    def _feedback(self, verdict: str, unit: dict, matched: List[str], missing: List[str]) -> str:
        if self.language == "en":
            if verdict == "strong":
                invariant = unit.get("invariant") or ""
                core = ", ".join(matched[:2])
                return self.engine.say(f"That's it. You caught {core} — {invariant}")
            if verdict == "partial":
                gap = ", ".join(missing[:1])
                hint = unit.get("check_yourself") or unit.get("phenomenon") or ""
                lead = f"Close — you're missing {gap}. " if gap else "Close. "
                return self.engine.say(f"{lead}Try checking it this way: {hint}")
            phenomenon = unit.get("phenomenon") or ""
            return self.engine.say(
                f"Don't define it yet. {phenomenon} What do you see in that?"
            )
        if verdict == "strong":
            invariant = unit.get("invariant") or ""
            # Overlapping bigrams ("偏离/离平/平衡") read like stutter; keep the
            # longest terms only.
            picked = [t for t in matched if not any(t != o and t in o for o in matched)]
            core = "、".join(picked[:2])
            return self.engine.say(f"对了。你抓住的是{core}——{invariant}")
        if verdict == "partial":
            # Only terms long enough to read as words; bigram halves such as
            # "随时/间按" are noise.
            gap = "、".join([t for t in missing if len(t) >= 3][:1])
            hint = unit.get("check_yourself") or unit.get("phenomenon") or ""
            lead = f"离得不远，还差{gap}。" if gap else "离得不远。"
            return self.engine.say(f"{lead}换个角度验一下：{hint}")
        phenomenon = unit.get("phenomenon") or ""
        return self.engine.say(
            f"先别急着定义。{phenomenon} 你从这里头看到了什么？说给我听听。"
        )
