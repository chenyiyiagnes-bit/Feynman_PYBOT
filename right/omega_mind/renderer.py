"""Turn a fact-bearing base reply into spoken text without touching the facts.

Rules:
- The base reply's wording and numbers are preserved (only trailing punctuation
  may be replaced).
- A prefix is used at most twice per rolling five turns.
- The same prefix is never used twice in a row.
- Statements can vary their endings. Questions and literal replies keep
  their original punctuation and may repeat.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Tuple

from .policy import StyleDirective

_COOLDOWN_TURNS = 5
_MAX_PER_PREFIX = 2
_TERSE_VERBOSITY = 0.30

# Expression hints carry a small set of decorative tags. They are only used
# when every neutral variant of a reply has already been spoken, so they never
# displace a fact or an ending.
_TAGS = {
    "zh": {
        "soft_smile": ("（笑）", "（点点头）"),
        "excited_curious": ("（眼睛一亮）", "（凑近了点）"),
        "blush_defensive": ("（别开脸）", "（嘴硬）"),
        "angry_pout": ("（撇嘴）", "（别过头）"),
        "raised_eyebrow": ("（挑眉）", "（停了一下）"),
    },
    "en": {
        "soft_smile": (" (smiles)", " (nods)"),
        "excited_curious": (" (eyes lighting up)", " (leans in)"),
        "blush_defensive": (" (looks away)", " (jaw tightens)"),
        "angry_pout": (" (pouts)", " (turns away)"),
        "raised_eyebrow": (" (raises an eyebrow)", " (pauses)"),
    },
}
_DEFAULT_TAGS = {
    "zh": ("（点头）", "（笑）", "（摊手）", "（想了想）"),
    "en": (" (nods)", " (laughs)", " (shrugs)", " (thinks)"),
}


class Renderer:
    def __init__(self) -> None:
        self._prefix_history: Deque[Tuple[int, str]] = deque()
        self._ending_cursor = 0
        self._last_reply: Optional[str] = None
        self._recent: Dict[str, Deque[str]] = {
            "zh": deque(maxlen=50), "en": deque(maxlen=50),
        }
        self.applied: List[Tuple[int, str]] = []  # (turn, prefix) for tests/inspection

    def render(self, base_reply: str, directive: StyleDirective, turn: int,
               language: str = "zh", posture: str = "answer", *,
               preserve: bool = False) -> str:
        base = base_reply.strip()
        if preserve or base.endswith(("？", "?")):
            self._last_reply = base
            self._recent.setdefault(language, deque(maxlen=50)).append(base)
            return base
        # A guarded turn drops the familiar ornaments: no catchphrase and no
        # playful ending. The facts are unchanged; only the shell tightens.
        careful = posture == "careful"
        # Low verbosity is the right brain saying "keep it short": the shell
        # sheds its ornaments the same way a guarded turn does.
        terse = careful or directive.verbosity <= _TERSE_VERBOSITY
        prefix = None if terse else self._pick_prefix(directive.prefixes, turn)
        endings = [] if terse else self._endings_for(
            directive.endings, directive.warmth, language)
        body = self._strip_trailing(base)
        recent = self._recent.setdefault(language, deque(maxlen=50))

        reply = self._with_ending((prefix or "") + body, endings, turn,
                                  language)
        if reply in recent:
            reply = self._dedupe(prefix, body, endings, language,
                                 playful=not careful,
                                 tags=self._tags_for(directive.expression,
                                                     language,
                                                     directive.humor))
        self._last_reply = reply
        recent.append(reply)
        return reply

    # -- internals ---------------------------------------------------------

    def _pick_prefix(self, candidates: List[str], turn: int) -> Optional[str]:
        if not candidates:
            return None
        self._expire(turn)
        counts = Counter(p for _, p in self._prefix_history)
        allowed = [p for p in candidates if counts.get(p, 0) < _MAX_PER_PREFIX]
        if not allowed:
            return None
        last = self._prefix_history[-1][1] if self._prefix_history else None
        allowed = [p for p in allowed if p != last] or allowed
        chosen = allowed[0]
        self._prefix_history.append((turn, chosen))
        self.applied.append((turn, chosen))
        return chosen

    def _expire(self, turn: int) -> None:
        while self._prefix_history and turn - self._prefix_history[0][0] >= _COOLDOWN_TURNS:
            self._prefix_history.popleft()

    @staticmethod
    def _strip_trailing(text: str) -> str:
        return re.sub(r"[。！!？?～~….\s]+$", "", text)

    def _with_ending(self, body: str, endings: List[str], turn: int,
                     language: str = "zh") -> str:
        if not endings:
            return body + ("." if language == "en" else "。")
        ending = endings[self._ending_cursor % len(endings)]
        self._ending_cursor += 1
        if body.endswith(ending):
            return body
        return body + ending

    @staticmethod
    def _endings_for(endings: List[str], warmth: float,
                     language: str = "zh") -> List[str]:
        """Let warmth choose the tone of the sentence ending, not just the prefix.

        A cold voice ends flat; a warm one may use the persona's soft endings.
        With no warmth signal the full persona pool is used.
        """
        if not endings:
            return []
        if warmth <= 0.30:
            return ["."] if language == "en" else ["。"]
        if warmth >= 0.65:
            soft = [e for e in endings
                    if "～" in e or "~" in e
                    or e.rstrip().endswith(("？", "?", "吗"))]
            return soft or endings
        return endings

    @staticmethod
    def _tags_for(expression: str, language: str = "zh",
                  humor: float = 0.5) -> Tuple[str, ...]:
        """Decorative tags for the dedupe fallback.

        A low-humor moment does not get a giggle tag even if the expression
        happens to be a smile.
        """
        pool = _TAGS.get(language, {}).get(expression)
        if pool is None or humor < 0.35:
            return _DEFAULT_TAGS.get(language, _DEFAULT_TAGS["zh"])
        return pool

    def _dedupe(self, prefix: Optional[str], body: str,
                endings: List[str], language: str = "zh",
                playful: bool = True,
                tags: Optional[Tuple[str, ...]] = None) -> str:
        """Find a phrasing not used in the recent window."""
        recent = self._recent.setdefault(language, deque(maxlen=50))
        unique_endings = list(dict.fromkeys(endings)) or (
            ["."] if language == "en" else ["。"])
        variants: List[str] = []
        for e in unique_endings:
            variants.append(f"{prefix or ''}{body}{e}")
        for e in unique_endings:
            variants.append(f"{body}{e}")
        if playful:
            for tag in (tags or _DEFAULT_TAGS.get(language, _DEFAULT_TAGS["zh"])):
                variants.append(f"{prefix or ''}{body}{tag}")
        for candidate in variants:
            if candidate not in recent:
                return candidate
        # A guarded turn stays guarded even when every neutral variant has
        # been used; the trailing mark is punctuation, not content.
        return body + ("……" if language == "zh" else " ...")

    def snapshot(self) -> dict:
        return {
            "prefix_history": list(self._prefix_history),
            "ending_cursor": self._ending_cursor,
            "last_reply": self._last_reply,
            "recent": {k: list(v) for k, v in self._recent.items()},
        }

    def restore(self, data: dict) -> None:
        self._prefix_history = deque(
            (int(t), str(p)) for t, p in data.get("prefix_history", [])
        )
        self._ending_cursor = int(data.get("ending_cursor", 0))
        self._last_reply = data.get("last_reply")
        recent = data.get("recent") or {}
        # Older saves kept a single list; treat it as Chinese.
        if not recent and data.get("recent_replies"):
            recent = {"zh": data["recent_replies"]}
        self._recent = {
            lang: deque((str(r) for r in recent.get(lang, [])), maxlen=50)
            for lang in ("zh", "en")
        }

    def reset(self) -> None:
        self._prefix_history.clear()
        self._ending_cursor = 0
        self._last_reply = None
        self._recent = {"zh": deque(maxlen=50), "en": deque(maxlen=50)}
        self.applied.clear()
