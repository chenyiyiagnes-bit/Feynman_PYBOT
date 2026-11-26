"""Keep evidence formatting within existing prose boundaries."""

import hashlib
import re

_MARKUP = re.compile(r"[`$\\<>\r\n]")
_PAIRS = {
    "“": "”", "‘": "’", "「": "」", "『": "』", "《": "》", "〈": "〉",
    "（": "）", "〔": "〕", "【": "】", "(": ")", "[": "]", "{": "}",
    '"': '"', "'": "'",
}
_CLOSING = frozenset(_PAIRS.values())
_BOUNDARIES = {
    "zh": re.compile(r"[。！？；](?=[ \t]*[^\s。！？；（(\[【〔])"),
    "en": re.compile(r"[!.?;](?=[ \t]+[A-Za-z])"),
}
_EN_NEXT_SENTENCE = re.compile(r"[ \t]+[A-Z]")
_ABBREVIATION = re.compile(
    r"(?:\b(?:[a-z]|mr|mrs|ms|dr|prof|fig|eq|vol|no|ch|etc|vs|al)|\d)\.$",
    re.I,
)


def candidates(text: str, language: str) -> tuple[int, ...]:
    """Offsets after punctuation where the text can safely be broken."""
    if _MARKUP.search(text):
        return ()
    punctuation = _BOUNDARIES.get(language)
    if punctuation is None:
        return ()
    offsets = []
    closing = []
    for index, char in enumerate(text):
        offset = index + 1
        if (char == "'" and 0 < index < len(text) - 1
                and text[index - 1].isalnum() and text[offset].isalnum()):
            continue  # an apostrophe within a word
        if closing and char == closing[-1]:
            closing.pop()
        elif char in _PAIRS:
            closing.append(_PAIRS[char])
        elif char in _CLOSING:
            return ()
        elif not closing and punctuation.match(text, index):
            if language == "en" and char == "." and (
                not _EN_NEXT_SENTENCE.match(text, offset)
                or _ABBREVIATION.search(text[max(0, offset - 12):offset])
            ):
                continue
            offsets.append(offset)
    return () if closing else tuple(offsets)


def apply_breaks(text: str, allowed: tuple[int, ...], plan: object) -> str | None:
    """Apply a responder's plan only when all offsets are allowed and ordered."""
    if type(plan) is not tuple or not plan:
        return None
    if any(type(offset) is not int or offset not in allowed for offset in plan):
        return None
    if any(a >= b for a, b in zip(plan, plan[1:])):
        return None
    bounds = (0, *plan, len(text))
    return "\n".join(text[start:end].strip(" \t") for start, end in zip(bounds, bounds[1:]))


_INTENT = {
    "zh": (
        ("condition", re.compile(r"^(?:那)?(?:如果|假如|在什么条件|什么情况下|什么时候|适用范围)")),
        ("mechanism", re.compile(r"为什么|怎么做到|如何导致|原理是什么")),
        ("example", re.compile(r"举个例子|举例|打个比方|换个例子")),
    ),
    "en": (
        ("condition", re.compile(r"^\s*(?:what if|under what conditions|when (?:does|will|would))\b", re.I)),
        ("mechanism", re.compile(r"\b(?:why|how does|what causes|what is the mechanism)\b", re.I)),
        ("example", re.compile(r"\b(?:an example|an analogy|another example|illustrate)\b", re.I)),
    ),
}
ROLES = ("conclusion", "mechanism", "example", "condition")


def intent_for(text: str, language: str) -> str:
    return next((kind for kind, pattern in _INTENT[language] if pattern.search(text)), "conclusion")


_CARD_FIELDS = {
    "teachable_unit": {"conclusion": ("force_invariant",), "mechanism": ("invariant",),
                       "condition": ("boundary",)},
    "method_card": {"conclusion": ("pattern",), "example": ("novel_use",),
                    "condition": ("guardrail",)},
    "physics_edge": {"conclusion": ("phenomenon",), "mechanism": ("invariant", "operator")},
}


def structured_card(card) -> bool:
    return "passages" in (card.row or {}) or card.kind in _CARD_FIELDS


def card_passages(card) -> list[dict]:
    """Accept only passages that can be traced to the card's rich answer."""
    from .dialogue import card_revision

    row = card.row or {}
    if "passages" in row:
        sections = row["passages"]
    else:
        sections = {role: [row[key] for key in fields if row.get(key)]
                    for role, fields in _CARD_FIELDS.get(card.kind, {}).items()}
    if not isinstance(sections, dict):
        return []
    rich = card.answer_for("rich")
    revision = card_revision(card)
    support = {}
    block = row.get("support")
    for item in block if isinstance(block, list) else ():
        if not isinstance(item, dict):
            continue
        quote, source = item.get("quote"), item.get("source")
        if not all(isinstance(key, str) and key for key in (quote, source)):
            continue
        if not isinstance(item.get("reviewed_by"), str) or not item["reviewed_by"].strip():
            continue
        if hashlib.sha256(quote.encode()).hexdigest() == item.get("quote_sha256"):
            support[quote] = {"provenance": "reviewed_quote", "source": item["source"],
                              "quote_sha256": item["quote_sha256"]}
    out = []
    for role in ROLES:
        value = sections.get(role, [])
        texts = [value] if isinstance(value, str) else value
        if not isinstance(texts, list):
            return []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                return []
            start = rich.find(text)
            if start < 0:
                return []
            out.append({"role": role, "text": text, "span": [start, start + len(text)],
                        "card_id": card.id, "revision": revision,
                        **support.get(text, {"provenance": "card_summary"})})
    return out


def compose(card, intent: str, detail: str, prefer_examples: bool = False):
    """Return verbatim passages or a reason why the scoped card cannot answer."""
    data = card_passages(card)
    if not data:
        return [], "invalid_passages"
    roles = {item["role"] for item in data}
    if intent not in roles and "passages" not in (card.row or {}) and intent == "mechanism":
        intent = "conclusion"  # existing typed card's own mechanism in the core statement
    if "conclusion" not in roles or intent not in roles:
        return [], "insufficient_passages"
    required = {"conclusion", "condition", intent}
    order = [intent, *(role for role in ROLES if role != intent)]
    if prefer_examples and intent == "conclusion":
        order = ["conclusion", "example", "mechanism", "condition"]
    return [item for role in order for item in data
            if item["role"] == role and (detail != "brief" or role in required)], ""


def conflict_among(cards) -> bool:
    claims = {}
    for card in cards:
        row = card.row or {}
        key, value = row.get("claim_key"), row.get("claim_value")
        if not isinstance(key, str) or not isinstance(value, str) or not key or not value:
            continue
        if key in claims and claims[key] != value:
            return True
        claims[key] = value
    return False


_INVENTED_ENTITY = {
    "zh": re.compile(r"(?:虚构的|我编的|并不存在的)"),
    "en": re.compile(r"\b(?:the|an?)\s+(?:fictional|imaginary|made[- ]up|invented)\b", re.I),
}


def hypothetical_measurement(text: str, language: str) -> bool:
    """An invented entity cannot have an observed value or reported event."""
    if not _INVENTED_ENTITY[language].search(text):
        return False
    if language == "zh":
        return bool(re.search(r"曾|已经|证实|观测|昨天|公布|测得", text))
    return bool(re.search(r"\b(?:did|was|were|measured|observed|announced|yesterday)\b", text, re.I))
