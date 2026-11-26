"""Keep evidence formatting within existing prose boundaries."""

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
