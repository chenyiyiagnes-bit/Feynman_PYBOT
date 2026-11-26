"""Choose pauses in an already selected passage."""

_COMPACT_ACTS = ("why", "boundary", "rephrase")
_SEMICOLONS = "；;"


def plan_breaks(evidence: str, *, language: str, detail: str,
                dialogue_act: str, candidates: tuple[int, ...]) -> tuple[int, ...]:
    """Choose original offsets, leaving the wording and punctuation intact."""
    if language not in ("zh", "en") or not candidates or detail == "brief":
        return ()
    minimum = 12 if language == "zh" else 28
    if dialogue_act in _COMPACT_ACTS:
        # A short follow-up can give a dense, two-part explanation a pause.
        choices = [
            offset for offset in candidates
            if (evidence[offset - 1] in _SEMICOLONS or dialogue_act == "rephrase")
            and offset >= minimum and len(evidence) - offset >= minimum
        ]
        if not choices:
            return ()
        return (min(choices, key=lambda offset: abs(2 * offset - len(evidence))),)

    if len(evidence) < (120 if language == "zh" else 240):
        return ()
    # Give long explanations room while keeping complete sentences together.
    pace = 45 if language == "zh" else 100
    if detail == "rich":
        pace *= 2
    cuts = []
    previous = 0
    for offset in candidates:
        if (offset - previous >= pace
                and len(evidence) - offset >= minimum
                and evidence[offset - 1] not in _SEMICOLONS):
            cuts.append(offset)
            previous = offset
    return tuple(cuts)
