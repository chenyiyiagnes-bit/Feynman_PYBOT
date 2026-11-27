"""Where a card's claim comes from: one canonical shape.

A card records provenance in one of three equivalent ways, and packages
historically used all three:

    "support": [{"source": "FLP I-1-1", "quote": "...", ...}]  # quote card
    "support": [{"source": "FLP I-1-1"}]                       # cite
    "source": "《费曼物理学讲义》第一卷"                          # top-level cite
    "source": {"book": "道德经", "chapter": "第一章"}            # structured cite

The ``support`` list is the canonical form. ``entries()`` is the one place
that knows the others: validators, tools and reports read it instead of
each growing its own acceptance rule. A package may keep the legacy
top-level ``source``; new cards should write ``support``.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _text(source: Any) -> str:
    """Readable source line from a string or a book/chapter mapping."""
    if isinstance(source, str):
        return source.strip()
    if isinstance(source, dict):
        parts = [str(v).strip() for v in source.values()
                 if isinstance(v, str) and v.strip()]
        return " · ".join(parts)
    return ""


def declared(row: Dict[str, Any]) -> bool:
    """True when the row writes a ``support`` block, even a malformed one."""
    return row.get("support") is not None


def entries(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Canonical provenance entries for one card row (possibly empty).

    A malformed ``support`` yields an empty list; ``declared()`` tells a
    caller whether that emptiness means "broken" or "absent".
    """
    support = row.get("support")
    if support is not None:
        return list(support) if isinstance(support, list) else []
    text = _text(row.get("source"))
    return [{"source": text}] if text else []


def cited(row: Dict[str, Any]) -> bool:
    """True when the card names any source at all."""
    return bool(entries(row))
