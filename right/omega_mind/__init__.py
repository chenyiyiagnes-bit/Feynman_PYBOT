"""omega_mind — a rule-based affect/persona layer over PyBot.

Public surface:
    from omega_mind import MindEngine, get_persona
"""

from .config import PersonaConfig, feynman_persona, generic_persona, get_persona
from .engine import MindEngine, TurnResult

__all__ = [
    "MindEngine",
    "TurnResult",
    "PersonaConfig",
    "get_persona",
    "generic_persona",
    "feynman_persona",
]

__version__ = "0.1.0"
