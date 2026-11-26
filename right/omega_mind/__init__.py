"""omega_mind — a rule-based affect/persona layer over PyBot.

Public surface:
    from omega_mind import MindEngine, get_persona
"""

from .config import (PersonaConfig, generic_persona, get_persona,
                     load_package_persona, persona_from_data)
from .engine import MindEngine, TurnResult

__all__ = [
    "MindEngine",
    "TurnResult",
    "PersonaConfig",
    "get_persona",
    "generic_persona",
    "load_package_persona",
    "persona_from_data",
]

__version__ = "0.1.0"
