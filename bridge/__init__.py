"""The corpus callosum: right-brain affect → left-brain typesetter."""

from .mapping import Affects, BridgeParams, map_affect, OPERATORS
from .persona import Persona, PersonaError, discover, load
from .responder import BridgeResult, LeftResponder

__all__ = [
    "Affects",
    "BridgeParams",
    "BridgeResult",
    "LeftResponder",
    "OPERATORS",
    "Persona",
    "PersonaError",
    "discover",
    "load",
    "map_affect",
]
