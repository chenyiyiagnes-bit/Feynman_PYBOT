"""Right brain → left brain responder.

Default OFF. Set ``OMEGA_LEFT_BRIDGE=1`` to enable.

Knowledge packages use ``plan_evidence()`` to lay out selected passages.
The standalone typesetter uses ``reply()`` with its own vault.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .mapping import BridgeParams, map_affect
from .persona import Persona
from . import slot_gate

ENV_FLAG = "OMEGA_LEFT_BRIDGE"
DEFAULT_LEFT_ROOT = Path(__file__).resolve().parents[1] / "left"


@dataclass
class BridgeResult:
    text: str
    raw: Dict[str, Any]
    params: Dict[str, Any]


class LeftResponder:
    def __init__(self, left_root: str | Path | None = None,
                 vault_path: str | Path | None = None,
                 enabled: bool | None = None,
                 lexicon_dir: str | Path | None = None) -> None:
        self.left_root = Path(left_root or os.getenv("QBOT_LEFT_ROOT") or DEFAULT_LEFT_ROOT).expanduser().resolve()
        env_vault = os.getenv("QBOT_FEYNMAN_VAULT")
        self.vault_path = Path(
            vault_path or env_vault or (self.left_root / "data" / "feynman_style_vault.json")
        ).expanduser()
        self.enabled = (os.getenv(ENV_FLAG) == "1") if enabled is None else bool(enabled)
        self.lexicon_dir = Path(lexicon_dir).expanduser() if lexicon_dir else None
        self._weaver = None
        self._request_cls = None
        self._vault = None
        self.last_error: Optional[str] = None

    # -- environment ---------------------------------------------------------

    def _load_modules(self):
        if self._weaver is not None:
            return self._weaver, self._request_cls
        root = str(self.left_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        weaver = importlib.import_module("plugins.paragraph_weaver")
        typesetter = importlib.import_module("plugins.feynman_typesetter")
        self._weaver = weaver
        self._request_cls = typesetter
        return weaver, typesetter

    def available(self) -> bool:
        if not self.enabled:
            self.last_error = self.last_error or "disabled (set OMEGA_LEFT_BRIDGE=1)"
            return False
        if not self.vault_path.is_file():
            self.last_error = f"vault missing: {self.vault_path}"
            return False
        try:
            self._load_modules()
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"import failed: {exc}"
            return False
        return True

    def _vault_obj(self):
        if self._vault is not None:
            return self._vault
        _, typesetter = self._load_modules()
        self._vault = typesetter.Vault(self.vault_path)
        return self._vault

    # -- public --------------------------------------------------------------

    def evidence_available(self) -> bool:
        if not self.enabled:
            return False
        try:
            layout = importlib.import_module("left.plugins.evidence_layout")
            if not callable(layout.plan_breaks):
                self.last_error = "evidence layout has no planner"
                return False
        except Exception as exc:  # noqa: BLE001 - readiness must degrade gracefully
            self.last_error = f"evidence layout unavailable: {exc}"
            return False
        self.last_error = None
        return True

    def plan_evidence(self, evidence: str, *, source: str, language: str,
                      detail: str, dialogue_act: str,
                      candidates: tuple[int, ...]) -> tuple[int, ...]:
        """Select from the caller's verified pauses in a sourced passage."""
        if not self.enabled or not source or not evidence or not candidates:
            return ()
        try:
            layout = importlib.import_module("left.plugins.evidence_layout")
            plan = layout.plan_breaks(
                evidence, language=language, detail=detail,
                dialogue_act=dialogue_act, candidates=candidates)
        except Exception as exc:  # noqa: BLE001 - display is optional
            self.last_error = f"evidence layout failed: {exc}"
            return ()
        self.last_error = None
        return plan

    @staticmethod
    def _jargon_aversion(persona) -> bool:
        """Accept both bridge.Persona (``switches`` dict) and the right
        brain's PersonaConfig (flat ``jargon_aversion`` bool)."""
        switches = getattr(persona, "switches", None)
        if isinstance(switches, dict):
            return bool(switches.get("jargon_aversion"))
        return bool(getattr(persona, "jargon_aversion", False))

    def params_for(self, state: Dict[str, float], persona,
                   cube_depth: float = 0.0,
                   cube_verbosity: float = 0.0) -> BridgeParams:
        return map_affect(
            state,
            persona.baselines,
            cube_depth=cube_depth,
            cube_verbosity=cube_verbosity,
            jargon_aversion=self._jargon_aversion(persona),
        )

    @property
    def gate_active(self) -> bool:
        """True when there is coded data to check a slot fill against."""
        return slot_gate.load(self.lexicon_dir) is not None

    def reply(self, text: str, state: Dict[str, float], persona: Persona,
              cube_depth: float = 0.0, cube_verbosity: float = 0.0,
              nonce: int = 0, session_id: Optional[str] = None,
              card_id: Optional[str] = None) -> Optional[BridgeResult]:
        if not self.enabled:
            return None
        try:
            if not self.vault_path.is_file():
                self.last_error = f"vault missing: {self.vault_path}"
                return None
            weaver, typesetter = self._load_modules()
            params = self.params_for(state, persona, cube_depth, cube_verbosity)
            operators = typesetter.OperatorWeights(**params.operators)
            request = weaver.ParagraphRequest(
                user_query=text[:200],
                style_deg=params.style_deg,
                nonce=max(0, min(int(nonce), 1_000_000)),
                session_id=session_id,
                operators=operators,
                **params.flags,
            )
            result = weaver.weave_paragraph(self._vault_obj(), request)
            paragraph = (result or {}).get("paragraph") or ""
            if not paragraph.strip():
                self.last_error = "empty paragraph"
                return None
            # Slot gate: a user word may only sit in a slot the chosen card
            # can account for. Unverifiable (no coded data, no card) passes.
            stray = slot_gate.unverified(result, card_id, self.lexicon_dir)
            if stray:
                self.last_error = "slot gate: " + "、".join(stray)
                return None
            return BridgeResult(text=paragraph, raw=result, params=params.to_dict())
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
