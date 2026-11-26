"""The mind: one turn = decay -> appraise -> apply -> base reply -> render."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import knowledge, left_brain, reflect, resonance, skepticism
from .appraisal import Activation, AppraisalEngine
from .challenge import ChallengeSession
from .config import PersonaConfig, get_persona
from .cubies import CUBIES_BY_NAME
from .cube import CubeState
from .cube_feedback import CubeFeedback, dream_restore_ratio
from .dream import Dream, Dreamer
from .feynman_kb import FeynmanKB
from .learner import (
    CORRECTION_MARKERS,
    PROMOTION_CORTISOL_MAX,
    PROMOTION_TTL,
    explicit_confirmation,
    looks_like_metaphor,
    metaphor_candidate,
    shared_pairs,
)
from .memory import Episode, MemoryTiers, salience_of
from .policy import Policy, StyleDirective
from .pybot_bridge import PyBotBridge
from .relations import RelationshipGraph
from .renderer import Renderer
from .state import AffectState


@dataclass
class TurnResult:
    reply: str
    base_reply: str
    source: str
    activations: List[Activation]
    directive: StyleDirective
    state: Dict[str, float]
    turn: int
    resonance: Dict[str, Any] = field(default_factory=dict)
    left_brain: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "base_reply": self.base_reply,
            "source": self.source,
            "activations": [asdict(a) for a in self.activations],
            "directive": asdict(self.directive),
            "state": self.state,
            "turn": self.turn,
            "resonance": self.resonance,
            "left_brain": self.left_brain,
            "retrieval": self.retrieval,
        }


_PROACTIVE = {
    "zh": [
        "喂，在忙吗？我刚想起来一个怪问题——为什么冰块浮在水上，玻璃球却沉？",
        "你好像很久没说话了。我这边挺安静的，顺手把昨天的事又想了一遍。",
        "突然冒个泡：如果现在让你用一句话解释“时间”，你会怎么说？",
        "我之前那个问题还没问完呢，你跑哪去了？",
    ],
    "en": [
        "Hey — still there? A weird one just hit me: why does ice float but a glass marble sink?",
        "It's quiet on your side. I've been turning yesterday's conversation over in my head.",
        "Random thought: if you had one sentence to explain time, what would it be?",
        "You never answered my last question. Where did you run off to?",
    ],
}

LANGUAGES = ("zh", "en")
_FALLBACK = {
    "zh": "嗯……这个问题我得再翻翻记忆库。",
    "en": "I'm not confident about that one.",
}

# A guarded turn says so when two cards are almost tied. This is a style
# sentence, not a fact: it only comments on the retrieval margin.
_HEDGE = {
    "zh": "这个我把握不算大，先说最贴近的一个。",
    "en": "I'm not fully certain here; this is the closest I have.",
}
_HEDGE_MARGIN = 0.06

# Questions the semantic tier can answer directly. These are the only paths
# by which stored facts reach a reply; the KB still owns everything else.
_FACT_QUESTIONS = {
    "zh": [
        (re.compile(r"我(?:最|很|比较)?喜欢什么"), "likes", "你提过你喜欢{0}。"),
        (re.compile(r"我(?:最|很)?(?:讨厌|不喜欢)什么"), "dislikes",
         "你说过你受不了{0}。"),
        (re.compile(r"我住在(?:哪|哪里|哪儿)"), "lives_in", "你住在{0}。"),
        (re.compile(r"我(?:做什么|干什么)工作|我是做什么的"), "work",
         "你提过你的工作跟{0}有关。"),
        (re.compile(r"我(?:叫|的名字是|名字叫)(?:什么|啥|什么名字)|我叫啥"),
         "name", "你叫{0}。"),
    ],
    "en": [
        (re.compile(r"what do i (?:like|love)", re.I), "likes",
         "You told me you like {0}."),
        (re.compile(r"what do i (?:dislike|hate)", re.I), "dislikes",
         "You told me you can't stand {0}."),
        (re.compile(r"where do i live", re.I), "lives_in", "You live in {0}."),
        (re.compile(r"where do i work|what do i do for work", re.I), "work",
         "You told me your work is about {0}."),
        (re.compile(r"what(?:'s| is) my name", re.I), "name", "You're {0}."),
    ],
}

# Kinship/role questions are answered from the relationship graph's tags.
_ROLE_QUESTIONS = {
    "zh": re.compile(
        r"我的?(妈|妈妈|母亲|爸|爸爸|父亲|哥|哥哥|姐|姐姐|弟|弟弟|妹|妹妹|"
        r"老板|同事|朋友|妻子|老公|丈夫|男朋友|女朋友)叫(?:什么|啥|什么名字)"),
    "en": re.compile(
        r"what(?:'s| is) my (mom|mother|dad|father|brother|sister|boss|wife|"
        r"husband|boyfriend|girlfriend)(?:'s name)?", re.I),
}


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _match_language(text: str, language: str) -> bool:
    """A reply belongs to a mode when its script matches the mode."""
    if not text:
        return False
    if language == "zh":
        return has_cjk(text)
    return not has_cjk(text)


def _with_association(reply: Optional[str], line: Optional[str],
                      detail: str) -> Optional[str]:
    """Attach one confirmed association remark, never more.

    Brief answers stay brief: the remark is an aside, not a fact, and the
    right brain has already asked for the short form.
    """
    if not reply or not line or detail == "brief":
        return reply
    return f"{reply}\n{line}"


class MindEngine:
    # Bounded in memory: a long-running server must not grow one log entry per
    # message forever. Snapshots keep the same tail.
    EVENT_LOG_LIMIT = 200

    def __init__(
        self,
        persona: Optional[PersonaConfig] = None,
        store_path: Optional[str] = None,
        clock: Callable[[], float] = time.time,
        bridge: Optional[PyBotBridge] = None,
        kb: Optional[FeynmanKB] = None,
        auto_save: bool = True,
        language: str = "zh",
    ):
        self.persona = persona or get_persona("generic")
        self.clock = clock
        self.store_path = Path(store_path) if store_path else None
        self.auto_save = auto_save and self.store_path is not None
        self.bridge = bridge or PyBotBridge()
        self.kb = kb or FeynmanKB()
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        self.language = language

        now = clock()
        self.state = AffectState.from_persona(self.persona, now)
        self.appraisal = AppraisalEngine(self.persona)
        self.policy = Policy(self.persona)
        self.renderer = Renderer()

        self.memory: Dict[str, Any] = {}
        self.tiers = MemoryTiers()
        self.relations = RelationshipGraph()
        self.cube = CubeState()
        self.cube_feedback = CubeFeedback()
        self.dreamer = Dreamer()
        self.current_user = "default"
        self.event_log: List[Dict[str, Any]] = []
        self.turn_index = 0
        self.last_interaction: Optional[float] = None
        self.last_proactive: Optional[float] = None
        self._proactive_cursor = 0
        self._resonance_energy = 0.0
        self._resonance_ts: Optional[float] = None
        self.last_left_brain: Dict[str, Any] = {}
        self.last_retrieval: Dict[str, Any] = {}
        self.challenge_session: Optional[ChallengeSession] = None
        # What the previous turn asked and where its answer came from. Praise
        # on the next turn is the only signal that turns that phrasing into a
        # learned alias.
        self._last_query: Optional[str] = None
        self._last_source: Optional[str] = None
        # Loose comparisons ("相当于", "可以看作") wait one or two turns for a
        # confirmation signal before becoming cards; explicit ones don't.
        self._pending_metaphors: List[Dict[str, Any]] = []
        # Set when a save file could not be read and the engine started fresh.
        self.load_warning: Optional[str] = None

    # -- main loop ---------------------------------------------------------

    def turn(
        self,
        text: str,
        now: Optional[float] = None,
        base_reply: Optional[str] = None,
        user: str = "default",
    ) -> TurnResult:
        now = self.clock() if now is None else now
        if user != self.current_user:
            # Affinity is per-person: it is written back to the graph at the
            # end of every turn, so switching users only has to read.
            self.state.set("affinity", self.relations.affinity(user, self.persona.baseline("affinity")))
            self.current_user = user
        # The cube's return path: how long each displaced cubie has held the
        # front face this turn, and which baselines the affect state should
        # relax toward while the arrangement stays scrambled.
        self.cube_feedback.observe(self.cube, self.turn_index)
        effective_baselines = self.cube_feedback.baselines(self.persona, self.cube)
        self.state.advance(now, self.persona, baselines=effective_baselines)

        gains = self.cube_feedback.gains(self.cube)
        activations = self.appraisal.evaluate(text, self.turn_index, gains=gains)

        # Praise is the only feedback signal this mind has. If the previous
        # turn was answered from a card and this turn is praise, remember the
        # phrasing that led there; later paraphrases of it will score higher.
        if (self._last_source and self._last_source.startswith("feynman_kb:")
                and self._last_query
                and any(a.name == "edge_humor" and a.strength >= 0.5
                        for a in activations)):
            card_id = self._last_source.split(":", 1)[1]
            self.kb.learn_alias(card_id, self._last_query)
            self.kb.note_praise(card_id)

        # A loose comparison from an earlier turn earns a card once the user
        # confirms it — by praising, or by using it again. Promotion happens
        # before retrieval so the confirming turn can already use it.
        self._promote_metaphors(text, activations)

        # A grand claim gets the evidence reflex before any card is retrieved.
        # The jolt is part of the affect dynamics, not a separate UI message.
        skeptic = skepticism.check(text, self.language) if base_reply is None else None
        if skeptic is not None:
            self.state.apply({
                "dopamine": skepticism.DOPAMINE_JOLT,
                "cortisol": skepticism.CORTISOL_JOLT,
            })

        deltas: Dict[str, float] = {}
        for act in activations:
            for dim, value in act.effects.items():
                deltas[dim] = deltas.get(dim, 0.0) + value * min(1.0, act.strength)
        # Effort: a turn costs a little energy, and a busy turn costs more.
        # Without this, energy sat at its persona baseline forever and the
        # fatigue branch of the left brain could never fire.
        effort = min(0.06, 0.008 + 0.002 * len(activations))
        deltas["energy"] = deltas.get("energy", 0.0) - effort
        self.state.apply(deltas)

        # The right brain has now reacted; let it choose how the left brain
        # should look for and compose facts. It never supplies facts itself.
        left_policy = left_brain.plan_for(
            self.state, self.persona, activations,
            cube_bias=self.cube_feedback.bias(self.cube))
        self.last_left_brain = left_policy.to_dict()

        self.relations.touch(user, now)
        self.relations.mentions(text, now, user)
        self._turn_cube(activations, gains)

        # Second-order breakthrough: a strong appraisal (or a correction)
        # re-enters the cube as a short phase-locked sequence.
        carried = resonance.carried_energy(
            self._resonance_energy, self._resonance_ts, now)
        res = resonance.assess(
            activations,
            bonus=skepticism.RESONANCE_JOLT if skeptic is not None else 0.0,
            carried=carried,
        )
        if res.breakthrough:
            face = resonance.dominant_face(activations) or "U"
            for move in resonance.rumination_moves(face, res.depth):
                try:
                    self.cube.apply(move)
                except (ValueError, KeyError):
                    break
        # Keep the loop's residual for the next turn. A breakthrough is
        # damped and loses its spring-back; an ordinary turn simply carries
        # whatever energy it had, which then decays by wall-clock time.
        self._resonance_energy = res.retained
        self._resonance_ts = now

        self.last_retrieval = {}
        if base_reply is None:
            if skeptic is not None:
                base_reply, source = skeptic.reply, f"skepticism:{skeptic.rule}"
            else:
                base_reply, source = self._base_reply(text, left_policy)
        else:
            source = "provided"
            if not _match_language(base_reply, self.language):
                base_reply, source = _FALLBACK[self.language], f"fallback:{self.language}"

        directive = self.policy.compute(self.state, activations, self.language)
        reply = self.renderer.render(base_reply, directive, self.turn_index,
                                     self.language, posture=left_policy.posture)
        # The association is an aside, so it goes after the shell: the
        # answer keeps its own ending, the remark stays one clean line.
        reply = _with_association(reply, self.last_retrieval.get("association"),
                                  left_policy.detail)

        result = TurnResult(
            reply=reply,
            base_reply=base_reply,
            source=source,
            activations=activations,
            directive=directive,
            state=self.state.snapshot(),
            turn=self.turn_index,
            resonance=res.to_dict(),
            left_brain=left_policy.to_dict(),
            retrieval=dict(self.last_retrieval),
        )
        self.relations.node(user).affinity = self.state.get("affinity")
        self.tiers.record(Episode(
            turn=self.turn_index,
            now=now,
            user=user,
            text=text,
            reply=reply,
            source=source,
            tags=[a.name for a in activations if a.strength > 0],
            affect=dict(result.state),
            salience=salience_of(activations, deltas),
            retrieval=dict(result.retrieval),
        ))
        self._log_event({
            "turn": self.turn_index,
            "now": now,
            "user": user,
            "text": text,
            **result.to_dict(),
        })
        self.turn_index += 1
        self.last_interaction = now
        self.state.last_ts = now
        # A metaphor the user hands us is experience, not just words: an
        # explicit one becomes a card right away; a loose comparison waits in
        # the pending buffer for a confirmation signal. Minting happens after
        # the reply, so the teaching turn cannot echo itself.
        if self.kb.available():
            if looks_like_metaphor(text):
                self.kb.learn_card(text, origin="conversation",
                                   topic=self.kb.topic_for(text))
            elif metaphor_candidate(text):
                self._remember_candidate(text)
        if source.startswith("feynman_kb") and ":" in source:
            card_id = source.split(":", 1)[1]
            self.kb.note_hit(card_id)
        self._last_query = text
        self._last_source = source

        if self.auto_save:
            self.save()
        return result

    def _remember_candidate(self, text: str) -> None:
        key = text.strip()
        if any(c["text"] == key for c in self._pending_metaphors):
            return
        self._pending_metaphors.append({"text": key, "turn": self.turn_index})
        self._pending_metaphors = self._pending_metaphors[-5:]

    def _promote_metaphors(self, text: str, activations: List[Activation]) -> None:
        """Loose comparisons become cards once the user confirms them.

        Confirmation is either praise on the next turn or a later utterance
        that reuses the comparison's wording. Unconfirmed candidates expire
        after PROMOTION_TTL turns and are dropped; a correction cancels them
        immediately.
        """
        if not self._pending_metaphors or not self.kb.available():
            return
        correcting = any(m in text for m in CORRECTION_MARKERS)
        praising = any(a.name == "edge_humor" and a.strength >= 0.5
                       for a in activations)
        # A suspicious mind is challenging the claim, not endorsing it. Only a
        # verbatim quote gets through that; praise and paraphrasing do not.
        suspicious = self.state.get("cortisol") > PROMOTION_CORTISOL_MAX
        keep: List[Dict[str, Any]] = []
        for cand in self._pending_metaphors:
            age = self.turn_index - cand["turn"]
            if correcting:
                continue
            if explicit_confirmation(text, cand["text"]):
                self._promote(cand["text"])
                continue
            if not suspicious and (praising
                                   or shared_pairs(text, cand["text"]) >= 2):
                self._promote(cand["text"])
                continue
            if age >= PROMOTION_TTL:
                continue
            keep.append(cand)
        self._pending_metaphors = keep

    def _promote(self, text: str) -> None:
        self.kb.learn_card(text, origin="conversation-pending",
                           topic=self.kb.topic_for(text))

    def _base_reply(self, text: str, policy: Optional[left_brain.LeftBrainPolicy] = None):
        """Left brain by language. Chinese and English never mix in one reply."""
        if policy is None:
            policy = left_brain.LeftBrainPolicy()
        personal = self._personal_reply(text)
        if personal is not None:
            self.last_retrieval = {}
            return personal
        if self.language == "zh":
            # PyBot only answers in English, so Chinese mode uses the cards.
            found = self.kb.lookup(
                text,
                min_score=policy.min_score,
                limit=policy.limit,
                prefer_kinds=policy.prefer_kinds,
                detail=policy.detail,
                associate=True,
            )
            self.last_retrieval = found.to_dict()
            reply, source = found.text, found.source
            if not _match_language(reply or "", "zh"):
                reply, source = knowledge.answer(text)
        else:
            # The translated card overlay is the mind's English knowledge;
            # PyBot remains the fallback voice when no card matches.
            found = self.kb.lookup(
                text, language="en",
                min_score=policy.min_score,
                limit=policy.limit,
                prefer_kinds=policy.prefer_kinds,
                detail=policy.detail,
                associate=True,
            )
            self.last_retrieval = found.to_dict()
            reply, source = found.text, found.source
            if not reply or not _match_language(reply, "en"):
                reply, source = self.bridge.reply(text)
            if not reply or not _match_language(reply, "en"):
                en_reply, en_source = knowledge.answer_en(text)
                reply, source = en_reply, en_source
        if not reply or not _match_language(reply, self.language):
            reply, source = _FALLBACK[self.language], f"fallback:{self.language}"
        # A guarded mind flags a near-tie instead of pretending to be sure.
        margin = self.last_retrieval.get("margin")
        if (policy.posture == "careful" and source.startswith("feynman_kb")
                and margin is not None and margin < _HEDGE_MARGIN):
            reply = reply + _HEDGE[self.language]
        return reply, source

    def _personal_reply(self, text: str):
        """Answer a direct question from what the mind was told before.

        Semantic facts and the relationship graph were written every turn and
        read by nobody; this is the read path. If the question is not one of
        the known shapes, return None and let the KB/fallback handle it.
        """
        role = _ROLE_QUESTIONS.get(self.language)
        if role is not None:
            match = role.search(text)
            if match:
                name = self.relations.role_of(self.current_user, match.group(1))
                if name:
                    if self.language == "en":
                        return f"Your {match.group(1)} is {name}.", "memory:relations"
                    return f"你{match.group(1)}叫{name}。", "memory:relations"
                return None
        facts = {}
        for entry in self.tiers.about(self.current_user):
            kind = entry["key"].split(":", 1)[0]
            facts[kind] = entry
        for pattern, kind, template in _FACT_QUESTIONS.get(self.language, []):
            if not pattern.search(text):
                continue
            entry = facts.get(kind)
            raw = entry.get("raw") if entry else None
            if raw:
                return template.format(raw), f"memory:semantic:{kind}"
            return None
        return None

    def set_language(self, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        if language != self.language and self.challenge_session is not None:
            progress = self.challenge_session.snapshot()
            self.challenge_session = ChallengeSession(
                self.kb, self, language=language)
            self.challenge_session.restore(progress)
        self.language = language
        if self.auto_save:
            self.save()
        return language

    def _turn_cube(self, activations: List[Activation],
                   gains: Optional[Dict[str, float]] = None) -> None:
        """A strong appraisal physically twists the face its cubie sits on.

        The gate lives in CubeFeedback: it adds hysteresis for a cubie that is
        already being boosted by the front face and a cooldown after each
        turn, so the cube records decisions instead of twitching.
        """
        if not activations:
            return
        target = self.cube_feedback.choose_turn(
            activations, self.cube, self.turn_index, gains)
        if target is not None:
            cubie = CUBIES_BY_NAME[target]
            self.cube.turn(cubie.face, 1)
            self.cube_feedback.note_turn(target, self.turn_index)

    def turn_cube(self, move: str) -> List[str]:
        return self.cube.apply(move)

    def dream(self, now: Optional[float] = None, force: bool = True) -> Optional[Dream]:
        now = self.clock() if now is None else now
        # The audit reads the batch as it was; consolidation flips dreamed=True.
        batch = list(self.tiers.undreamed())
        dream = self.dreamer.consolidate(self.tiers, self.state, self.persona, now,
                                          force=force, language=self.language)
        if dream is not None:
            # Sleep settles the cube the same way it settles the affect state:
            # walk partway home along the inverse of the recent history, so
            # the chronic baseline offset shrinks with the displacement. How
            # far home it walks depends on how heavy the batch was: a dream
            # about ordinary chatter settles fully, a dream about a fight or
            # a loss leaves most of the stance intact.
            ratio = dream_restore_ratio(dream.salience)
            cube_report = self.cube_feedback.partial_restore(
                self.cube, ratio=ratio)
            cube_report["salience"] = dream.salience
            cube_report["restore_ratio"] = ratio
            # Sleep also tidies the learned shelf: near-duplicate cards of the
            # same topic are absorbed into the more useful one. Merge first,
            # then capacity — merging frees slots rather than creating them.
            sleep_merge: List[Dict[str, Any]] = []
            if self.kb.available() and self.kb.learned is not None:
                sleep_merge = self.kb.learned.sleep_merge()
                if sleep_merge:
                    self.kb.reload_learned()
            # Self-audit: repeat-missed questions can mint a gap card or lend
            # an alias to the card they nearly reached, but only after the
            # golden questions still pass. Stress makes the pass read-only.
            audit: Dict[str, Any] = {}
            if self.kb.available() and self.kb.learned is not None and batch:
                peak = max((float((e.affect or {}).get("cortisol", 0.0))
                            for e in batch), default=0.0)
                peak = max(peak, float(self.state.get("cortisol")))
                first_turn = min(getattr(e, "turn", 0) for e in batch)
                prior = next(
                    (e for e in reversed(self.tiers.episodes)
                     if getattr(e, "turn", 0) < first_turn), None)
                try:
                    report = reflect.audit(self.kb, self.kb.learned, batch,
                                           cortisol=peak, prior=prior)
                    audit = report.to_dict()
                    dream.audit = audit
                except Exception as exc:  # a broken audit never costs a dream
                    audit = {"error": f"{type(exc).__name__}: {exc}"}
                    dream.audit = audit
            self._log_event({"turn": self.turn_index, "now": now,
                             "dream": dream.to_dict(),
                             "cube_restore": cube_report,
                             "sleep_merge": sleep_merge,
                             "audit": audit})
            if self.auto_save:
                self.save()
        return dream

    def maintenance(self, now: Optional[float] = None) -> Optional[Dream]:
        """Housekeeping for idle time: consolidate when there is enough to
        consolidate. Independent of the proactive-speech gates."""
        now = self.clock() if now is None else now
        if not self.dreamer.due(self.tiers):
            return None
        return self.dream(now, force=True)

    def challenge(self, reset: bool = False) -> ChallengeSession:
        """The engine owns one challenge session so the CLI, the server and
        the save file share the same progress."""
        if self.challenge_session is None or reset:
            self.challenge_session = ChallengeSession(
                self.kb, self, language=self.language)
        return self.challenge_session

    def say(self, text: str, now: Optional[float] = None) -> str:
        """Render an utterance the mind initiates, without counting a turn."""
        now = self.clock() if now is None else now
        self.state.advance(now, self.persona,
                           baselines=self.cube_feedback.baselines(self.persona, self.cube))
        directive = self.policy.compute(self.state, [], self.language)
        return self.renderer.render(text, directive, self.turn_index,
                                    self.language)

    def _log_event(self, entry: Dict[str, Any]) -> None:
        self.event_log.append(entry)
        overflow = len(self.event_log) - self.EVENT_LOG_LIMIT
        if overflow > 0:
            del self.event_log[:overflow]

    # -- proactive ---------------------------------------------------------

    def tick(self, now: Optional[float] = None, silence_s: float = 600.0,
             cooldown_s: float = 1800.0) -> Optional[str]:
        now = self.clock() if now is None else now
        if self.last_interaction is None:
            return None
        if now - self.last_interaction < silence_s:
            return None
        if self.state.get("affinity") <= 0.6:
            return None
        if self.relations.node(self.current_user).trust < 0.4:
            return None
        # A lonely mind reaches out; a euphoric one is busy enjoying itself.
        # The old 0.35 cutoff meant any mild good mood silenced it entirely.
        if self.state.get("dopamine") >= 0.60:
            return None
        if self.last_proactive is not None and now - self.last_proactive < cooldown_s:
            return None

        if self.dreamer.due(self.tiers):
            self.dream(now, force=True)

        self.state.advance(now, self.persona,
                           baselines=self.cube_feedback.baselines(self.persona, self.cube))
        pool = _PROACTIVE[self.language]
        # Proactive lines go through the same renderer as replies: the mind's
        # current tone should colour what it says first, too.
        message = self.say(pool[self._proactive_cursor % len(pool)], now)
        self._proactive_cursor += 1
        self.last_proactive = now
        self._log_event({"turn": self.turn_index, "now": now,
                         "proactive": message})
        self.turn_index += 1
        if self.auto_save:
            self.save()
        return message

    # -- persistence -------------------------------------------------------

    def resonance_snapshot(self) -> Dict[str, Any]:
        return {
            "energy": round(self._resonance_energy, 4),
            "ts": self._resonance_ts,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "persona": self.persona.name,
            "state": self.state.snapshot(),
            "last_ts": self.state.last_ts,
            "memory": self.memory,
            "turn_index": self.turn_index,
            "last_interaction": self.last_interaction,
            "last_proactive": self.last_proactive,
            "proactive_cursor": self._proactive_cursor,
            "renderer": self.renderer.snapshot(),
            "tiers": self.tiers.snapshot(),
            "relations": self.relations.snapshot(),
            "cube": self.cube.snapshot(),
            "cube_feedback": self.cube_feedback.snapshot(),
            "dreams": self.dreamer.snapshot(),
            "current_user": self.current_user,
            "language": self.language,
            "resonance": self.resonance_snapshot(),
            "left_brain": self.last_left_brain,
            "retrieval": self.last_retrieval,
            "challenge": (self.challenge_session.snapshot()
                          if self.challenge_session is not None else {}),
            "pybot": self._pybot_snapshot(),
            "learned_aliases": {cid: list(p)
                                for cid, p in self.kb.aliases.items()},
            "event_log": self.event_log[-50:],
        }

    def _pybot_snapshot(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"mode": self.bridge.mode}
        try:
            json.dumps(self.bridge.memory, ensure_ascii=False)
            data["memory"] = self.bridge.memory
        except (TypeError, ValueError):
            # PyBot memory is normally plain dicts; if a module stashed
            # something unjsonable, drop it rather than break saving.
            data["memory"] = {}
            data["memory_dropped"] = True
        return data

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path) if path else self.store_path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # Compact on purpose: the snapshot is written every turn, and the
        # indented form is roughly 40% larger for no reader's benefit.
        tmp.write_text(json.dumps(self.snapshot(), ensure_ascii=False,
                                  separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, target)

    @classmethod
    def load(
        cls,
        path: str,
        persona: Optional[PersonaConfig] = None,
        clock: Callable[[], float] = time.time,
        bridge: Optional[PyBotBridge] = None,
        kb: Optional[FeynmanKB] = None,
        auto_save: bool = True,
        language: Optional[str] = None,
    ) -> "MindEngine":
        source = Path(path)
        bad_reason: Optional[str] = None
        data: Any = None
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("save file is not a JSON object")
        except (OSError, ValueError) as exc:
            # A truncated or hand-edited save must not make the app refuse to
            # start. Keep the bytes for inspection and begin from scratch.
            bad_reason = str(exc)
        if bad_reason is not None:
            return cls._fresh_after_bad_save(
                source, bad_reason, persona=persona, clock=clock,
                bridge=bridge, kb=kb, auto_save=auto_save, language=language)
        try:
            return cls.load_from_dict(
                data, persona=persona, clock=clock, bridge=bridge, kb=kb,
                store_path=path, auto_save=auto_save, language=language,
            )
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            # The JSON parsed but its shape is wrong (hand-edited, or written
            # by a different version). Same rule: start, do not refuse.
            return cls._fresh_after_bad_save(
                source, f"{type(exc).__name__}: {exc}", persona=persona,
                clock=clock, bridge=bridge, kb=kb, auto_save=auto_save,
                language=language)

    @classmethod
    def _fresh_after_bad_save(cls, source: Path, bad_reason: str, **kwargs):
        backup: Optional[Path] = None
        try:
            backup = Path(str(source) + ".corrupt")
            shutil.copyfile(source, backup)
        except OSError:
            backup = None
        engine = cls(store_path=str(source), persona=kwargs.get("persona"),
                     clock=kwargs["clock"], bridge=kwargs.get("bridge"),
                     kb=kwargs.get("kb"), auto_save=kwargs["auto_save"],
                     language=kwargs.get("language") or "zh")
        note = f"could not read {source}: {bad_reason}"
        if backup is not None:
            note += f" (a copy is at {backup})"
        engine.load_warning = note
        return engine

    @classmethod
    def load_from_dict(
        cls,
        data: Dict[str, Any],
        persona: Optional[PersonaConfig] = None,
        clock: Callable[[], float] = time.time,
        bridge: Optional[PyBotBridge] = None,
        kb: Optional[FeynmanKB] = None,
        store_path: Optional[str] = None,
        auto_save: bool = False,
        language: Optional[str] = None,
    ) -> "MindEngine":
        persona = persona or get_persona(data.get("persona", "generic"))
        engine = cls(
            persona=persona, store_path=store_path, clock=clock,
            bridge=bridge, kb=kb, auto_save=auto_save,
            language=language or data.get("language", "zh"),
        )
        engine.state.restore(data.get("state", {}), last_ts=data.get("last_ts"))
        # Aliases ride in the save so a learned phrasing survives a restart
        # even when the learned-cards directory is read-only or absent.
        learned = data.get("learned_aliases") or {}
        if isinstance(learned, dict):
            for card_id, phrases in learned.items():
                if not isinstance(phrases, list):
                    continue
                for phrase in phrases:
                    if isinstance(phrase, str):
                        engine.kb.learn_alias(str(card_id), phrase, persist=False)
        engine.memory = data.get("memory", engine.memory)
        engine.turn_index = int(data.get("turn_index", 0))
        engine.last_interaction = data.get("last_interaction")
        engine.last_proactive = data.get("last_proactive")
        engine._proactive_cursor = int(data.get("proactive_cursor", 0))
        engine.renderer.restore(data.get("renderer", {}))
        engine.tiers.restore(data.get("tiers", {}))
        engine.relations.restore(data.get("relations", {}))
        engine.cube.restore(data.get("cube", {}))
        engine.cube_feedback.restore(data.get("cube_feedback", {}))
        engine.dreamer.restore(data.get("dreams", {}))
        engine.current_user = data.get("current_user", "default")
        engine.language = data.get("language", "zh")
        # These two are the last turn's visible policy/retrieval; they used to
        # be saved and silently dropped on load.
        engine.last_left_brain = dict(data.get("left_brain") or {})
        engine.last_retrieval = dict(data.get("retrieval") or {})
        challenge = data.get("challenge") or {}
        if challenge:
            engine.challenge().restore(challenge)
        res = data.get("resonance") or {}
        try:
            engine._resonance_energy = max(0.0, float(res.get("energy", 0.0) or 0.0))
        except (TypeError, ValueError):
            engine._resonance_energy = 0.0
        try:
            engine._resonance_ts = float(res["ts"]) if res.get("ts") is not None else None
        except (TypeError, ValueError):
            engine._resonance_ts = None
        pybot = data.get("pybot") or {}
        if isinstance(pybot.get("memory"), dict):
            engine.bridge.memory = pybot["memory"]
        engine.event_log = list(data.get("event_log", []))[-engine.EVENT_LOG_LIMIT:]
        return engine

    def reset(self, now: Optional[float] = None) -> None:
        now = self.clock() if now is None else now
        self.state = AffectState.from_persona(self.persona, now)
        self.appraisal.reset()
        self.renderer.reset()
        self.memory = {}
        self.tiers = MemoryTiers()
        self.relations = RelationshipGraph()
        self.cube = CubeState()
        self.cube_feedback.reset()
        self.dreamer = Dreamer()
        self.current_user = "default"
        self.event_log.clear()
        self.turn_index = 0
        self.last_interaction = None
        self.last_proactive = None
        self._proactive_cursor = 0
        self._resonance_energy = 0.0
        self._resonance_ts = None
        self.last_left_brain = {}
        self.last_retrieval = {}
        self.challenge_session = None
        if self.store_path and self.store_path.exists():
            self.store_path.unlink()
