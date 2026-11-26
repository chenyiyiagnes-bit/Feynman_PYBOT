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

from . import dialogue, evidence, knowledge, left_brain, reflect, resonance, review, skepticism
from .appraisal import Activation, AppraisalEngine
from .challenge import ChallengeSession
from .config import PersonaConfig, get_persona
from .cubies import CUBIES_BY_NAME
from .cube import CubeState
from .cube_feedback import CubeFeedback, dream_restore_ratio
from .dream import Dream, Dreamer
from .feynman_kb import FeynmanKB, sibling_package
from .learner import (
    CORRECTION_MARKERS,
    PROMOTION_CORTISOL_MAX,
    PROMOTION_TTL,
    explicit_confirmation,
    looks_like_metaphor,
    metaphor_candidate,
    shared_pairs,
)
from .memory import Episode, MemoryTiers, fact_changes, salience_of
from .policy import Policy, StyleDirective
from .pybot_bridge import PyBotBridge
from .relations import RelationshipGraph
from .renderer import Renderer
from .state import AffectState


def _default_kb(persona: PersonaConfig, language: str) -> FeynmanKB:
    """The KB a mind without a supplied package should read.

    The persona's own package wins; a mind that switches language gets the
    same family's package in that language (``x-zh`` → ``x-en``) rather than
    whichever package the directory listing happens to put first.
    """
    name = str(getattr(persona, "name", "") or "")
    if name and name != "generic":
        own = FeynmanKB.for_package(name)
        if own is not None and own.language == language:
            return own
        sibling = sibling_package(name, language)
        if sibling is not None:
            return FeynmanKB.for_package(sibling.name, root=sibling.parent)
    fallback = FeynmanKB.for_language(language)
    return fallback if fallback is not None else FeynmanKB(language=language)


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


@dataclass
class _DialogueFocus:
    card_id: str
    source: str
    spoken: List[str]
    last_reply: str
    revision: str


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
_FOLLOWUP_GAP = {
    "zh": "我手头就这些资料。你想先弄清哪一步？",
    "en": "That's as far as my sources go. Which step should we look at again?",
}
_ACK = {"zh": "嗯。", "en": "Got it."}

# A guarded turn says so when two cards are almost tied. This is a style
# sentence, not a fact: it only comments on the retrieval margin.
_HEDGE = {
    "zh": "这个我把握不算大，先说最贴近的一个。",
    "en": "I'm not fully certain here; this is the closest I have.",
}
_HEDGE_MARGIN = 0.06

# Proactive interference: when a new subject arrives while most of the cube
# is still displaced, the mind is carrying the last subject into this one.
# The confidence line widens a little and the hedge may fire outside
# ``careful`` — but only once per shift, never more than this many turns
# apart, and only when the cube is actually loaded.
_INTERFERENCE_DISPLACEMENT = 0.35
_INTERFERENCE_MARGIN = 0.04
_HEDGE_COOLDOWN = 2


def hedge_applies(margin: Optional[float], posture: str,
                  interference: bool = False,
                  limit: float = _HEDGE_MARGIN) -> bool:
    """Whether the near-tie hedge belongs on this answer.

    A clean cube and a confident posture give exactly the old behaviour:
    only ``careful`` with a margin inside the line. Interference widens the
    line and lets the hedge through outside ``careful`` — a style sentence,
    never a change to the answer.
    """
    if margin is None:
        return False
    if interference:
        limit += _INTERFERENCE_MARGIN
    return margin < limit and (posture == "careful" or interference)

# Retrieval inhibition: a card spoken within this many turns is demoted in
# the next lookup (see FeynmanKB.INHIBIT_FACTOR). Inhibition only breaks
# near-ties; a question whose only card is that one still gets it.
INHIBIT_TURNS = 3

# A recast is a reframing, and the cube is reoriented with it: same facts,
# different face in front. One reframe per few turns, so the frame settles
# instead of spinning.
REFRAME_COOLDOWN = 3

# Reflection has to stop somewhere or it becomes rumination: the same card
# may show at most this many new faces in a row, after which the nudge is
# answered with the card again, plainly.
REFLECT_CAP = 2

# A curious mind may ask one question, and then has to listen for a while.
# Three turns is long enough that a question reads as interest rather than
# as an interview.
QUESTION_COOLDOWN = 3

# A bare follow-up that asks for the reason behind the last answer. It must
# carry no subject of its own: "为什么天空是蓝的" is a new question, not a
# request to go deeper into the rainbow.
_WHY_ZH = re.compile(
    r"^(?:为什么|为啥|凭什么|何以|怎么知道|依据是什么|根据是什么)"
    r"(?:呢|呀|啊|嘛)?[？?。！!~～\s]*$")
_WHY_EN = re.compile(
    r"^(?:why|why so|how come|how so|on what grounds|what makes that so)"
    r"[?.! ]*$", re.IGNORECASE)

# A bare "I don't get it". Repeating the card is useless, so the mind may
# recast it through a confirmed analogy — once per card, never twice.
_CONFUSION_ZH = re.compile(
    r"^(?:没懂|没听懂|没明白|不明白|什么意思|啥意思|换个说法|再说一遍|"
    r"说人话|打个比方)[？?。！!~～\s]*$")
_CONFUSION_EN = re.compile(
    r"^(?:i (?:don'?t|do not) (?:get|understand)(?: it)?|"
    r"what do you mean|explain (?:that )?(?:again|simpler)|"
    r"in plain (?:english|words)|put it another way)[?.! ]*$",
    re.IGNORECASE)

# A bare nudge to go round the last answer once more. It carries no subject
# of its own, so it asks for a *different face of the same card*, never a
# new lookup: a paraphrase with a subject ("为什么天空是蓝的") is a new
# question and must not be forced onto the previous card.
_REFLECT_ZH = re.compile(
    r"^(?:再想想|再想一次|再想一遍|重新想想|再琢磨(?:一下)?|"
    r"往深里想|再深入(?:一点|些)?|深入(?:一点|些)|"
    r"换个角度(?:想)?|换个方向(?:想)?|从另一个角度(?:想)?|"
    r"还有呢|还有吗|再多说(?:一)?点|再展开(?:说说|讲讲)?)"
    r"[？?。！!~～\s]*$")
_REFLECT_EN = re.compile(
    r"^(?:think (?:about it )?again|think (?:it )?through again|"
    r"what else|go deeper|dig deeper|another angle|"
    r"from another angle|keep going)[?.! ]*$", re.IGNORECASE)

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
        left_responder: Optional[Any] = None,
        auto_save: bool = True,
        language: str = "zh",
    ):
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        self.persona = persona or get_persona("generic")
        self.clock = clock
        self.store_path = Path(store_path) if store_path else None
        self.auto_save = auto_save and self.store_path is not None
        self.bridge = bridge or PyBotBridge()
        # One mind, one package: a KB the caller supplies keeps its own
        # package language, and a language switch loads the package for the
        # new language (or reports that none is installed).
        # Without a supplied KB, prefer the package this persona actually is
        # (its cards, profile and learned store); only a persona that is not
        # a package falls back to "the first package of this language".
        self.kb = (kb if kb is not None else
                   _default_kb(self.persona, language))
        self._kb_supplied = kb is not None
        # Optional corpus-callosum responder (bridge.LeftResponder). Kept
        # duck-typed and default-None: the right brain must run without the
        # left brain installed, and with None the turn path is unchanged.
        # The standalone prototype can use it after a Chinese lookup miss.
        self.left_responder = left_responder
        self.language = language
        self.retrieval_mode = os.getenv("OMEGA_RETRIEVAL", "legacy")
        if self.retrieval_mode not in ("legacy", "shadow", "rrf"):
            self.retrieval_mode = "legacy"
        self.structured_evidence = os.getenv(
            "OMEGA_STRUCTURED_EVIDENCE", "1" if self.kb.managed_package else "0") != "0"
        self.posture_mode = os.getenv("OMEGA_POSTURE", "affect_cube")
        if self.posture_mode not in ("fixed", "affect", "affect_cube"):
            self.posture_mode = "affect_cube"
        self.review_learning = self.kb.uses_review()

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
        self._dialogue_focus: Optional[_DialogueFocus] = None
        self._topics = dialogue.TopicHistory()
        # The card answered on the previous turn, for the association line to
        # continue: "following on from what I just said". Only an immediate
        # neighbour counts, and only when that turn was answered from a card.
        self._last_card: Optional[str] = None
        self._last_card_turn: int = -10
        # Cards already visited by a "why?" chain, so a cyclic pair of
        # premise edges cannot ping-pong forever.
        self._why_seen: set = set()
        # Cards already recast through an analogy after "I don't get it".
        # Saying the same recast twice is worse than saying nothing.
        self._recast_seen: set = set()
        # When the mind last asked a question of its own. A curious mind may
        # ask, but not every turn: an engine that interrogates the user is
        # not curious, it is tiresome.
        self._last_question_turn: int = -10 ** 9
        # Cards answered in the last few turns, with the turn number. The
        # next lookup demotes them a little (retrieval inhibition).
        self._recent_cards: Dict[str, int] = {}
        # When the hedge last fired. Proactive interference reads the
        # previous card from ``_last_card`` and the load from the cube.
        self._last_hedge_turn: int = -10 ** 9
        # When the cube was last reframed (a recast turns the whole cube).
        self._last_reframe_turn: int = -10 ** 9
        # Faces of a card already shown by reflection, so a second
        # "think again" moves on instead of repeating the first. Also the
        # card and count of the running run of reflections, capped by
        # ``REFLECT_CAP`` so it cannot become rumination.
        self._spoken_facets: Dict[str, set] = {}
        self._reflect_card: Optional[str] = None
        self._reflect_streak: int = 0
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
            self._dialogue_focus = None
            self._last_query = self._last_source = None
            self._pending_metaphors.clear()
        if self.review_learning and self._uses_knowledge_package():
            self.kb.refresh_review()
        changes = fact_changes(text, self.language)
        # The cube's return path: how long each displaced cubie has held the
        # front face this turn, and which baselines the affect state should
        # relax toward while the arrangement stays scrambled.
        self.cube_feedback.observe(self.cube, self.turn_index)
        effective_baselines = (self.cube_feedback.baselines(self.persona, self.cube)
                               if self.posture_mode == "affect_cube" else None)
        self.state.advance(now, self.persona, baselines=effective_baselines)

        gains = (self.cube_feedback.gains(self.cube)
                 if self.posture_mode == "affect_cube" else None)
        activations = self.appraisal.evaluate(text, self.turn_index, gains=gains)

        review_learning = self.review_learning and self._uses_knowledge_package()
        # Praise is the only feedback signal this mind has. If the previous
        # turn was answered from a card and this turn is praise, remember the
        # phrasing that led there; later paraphrases of it will score higher.
        if (self._last_source
                and self._last_source.startswith(self.kb.source_prefix + ":")
                and self._last_query
                and any(a.name == "edge_humor" and a.strength >= 0.5
                        for a in activations)):
            card_id = self._last_source.split(":", 1)[1]
            if review_learning:
                review.queue_candidate(self.kb.learned, "alias",
                                {"card_id": card_id, "phrase": self._last_query})
            else:
                self.kb.learn_alias(card_id, self._last_query)
            self.kb.note_praise(card_id)

        # A loose comparison from an earlier turn earns a card once the user
        # confirms it — by praising, or by using it again. Promotion happens
        # before retrieval so the confirming turn can already use it.
        if not review_learning:
            self._promote_metaphors(text, activations)

        # A grand claim gets the evidence reflex before any card is retrieved.
        # The jolt is part of the affect dynamics, not a separate UI message.
        skeptic = skepticism.check(text, self.language) if base_reply is None and not changes else None
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
        cube_bias = (self.cube_feedback.bias(self.cube)
                     if self.posture_mode == "affect_cube" else None)
        left_policy = (left_brain.LeftBrainPolicy() if self.posture_mode == "fixed"
                       else left_brain.plan_for(self.state, self.persona, activations,
                                                cube_bias=cube_bias,
                                                cube_signal=self.cube_feedback.signal))
        self.last_left_brain = left_policy.to_dict()

        self.relations.touch(user, now)
        self.relations.mentions(text, now, user)
        self._turn_cube(activations, gains if gains is not None else {})

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
            if changes:
                self.last_retrieval = {"dialogue_act": "memory_update",
                                       "fact_kinds": [change.kind for change in changes]}
                changed = changes[-1]
                if changed.operation == "retract" and changed.kind == "lives_in":
                    base_reply = ("明白，你已经不住在那里了。" if self.language == "zh"
                                  else "Got it. You don't live there anymore.")
                elif changed.kind == "lives_in":
                    base_reply = (f"记住了，你现在住在{changed.raw}。" if self.language == "zh"
                                  else f"Got it. You live in {changed.raw} now.")
                else:
                    base_reply = "我记住了。" if self.language == "zh" else "I'll remember that."
                source = "memory:update"
            elif skeptic is not None:
                base_reply, source = skeptic.reply, f"skepticism:{skeptic.rule}"
            else:
                base_reply, source = self._base_reply(text, left_policy, cube_bias)
        else:
            source = "provided"
            if not _match_language(base_reply, self.language):
                base_reply, source = _FALLBACK[self.language], f"fallback:{self.language}"

        style_state = (AffectState.from_persona(self.persona, now)
                       if self.posture_mode == "fixed" else self.state)
        directive = self.policy.compute(
            style_state, [] if self.posture_mode == "fixed" else activations, self.language)
        presented = self._present_evidence(base_reply, source, left_policy.detail)
        # The aside is one trailing line: a verified magnitude anchor wins
        # over the association when the question asked for one, then a
        # curiosity question may take the same slot.
        aside = (self.last_retrieval.get("scale")
                 or self.last_retrieval.get("association"))
        reply = self.renderer.render(
            presented if presented is not None else base_reply,
            directive, self.turn_index, self.language,
            posture=left_policy.posture,
            preserve=presented is not None or bool(self.last_retrieval.get("dialogue_act"))
            or source.startswith("fallback") or bool(self.last_retrieval.get("evidence_plan")))
        # The association is an aside, so it goes after the shell: the
        # answer keeps its own ending, the remark stays one clean line.
        reply = _with_association(reply, aside, left_policy.detail)
        # Curiosity, if there is room: the question was built by the lookup
        # from a real near-tie or a confirmed premise edge. It rides in the
        # same single trailing slot as an aside, and only in ``explore``.
        question = self.last_retrieval.get("question")
        if (question and not aside
                and left_policy.posture == "explore"
                and self.turn_index - self._last_question_turn
                >= QUESTION_COOLDOWN):
            spoken = _with_association(reply, question, left_policy.detail)
            if spoken != reply:
                reply = spoken
                self._last_question_turn = self.turn_index
        if self._uses_knowledge_package():
            self._remember_dialogue(source, base_reply, reply, text)

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
        ), changes=changes)
        self._log_event({
            "turn": self.turn_index,
            "now": now,
            "user": user,
            "text": text,
            **result.to_dict(),
        })
        # Remember where this answer came from, so the next turn's aside can
        # continue the thought. Only a card counts; small talk breaks the
        # thread on purpose.
        if source.startswith(("feynman_kb:", "feynman_kb_en:")):
            self._last_card = source.split(":", 1)[1]
        else:
            self._last_card = None
        self._last_card_turn = self.turn_index
        # A different card (or no card) ends a run of reflections: the cap
        # counts consecutive returns to one subject, not lifetime visits.
        if self._reflect_card is not None and self._last_card != self._reflect_card:
            self._reflect_streak = 0
        self.turn_index += 1
        self.last_interaction = now
        self.state.last_ts = now
        # A metaphor the user hands us is experience, not just words: an
        # explicit one becomes a card right away; a loose comparison waits in
        # the pending buffer for a confirmation signal. Minting happens after
        # the reply, so the teaching turn cannot echo itself.
        if self.kb.available() and not changes:
            if review_learning and (looks_like_metaphor(text) or metaphor_candidate(text)):
                review.queue_candidate(self.kb.learned, "analogy", {"text": text})
            elif looks_like_metaphor(text):
                self.kb.learn_card(text, origin="conversation",
                                   topic=self.kb.topic_for(text))
            elif metaphor_candidate(text):
                self._remember_candidate(text)
        if source.startswith(self.kb.source_prefix) and ":" in source:
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

    def _left_reply(self, text: str, cube_bias: Optional[Any] = None):
        """Ask the optional typesetter responder; None means "use the old path".

        The responder is expected to fail closed already, but the engine does
        not trust an add-on with the conversation: any exception here also
        counts as "no answer", and the turn continues as if it were absent.
        """
        if self.left_responder is None:
            return None
        # Which card this turn is about, when the gate needs one: the slot
        # filler may only use the user's word on a card that accounts for it.
        card_id = None
        gated = getattr(self.left_responder, "gate_active", False)
        if gated:
            found = self.kb.lookup(text, language=self.language, limit=1)
            if found.text and ":" in found.source:
                card_id = found.source.split(":", 1)[1]
        # Only a responder that runs the gate is asked for a card id: earlier
        # responders (and tests) keep their narrower ``reply`` signature.
        extra = {"card_id": card_id} if gated else {}
        try:
            result = self.left_responder.reply(
                text, self.state.values, self.persona,
                cube_depth=getattr(cube_bias, "depth", 0),
                cube_verbosity=getattr(cube_bias, "verbosity", 0.0),
                nonce=self.turn_index,
                session_id=self.current_user,
                **extra,
            )
        except Exception:  # noqa: BLE001 - the add-on may never break a turn
            return None
        if result is None:
            return None
        body = getattr(result, "text", None)
        if not body or not _match_language(body, "zh"):
            return None
        return body, "left:typesetter"

    def _present_evidence(self, text: str, source: str, detail: str) -> str | None:
        act = self.last_retrieval.get("dialogue_act", "")
        prefix, _, card_id = source.partition(":")
        own = self.kb.source_prefix
        if (self.left_responder is None or not self._uses_knowledge_package()
                or prefix not in (own, own + "_soft") or not card_id
                or card_id == "no_match" or self.last_retrieval.get("source") != source
                or act == "repeat" or text.endswith(_HEDGE[self.language])):
            return None
        allowed = evidence.candidates(text, self.language)
        if not allowed:
            return None
        try:
            plan = self.left_responder.plan_evidence(
                text, source=source, language=self.language, detail=detail,
                dialogue_act=act, candidates=allowed)
        except Exception:  # noqa: BLE001 - a display add-on must never break a turn
            return None
        presented = evidence.apply_breaks(text, allowed, plan)
        if presented is not None:
            self.last_retrieval["evidence_layout"] = list(plan)
        return presented

    def _continuity_card(self) -> Optional[str]:
        """The previous turn's card, only if that turn was the one before."""
        if self._last_card and self._last_card_turn == self.turn_index - 1:
            return self._last_card
        return None

    def _is_why_followup(self, text: str) -> bool:
        """A short, subject-free "why?" that asks to go one step deeper."""
        stripped = text.strip()
        if not stripped or len(stripped) > 12:
            return False
        pattern = _WHY_EN if self.language == "en" else _WHY_ZH
        return bool(pattern.match(stripped))

    def _is_confusion(self, text: str) -> bool:
        """A short, subject-free "I don't get it" — a recast, not a query."""
        stripped = text.strip()
        if not stripped or len(stripped) > 24:
            return False
        pattern = _CONFUSION_EN if self.language == "en" else _CONFUSION_ZH
        return bool(pattern.match(stripped))

    def _is_reflection(self, text: str) -> bool:
        """A short, subject-free nudge to think the last answer again.

        Only the phrase itself triggers this: similarity between questions
        is too blunt to tell "same subject, new question" from "say that
        again", and a wrong guess here answers the wrong question.
        """
        stripped = text.strip()
        if not stripped or len(stripped) > 16:
            return False
        pattern = _REFLECT_EN if self.language == "en" else _REFLECT_ZH
        return bool(pattern.match(stripped))

    def _inhibit_set(self, text: str = "") -> set:
        """Cards answered so recently that repeating them would circle.

        Re-asking the very same question is not circling: the user wants the
        same answer, so inhibition is suspended for an identical repeat.
        """
        if self._last_query is not None and text.strip() == self._last_query.strip():
            return set()
        cutoff = self.turn_index - INHIBIT_TURNS
        return {cid for cid, turn in self._recent_cards.items() if turn > cutoff}

    def _note_card(self, source: str) -> None:
        """Remember the card just spoken, for retrieval inhibition."""
        if not source.startswith("feynman_kb"):
            return
        prefix, _, card_id = source.partition(":")
        if not card_id or "_soft" in prefix:
            return
        self._recent_cards[card_id] = self.turn_index
        if len(self._recent_cards) > 12:
            for old in sorted(self._recent_cards,
                              key=self._recent_cards.get)[:-12]:
                del self._recent_cards[old]

    def _base_reply(self, text: str, policy: Optional[left_brain.LeftBrainPolicy] = None,
                    cube_bias: Optional[Any] = None):
        """Left brain by language. Chinese and English never mix in one reply."""
        if policy is None:
            policy = left_brain.LeftBrainPolicy()
        personal = self._personal_reply(text)
        if personal is not None:
            self.last_retrieval = {"dialogue_act": "memory_query"}
            return personal
        packaged = self._uses_knowledge_package()
        target = ""
        if packaged:
            resume = dialogue.resume_target(text, self.language)
            if resume is not None:
                doc = self._topics.resolve(self.current_user, resume, self.kb.by_id)
                self.last_retrieval = {"dialogue_act": "resume_gap"}
                if doc is not None and (not self.review_learning or self.kb.public_doc(doc)):
                    if self.structured_evidence and evidence.structured_card(doc):
                        plan, reason = evidence.compose(doc, "conclusion", policy.detail,
                                                        self.persona.prefer_examples)
                        if not plan:
                            self.last_retrieval["decision"] = reason
                            return dialogue.RESUME_GAP[self.language], f"clarification:{self.language}"
                        self.last_retrieval["evidence_plan"] = plan
                        reply = " ".join(part["text"] for part in plan)
                    else:
                        reply = doc.answer_for(policy.detail)
                    self.last_retrieval.update(dialogue_act="resume", card_id=doc.id,
                                               revision=dialogue.card_revision(doc))
                    return reply, f"{self.kb.source_root}:{doc.id}"
                return dialogue.RESUME_GAP[self.language], f"clarification:{self.language}"
            social = knowledge.social_reply(text, self.language)
            if social is not None:
                self.last_retrieval = {"dialogue_act": "social"}
                return social, "builtin_en" if self.language == "en" else "builtin_kb"
            kind, target = self.kb.core.followup(text)
            followup = self._contextual_reply(kind, target)
            if followup is not None:
                return followup

        # "I don't get it" is not a new question. Say the last card a second
        # way, through a confirmed analogy — and only once per card, because
        # a recast repeated is just the same explanation again.
        if self._is_confusion(text):
            previous = self._continuity_card()
            if previous and previous not in self._recast_seen:
                again = self.kb.recast(previous, language=self.language)
                if again is not None:
                    self._recast_seen.add(previous)
                    self.last_retrieval = {"source": again[1],
                                           "recast": again[0]}
                    # A recast is a reframe: the cube turns in the mind's
                    # hands so the next turn is seen from a different face.
                    # Nothing physical moves; only what counts as front.
                    if (self.turn_index - self._last_reframe_turn
                            >= REFRAME_COOLDOWN):
                        self.cube.reframe(1)
                        self._last_reframe_turn = self.turn_index
                    return again[0], again[1]
        else:
            self._recast_seen.clear()
        # A bare nudge ("think again") is reflection: the same card, one
        # face it has not shown yet — a confirmed premise, boundary, limit
        # or analogy, never an invented angle. The card is fixed; only the
        # angle moves. Past the cap, or with no face left, the card is said
        # again plainly: silence about having nothing new is worse.
        if self._is_reflection(text):
            previous = self._continuity_card()
            if previous:
                source = f"{self.kb.source_root}:{previous}"
                streak = (self._reflect_streak + 1
                          if self._reflect_card == previous else 1)
                hit = None
                if streak <= REFLECT_CAP:
                    hit = self.kb.rethink(
                        previous, language=self.language,
                        spoken=self._spoken_facets.get(previous, ()))
                if hit is not None:
                    line, key = hit
                    self._spoken_facets.setdefault(previous, set()).add(key)
                    self._reflect_card = previous
                    self._reflect_streak = streak
                    self.last_retrieval = {"source": source,
                                           "reflection": key}
                    self._note_card(source)
                    return line, source
                body = self.kb.card_body(previous,
                                         language=self.language,
                                         detail=policy.detail)
                if body:
                    self._reflect_card = previous
                    self._reflect_streak = streak
                    self.last_retrieval = {"source": source,
                                           "reflection": "repeat"}
                    self._note_card(source)
                    return body, source
        # A bare "why?" is not a new query: it asks the previous answer to go
        # one premise deeper. Only a confirmed premise edge counts, so this
        # never invents a reason; with no edge the ordinary lookup runs.
        if self._is_why_followup(text):
            previous = self._continuity_card()
            if previous:
                deeper = self.kb.follow_premise(
                    previous, language=self.language,
                    detail=policy.detail, avoid=self._why_seen)
                if deeper is not None:
                    self.last_retrieval = deeper.to_dict()
                    self._why_seen.add(previous)
                    if ":" in deeper.source:
                        self._why_seen.add(deeper.source.split(":", 1)[1])
                    return deeper.text, deeper.source
            else:
                self._why_seen.clear()
        else:
            self._why_seen.clear()
        found = self.kb.lookup(
            text, language=self.language, min_score=None if packaged else policy.min_score,
            limit=3 if packaged else policy.limit,
            prefer_kinds=() if packaged else policy.prefer_kinds,
            detail=policy.detail, associate=True,
            public_only=packaged and self.review_learning,
            mode=self.retrieval_mode if packaged else "legacy",
            structured=packaged and self.structured_evidence,
            prefer_examples=self.persona.prefer_examples,
            previous_card=self._continuity_card(),
            scale=packaged,
            curiosity=packaged and policy.posture == "explore",
            inhibit=self._inhibit_set(text))
        self.last_retrieval = found.to_dict()
        reply, source = found.text, found.source
        if (target and self._dialogue_focus is not None
                and source.startswith(self.kb.source_prefix + "_soft:")):
            # A weak match elsewhere cannot answer a follow-up that could
            # not be grounded in the current topic.
            self.last_retrieval["unverified_followup"] = True
            reply = None
        if not _match_language(reply or "", self.language):
            if packaged and self.last_retrieval.get("decision") == "conflict":
                self.last_retrieval["dialogue_act"] = "clarify_conflict"
                return (("我找到的材料给出了不同结论。你问的是哪一种条件？"
                         if self.language == "zh" else
                         "My sources give different answers. Which conditions do you mean?"),
                        f"clarification:{self.language}")
            if packaged:
                return _FALLBACK[self.language], f"fallback:{self.language}"
            # Keep the standalone prototype's fallback tools for callers
            # without a persona knowledge package.
            if self.language == "zh":
                bridged = self._left_reply(text, cube_bias)
                reply, source = bridged if bridged is not None else knowledge.answer(text)
            else:

                reply, source = self.bridge.reply(text)
                if not _match_language(reply or "", "en"):
                    reply, source = knowledge.answer_en(text)
        if not reply or not _match_language(reply, self.language):
            reply, source = _FALLBACK[self.language], f"fallback:{self.language}"
        # A guarded mind flags a near-tie instead of pretending to be sure.
        # Proactive interference widens the line: a new subject arriving while
        # the cube is still loaded from the last one gets the same hedge even
        # outside ``careful``. It never changes which card answered or what
        # the card says.
        margin = self.last_retrieval.get("margin")
        interference = (
            source.startswith(self.kb.source_prefix)
            and self._last_card is not None
            and ":" in source
            and source.split(":", 1)[1] != self._last_card
            and self.cube_feedback.signal.displacement
            >= _INTERFERENCE_DISPLACEMENT
            and self.turn_index - self._last_hedge_turn >= _HEDGE_COOLDOWN)
        if (source.startswith(self.kb.source_prefix)
                and not self.last_retrieval.get("evidence_plan")
                and (hedge_applies(margin, policy.posture, interference)
                     or (policy.posture == "careful"
                         and margin is not None and margin < _HEDGE_MARGIN))):
            reply = reply + _HEDGE[self.language]
            self.last_retrieval["hedge"] = True
            self._last_hedge_turn = self.turn_index
        self._note_card(source)
        return reply, source

    def _uses_knowledge_package(self) -> bool:
        directory = self.kb.dir
        return directory is not None and directory.name == "knowledge"

    def _contextual_reply(self, kind: str, target: str) -> Optional[tuple[str, str]]:
        if kind == "ack":
            self.last_retrieval = {"dialogue_act": kind}
            return _ACK[self.language], f"ack:{self.language}"
        focus = self._dialogue_focus
        if not kind or focus is None:
            return None
        card = self.kb.by_id.get(focus.card_id)
        if (card is None or focus.revision != dialogue.card_revision(card)
                or (self.review_learning and not self.kb.public_doc(card))):
            return None
        edge = None
        plan = []
        if kind == "repeat":
            passage = focus.last_reply
        elif self.structured_evidence and evidence.structured_card(card) and not target:
            intent = {"why": "mechanism", "boundary": "condition", "rephrase": "example"}[kind]
            plan, _ = evidence.compose(card, intent, "brief")
            if plan and not any(part["role"] == intent and all(
                    part["text"] not in said for said in focus.spoken) for part in plan):
                plan = []
            passage = " ".join(part["text"] for part in plan) if plan else None
        else:
            passage, edge = self.kb.follow_up_evidence(focus.card_id, kind, target, tuple(focus.spoken))
        if not _match_language(passage or "", self.language):
            if target:
                return None
            self.last_retrieval = {"dialogue_act": "followup_gap", "card_id": focus.card_id}
            return _FOLLOWUP_GAP[self.language], f"fallback:{self.language}"
        self.last_retrieval = {
            "dialogue_act": kind, "source": focus.source,
            "card_id": focus.card_id, "revision": dialogue.card_revision(card),
            "evidence": passage,
        }
        if edge is not None:
            self.last_retrieval["association_edge"] = edge
        if plan:
            self.last_retrieval["evidence_plan"] = plan
        return passage, focus.source

    def _remember_dialogue(self, source: str, base: str, reply: str, query: str) -> None:
        kind = self.last_retrieval.get("dialogue_act")
        focus = self._dialogue_focus
        if kind in ("ack", "followup_gap", "resume_gap", "memory_query", "memory_update"):
            return
        prefix, _, card_id = source.partition(":")
        own = self.kb.source_prefix
        doc = self.kb.by_id.get(card_id)
        if prefix not in (own, own + "_soft") or doc is None:
            self._dialogue_focus = None
            return
        revision = dialogue.card_revision(doc)
        if kind == "repeat":
            return
        if focus is not None and focus.card_id == card_id and focus.revision == revision:
            focus.spoken = (focus.spoken + [base, reply])[-12:]
            focus.last_reply = reply
        else:
            self._dialogue_focus = _DialogueFocus(card_id, source, [base, reply], reply, revision)
        if not kind and self.kb.public_doc(doc):
            self._topics.remember(self.current_user, dialogue.Topic(
                card_id, revision, doc.title, query[:200]))

    def _personal_reply(self, text: str):
        """Answer a direct question from what the mind was told before.

        Semantic facts and the relationship graph were written every turn and
        read by nobody; this is the read path. If the question is not one of
        the known shapes, return None and let the KB/fallback handle it.
        """
        if self.language == "en":
            text = text.replace("’", "'")
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
        historical = dialogue.historical_kind(text, self.language)
        if historical:
            entry = self.tiers.previous(self.current_user, historical)
            if entry is not None and entry.get("raw"):
                return dialogue.MEMORY_PAST[self.language].format(entry["raw"]), "memory:history"
            return dialogue.MEMORY_UNKNOWN[self.language], "memory:unknown"
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
            return dialogue.MEMORY_UNKNOWN[self.language], "memory:unknown"
        return None

    def set_language(self, language: str) -> str:
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        if language != self.language and language != self.kb.language:
            if self._kb_supplied and (self.kb.available()
                                      or self.kb.mode != "disabled"):
                # A supplied KB is a package: one package, one language. The
                # other language is a different package and a different mind,
                # built by the caller that owns the package paths.
                raise ValueError(
                    f"this mind holds the {self.kb.language!r} package; "
                    f"build a mind for the {language!r} package instead")
            if not self._kb_supplied:
                # Same family first (x-zh → x-en); another character's
                # package would answer in the right language but the wrong
                # voice, so it is only the last resort.
                sibling = sibling_package(str(getattr(self.persona, "name", "")),
                                          language)
                replacement = (FeynmanKB.for_package(sibling.name,
                                                     root=sibling.parent)
                               if sibling is not None
                               else FeynmanKB.for_language(language))
                if replacement is None:
                    raise ValueError(f"no {language!r} package is installed")
                self.kb = replacement
            # A disabled KB holds no cards in any language; the mind may
            # still change modes for tone and fallbacks.
        if language != self.language and self.challenge_session is not None:
            progress = self.challenge_session.snapshot()
            self.challenge_session = ChallengeSession(
                self.kb, self, language=language)
            self.challenge_session.restore(progress)
        if language != self.language:
            self._dialogue_focus = None
            self._topics = dialogue.TopicHistory()
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
                                           cortisol=peak, prior=prior,
                                           review_only=(self.review_learning
                                                        and self._uses_knowledge_package()))
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
            "package_scope": (self.kb.source_root if self._uses_knowledge_package() else None),
            "state": self.state.snapshot(),
            "last_ts": self.state.last_ts,
            "memory": self.memory,
            "turn_index": self.turn_index,
            "last_interaction": self.last_interaction,
            "last_proactive": self.last_proactive,
            "proactive_cursor": self._proactive_cursor,
            "renderer": self.renderer.snapshot(),
            "tiers": self.tiers.snapshot(),
            "topics": self._topics.snapshot(self.kb.source_root),
            "relations": self.relations.snapshot(),
            "cube": self.cube.snapshot(),
            "cube_feedback": self.cube_feedback.snapshot(),
            "dreams": self.dreamer.snapshot(),
            "current_user": self.current_user,
            "language": self.language,
            "resonance": self.resonance_snapshot(),
            "left_brain": self.last_left_brain,
            "retrieval": self.last_retrieval,
            "last_question_turn": self._last_question_turn,
            "recent_cards": dict(self._recent_cards),
            "last_hedge_turn": self._last_hedge_turn,
            "last_reframe_turn": self._last_reframe_turn,
            "spoken_facets": {cid: sorted(facets)
                              for cid, facets in self._spoken_facets.items()},
            "reflect_card": self._reflect_card,
            "reflect_streak": self._reflect_streak,
            "challenge": (self.challenge_session.snapshot()
                          if self.challenge_session is not None else {}),
            "pybot": self._pybot_snapshot(),
            "learned_aliases": self.kb.unreviewed_aliases(),
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
        if (kb is not None and kb.managed_package
                and data.get("package_scope") not in (None, kb.source_root)):
            raise ValueError("state file belongs to another persona package")
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
        if (engine.kb.managed_package
                and data.get("package_scope") not in (None, engine.kb.source_root)):
            raise ValueError("state file belongs to another persona package")
        engine.state.restore(data.get("state", {}), last_ts=data.get("last_ts"))
        # Aliases ride in the save so a learned phrasing survives a restart
        # even when the learned-cards directory is read-only or absent.
        learned = data.get("learned_aliases") or {}
        if isinstance(learned, dict) and not engine.review_learning:
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
        engine._topics.restore(data.get("topics"), engine.kb.source_root)
        engine.relations.restore(data.get("relations", {}))
        engine.cube.restore(data.get("cube", {}))
        engine.cube_feedback.restore(data.get("cube_feedback", {}))
        engine.dreamer.restore(data.get("dreams", {}))
        engine.current_user = data.get("current_user", "default")
        # These two are the last turn's visible policy/retrieval; they used to
        # be saved and silently dropped on load.
        engine.last_left_brain = dict(data.get("left_brain") or {})
        engine.last_retrieval = dict(data.get("retrieval") or {})
        engine._last_question_turn = int(data.get("last_question_turn",
                                                  -10 ** 9))
        engine._recent_cards = {str(k): int(v) for k, v in
                                (data.get("recent_cards") or {}).items()}
        engine._last_hedge_turn = int(data.get("last_hedge_turn", -10 ** 9))
        engine._last_reframe_turn = int(data.get("last_reframe_turn", -10 ** 9))
        engine._spoken_facets = {
            str(cid): {str(key) for key in keys}
            for cid, keys in (data.get("spoken_facets") or {}).items()
        }
        reflect_card = data.get("reflect_card")
        engine._reflect_card = str(reflect_card) if reflect_card else None
        engine._reflect_streak = int(data.get("reflect_streak", 0))
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
        self._dialogue_focus = None
        self._topics = dialogue.TopicHistory()
        self.challenge_session = None
        self._last_question_turn = -10 ** 9
        self._recent_cards = {}
        self._last_hedge_turn = -10 ** 9
        self._last_reframe_turn = -10 ** 9
        if self.store_path and self.store_path.exists():
            self.store_path.unlink()
