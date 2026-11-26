"""Retrieval layer over the public feynman-open knowledge cards.

The corpus is ``~/Projects/feynman-open/knowledge/*.jsonl`` (301 rows at
first wiring): lecture cards, teachable units, physics guardrails, persona
cards, biography. This module turns them into a small TF-IDF index with a
character-bigram tokenizer, pure standard library.

Rows are addressed by ``id`` and the reply is composed from the row's own
fields — nothing is invented, and a low score returns no answer instead of a
bad one.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import assoc, spreading
from .learner import LearnedStore, card_row, normalize

DEFAULT_DIRS = [
    # The repo ships the Feynman cards next to the personas; a checkout answers
    # out of the box. Set OMEGA_KB_PATH to point at your own card directory.
    Path(__file__).resolve().parents[2] / "personas" / "feynman" / "knowledge",
]

# English is a translated overlay: the Chinese cards stay the source of truth,
# and rows here replace only the fields the reply is built from. Cards that
# were already written in English (the verified web quotes) need no overlay.
DEFAULT_EN_DIR = Path(__file__).resolve().parent / "data" / "kb_en"

# Files whose rows carry a usable "text" (or equivalent) body. Rows without
# prose (relations, people, manifest) are skipped automatically.
SKIP_FILES = {"MANIFEST.json"}

# Rows sometimes omit "type"; the file name then names the kind.
STEM_KIND = {
    "teachable_units": "teachable_unit",
    "physics_edges": "physics_edge",
    "method_cards": "method_card",
}

_ASCII = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_CJK = re.compile(r"[一-鿿]+")

# Question scaffolding that says nothing about the subject. Left in, these
# bigrams let any card that happens to contain "什么" win the query.
_STOP_BIGRAMS = {
    "什么", "么是", "是什", "为什", "怎么", "么样", "有没", "多少", "哪个", "哪些",
    "这个", "那个", "就是", "一个", "我们", "你们", "他们", "可以", "是不",
    "不是", "没有", "一下", "知道", "告诉", "为会", "么会", "子是",
    # Conversational scaffolding around a real question: "能不能聊聊X到底
    # 是怎么回事". Left in, these bigrams dilute the subject tokens and push
    # an on-topic card below the answer gate.
    "聊聊", "回事", "到底", "能不",
}

# English questions carry the same scaffolding problem. These words are
# dropped from both the index and the query, so "What is an atom?" is a query
# about atoms rather than a query about "what".
_EN_STOP = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by",
    "with", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "am", "do", "does", "did", "has", "have", "had", "can", "could",
    "will", "would", "should", "may", "might", "must", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "not", "no", "yes", "so", "than", "then", "there", "here", "as", "if",
    "about", "into", "over", "under", "again", "more", "most", "some", "any",
    "all", "own", "same", "such", "only", "very", "just", "up", "down",
    "out", "off", "also", "because", "while", "during", "before", "after",
    "between", "through", "against", "without", "within", "upon", "really",
}


def _stem_en(word: str) -> str:
    """Light, deterministic English stemming for retrieval and grading.

    It deliberately normalises derivational pairs that matter in concept
    answers ("deviation"/"deviates", "conservation"/"conserved",
    "disturbance"/"disturbances"), not just plurals and tense.
    """
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ied") and len(word) > 4:
        return word[:-3] + "y"
    for _ in range(2):
        for suffix in ("ation", "ition", "ance", "ence", "ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _tokens_en(text: str) -> List[str]:
    """Lowercase word tokens, stopwords removed, light stemming."""
    out: List[str] = []
    for tok in _ASCII.findall(text.lower()):
        if tok.isdigit():
            out.append(tok)
            continue
        if len(tok) < 2 or tok in _EN_STOP:
            continue
        out.append(_stem_en(tok))
    return out


# Questions about the speaker's own memory are not questions about
# Feynman's biography. The card layer stays out of them instead of matching
# "remembered" in an obituary sentence.
_EN_EPISODIC = re.compile(
    r"\b(?:why\s+(?:can'?t|cannot|can\s+not|don'?t)\s+(?:i\s+)?remember"
    r"|(?:do|can|could)\s+you\s+remember"
    r"|what\s+did\s+(?:we|i|you)\s+(?:talk|say|discuss)"
    r"|(?:my|your)\s+memory"
    r"|remember\s+(?:yesterday|last\s+time|our\s+conversation))\b",
    re.I,
)
_EN_MEMORY_QUERY = re.compile(r"\bmemory\b", re.I)

# Requests for entertainment belong to the chat layer (PyBot), not to a card
# about Feynman's family joking at dinner. Without this guard "tell me a joke"
# retrieves the dinner-table memory.
_EN_SMALLTALK = re.compile(
    r"\b(?:tell\s+(?:me\s+)?(?:a\s+)?joke|make\s+me\s+laugh"
    r"|tell\s+(?:me\s+)?(?:a\s+)?story|sing\s+(?:me\s+)?(?:a\s+)?song"
    r"|what'?s\s+up)\b",
    re.I,
)

# The same problem in Chinese: "我昨天吃了什么" is a question about the
# speaker's own episode. It must not be answered from a science memory card.
_ZH_EPISODIC = re.compile(
    r"(?:我|我们|你|咱们)(?:昨天|前天|上次|上回|刚才|刚刚|今早|今天早上)"
    r".{0,8}(?:吃|喝|干|做|聊|说|去|见)(?:了|过)?(?:什么|啥|谁|哪儿|哪里)"
)

# "Why does ice float?" is a physics question, not a request for an anecdote
# from the biography. A biography/persona card may only answer a why-question
# when the question is actually about Feynman.
_EN_THIRD_PERSON_WHY = re.compile(
    r"^\s*why\s+(?:does|do|is|are|did|can|would)\b", re.I)

_EN_INTENTS = (
    (re.compile(r"\b(who are you|what are you|your name|introduce yourself)\b", re.I),
     "selfidentity"),
    (re.compile(r"\b(what is science|scientific method|how science works)\b", re.I),
     "sciencemethod"),
    (re.compile(r"\b(memory|forget|forgot|recall)\b", re.I),
     "memory remembering forgetting"),
)


def _expand_en(text: str) -> str:
    extra = [tokens for pattern, tokens in _EN_INTENTS if pattern.search(text)]
    return text + (" " + " ".join(extra) if extra else "")


# Chinese questions need the same treatment: "什么是科学" contributes only
# generic bigrams, so the science-attitude card has nothing distinctive to
# match. A narrow intent expansion adds the card's own subject terms.
_ZH_INTENTS = (
    (re.compile(r"(什么是科学|科学是什么|科学方法|科学态度)"), "科学 态度 证据 专家 观察 权威"),
)


# A bare concept is often split by question grammar ("熵是什么" tokenises as
# "熵是", which no card contains). A small curated lexicon re-adds the concept
# and a couple of subject words so the card's own title/alias can match.
_ZH_TERMS = (
    ("熵", "熵 无序 热力学"),
    ("阻尼", "阻尼 过阻尼 欠阻尼 阻尼比"),
    ("共振", "共振 振荡 频率"),
    ("扩散", "扩散 随机 游走"),
    ("重整化", "重整化 尺度"),
    ("守恒", "守恒 能量"),
    ("反馈", "反馈 放大 抑制"),
    ("权威", "权威 怀疑 尊重 证据 专家"),
    ("浮力", "浮力 漂浮"),
    ("浮在水", "浮力 漂浮"),
    ("费曼是谁", "我是谁 自我介绍"),
    ("费曼是什么人", "我是谁 自我介绍"),
)


def _expand_zh(text: str) -> str:
    extra = [tokens for pattern, tokens in _ZH_INTENTS if pattern.search(text)]
    extra += [tokens for key, tokens in _ZH_TERMS if key in text]
    return text + (" " + " ".join(extra) if extra else "")


# Everyday words rarely appear in the cards, so a natural question about
# them never reaches the right card's vocabulary. This table is bridging
# vocabulary, not answers: it adds the words the corpus itself would use.
_ZH_BRIDGE = (
    ("发热", "热 能量 擦除"),
    ("烫手", "热 能量"),
    ("能量", "能量 热 守恒 动能"),
    # "光" alone must not drag every light question to the double-slit card:
    # a question about light speed was being answered with 双缝. The specific
    # compounds below carry their own vocabulary instead.
    ("光", "光 干涉 双缝"),
    ("光速", "光速 相对论 速度 极限"),
    ("光子", "光子 动量 能量"),
    ("介质", "电介质 折射 色散"),
    ("双缝", "双缝 干涉 量子"),
    ("量子电动力", "路径积分 费曼图 qed 概率幅"),
    ("量子", "量子 叠加 测量"),
    ("数字", "概率 巧合 统计"),
    ("预兆", "概率 巧合 统计"),
    ("巧合", "概率 巧合"),
    ("优雅", "美 观察"),
    ("美", "美 观察 敬畏"),
    ("代码", "计算 程序"),
    ("bug", "问题 拆解"),
    ("头绪", "问题 拆解"),
)


def _bridge_zh(text: str) -> str:
    lower = text.lower()
    extra = [tokens for key, tokens in _ZH_BRIDGE if key in lower]
    return text + (" " + " ".join(extra) if extra else "")


_UNIGRAM_STOP = set(
    "的了吗呢吧啊呀哦嗯是不我你他她它们这那哪谁啥有为在就都也还又很太"
    "会能要想要说给对把被让从到和与及或个一什么怎怎样如何所以因为但是"
    "如果而且之着过它所其此该些每各"
)


def _unigrams(text: str) -> List[str]:
    """Single CJK characters as weak topical evidence.

    Bigrams carry the subject signal; unigrams only widen recall for short
    or everyday phrasings. They are weighted low enough that they can never
    pass the precision gate on their own.
    """
    out: List[str] = []
    for run in _CJK.findall(text):
        out.extend(ch for ch in run if ch not in _UNIGRAM_STOP)
    return out


def _tokens(text: str) -> List[str]:
    text = text.lower()
    out: List[str] = _ASCII.findall(text)
    for run in _CJK.findall(text):
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [tok for tok in out if tok not in _STOP_BIGRAMS]




def _join(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_join(v) for v in value)
    return ""


@dataclass(frozen=True)
class Lookup:
    """Result of one retrieval: text plus the evidence behind it."""
    text: Optional[str]
    source: str
    score: float = 0.0
    margin: Optional[float] = None   # top1 - top2, only when 2+ candidates
    considered: int = 0
    # Query tokens the answering card actually matched. An audit field: it
    # makes a wrong answer diagnosable without re-running the search.
    matched: Tuple[str, ...] = ()
    # Top three (card id, score) pairs, winner included. A miss keeps its
    # near-miss here instead of throwing it away, so a later audit can see
    # which card the question almost reached.
    alternatives: Tuple[Tuple[str, float], ...] = ()
    # One opt-in remark connecting the answering card to a hand-confirmed
    # neighbour (see assoc.py). It is never part of the answer and never
    # changes the source.
    association: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "score": round(self.score, 4),
            "margin": None if self.margin is None else round(self.margin, 4),
            "considered": self.considered,
            "matched": list(self.matched),
            "alternatives": [[card_id, round(score, 4)]
                             for card_id, score in self.alternatives],
            "association": self.association,
        }


@dataclass
class Doc:
    id: str
    kind: str
    title: str
    answer: str
    weights: Dict[str, float] = field(default_factory=dict)
    row: Optional[dict] = None
    language: str = "zh"

    # Challenge mode needs the structured fields of a teachable unit.
    unit: Optional[dict] = None

    def answer_for(self, detail: str = "normal") -> str:
        if self.row is None:
            return self.answer
        text = _answer_for(self.kind, self.row, language=self.language,
                           detail=detail)
        if self.kind == "persona_memory":
            text = _strip_persona_note(text)
        return text or self.answer


# Persona cards sometimes append a note telling the assistant when to recall
# them. It is useful metadata, but it is not part of the answer and reads as a
# leaked instruction once the card is spoken aloud.
_PERSONA_NOTE = re.compile(
    r"(?:这个记忆应在[^。]*。|在吃饭和闲聊时不要[^。]*。"
    r"|This memory should be triggered[^.]*\.|"
    r"[Dd]uring meals and casual conversation,[^.]*\.)"
)


def _strip_persona_note(text: str) -> str:
    return _PERSONA_NOTE.sub("", text).strip()


def _answer_for(kind: str, row: dict, language: str = "zh",
                detail: str = "normal") -> str:
    if detail not in ("brief", "normal", "rich"):
        detail = "normal"
    rich = detail == "rich"
    if language == "en":
        if kind == "teachable_unit":
            parts = [row.get("force_invariant", "")]
            if detail != "brief" and row.get("invariant"):
                parts.append(f"The invariant that pins it down: {row['invariant']}")
            if rich and row.get("check_yourself"):
                parts.append(f"Check yourself: {row['check_yourself']}")
            if rich and row.get("boundary"):
                parts.append(f"Boundary: {row['boundary']}")
            return "\n\n".join(p for p in parts if p)
        if kind == "physics_edge":
            parts = [row.get("phenomenon", "")]
            if detail != "brief" and row.get("invariant"):
                parts.append(f"Invariant: {row['invariant']}")
            if detail != "brief" and row.get("operator"):
                parts.append(f"Operator: {row['operator']}")
            transfer = row.get("transfer")
            if rich and transfer:
                parts.append("The same shape shows up in: " + "; ".join(transfer[:3]))
            return "\n\n".join(p for p in parts if p)
        if kind == "method_card":
            # Method cards have no single body field; their reply is the
            # pattern, one use, and (when the mind has room) the guardrail.
            parts = [row.get("pattern", "")]
            if detail != "brief":
                parts.append(row.get("novel_use", ""))
            if rich:
                parts.append(row.get("guardrail", ""))
            return "\n\n".join(p for p in parts if p)
        return row.get("text", "") or row.get("move", "")
    if kind == "teachable_unit":
        parts = [row.get("force_invariant", "")]
        if detail != "brief" and row.get("invariant"):
            parts.append(f"卡死的那条不变量：{row['invariant']}")
        if rich and row.get("check_yourself"):
            parts.append(f"自己验一下：{row['check_yourself']}")
        if rich and row.get("boundary"):
            parts.append(f"边界：{row['boundary']}")
        return "\n\n".join(p for p in parts if p)
    if kind == "method_card":
        parts = [row.get("pattern", "")]
        if detail != "brief":
            parts.append(row.get("novel_use", ""))
        if rich:
            parts.append(row.get("guardrail", ""))
        return "\n\n".join(p for p in parts if p)
    if kind == "physics_edge":
        parts = [row.get("phenomenon", "")]
        if detail != "brief" and row.get("invariant"):
            parts.append(f"不变量：{row['invariant']}")
        if detail != "brief" and row.get("operator"):
            parts.append(f"式子：{row['operator']}")
        transfer = row.get("transfer")
        if rich and transfer:
            parts.append("同样的形状也出现在：" + "；".join(transfer[:3]))
        return "\n\n".join(p for p in parts if p)
    return row.get("text", "") or row.get("move", "")


class FeynmanKB:
    # A query must clear this normalised score to be answered from the cards.
    MIN_SCORE = 0.30
    # Below the confident gate a mind may still answer, but only when the
    # best card is clearly ahead of the runner-up. This is the "curiosity
    # loosens the gate" branch; it never applies to a near-tie.
    SOFT_SCORE = 0.24
    SOFT_MARGIN = 0.02

    def __init__(self, knowledge_dir: Optional[str] = None, autoload: bool = True,
                 english_dir: Optional[str] = None,
                 learned_dir: Optional[str] = None,
                 spreading: Optional[bool] = None):
        self.docs: List[Doc] = []
        self.docs_en: List[Doc] = []
        self.idf: Dict[str, float] = {}
        self.idf_en: Dict[str, float] = {}
        # Off unless asked for: OMEGA_SPREADING=1 or an explicit argument.
        # The bonus is small and gated, but it still needs golden-net proof
        # before it may become the default.
        self.spreading = (os.environ.get("OMEGA_SPREADING") == "1"
                          if spreading is None else bool(spreading))
        self._spread: Optional[Dict[str, List[Tuple[str, float]]]] = None
        self._assoc: Optional[assoc.AssocGraph] = None
        self.dir: Optional[Path] = None
        self.en_dir: Optional[Path] = None
        self.learned_dir: Optional[Path] = Path(learned_dir) if learned_dir else None
        self.learned: Optional[LearnedStore] = None
        self.aliases: Dict[str, List[str]] = {}
        self._source_rows: List[Tuple[str, dict]] = []
        self.mode = "unloaded"
        if autoload:
            path = self._find_dir(knowledge_dir)
            if path is not None:
                self.load(path, english_dir=english_dir)
        # A learned overlay can exist with no corpus at all: the user may teach
        # the mind before any cards are installed.
        if self.learned_dir is not None and self.learned is None:
            self._load_learned()

    @staticmethod
    def _find_dir(explicit: Optional[str]) -> Optional[Path]:
        env = os.environ.get("OMEGA_KB_PATH")
        candidates = [Path(p) for p in (explicit, env) if p]
        candidates += DEFAULT_DIRS
        for path in candidates:
            if path.is_dir() and any(path.glob("*.jsonl")):
                return path
        return None

    @classmethod
    def disabled(cls) -> "FeynmanKB":
        kb = cls(autoload=False)
        kb.mode = "disabled"
        return kb

    def available(self) -> bool:
        return bool(self.docs)

    def topic_for(self, text: str, language: str = "zh") -> str:
        """Coarse topic of a user utterance: the kind of the nearest card.

        Deliberately not a tag — tags are fine-grained and half of them are
        proper nouns. The card kind is stable, visible in the corpus, and
        enough to protect whole areas of memory from a single busy subject.
        Learned cards are skipped: a new metaphor should join the topic of the
        subject it explains, not the topic "learned_card".
        """
        hits = self.search(text, language=language, limit=5)
        for _score, doc in hits:
            if not doc.id.startswith("learned-"):
                return doc.kind
        return "general"

    def load(self, path: Path, english_dir: Optional[str] = None) -> None:
        docs: List[Doc] = []
        source_rows: List[Tuple[str, dict]] = []
        for file in sorted(path.glob("*.jsonl")):
            if file.name in SKIP_FILES:
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                source_rows.append((file.stem, row))
                doc = self._row_to_doc(file.stem, row)
                if doc is not None:
                    docs.append(doc)
        self.docs = docs
        self._source_rows = source_rows
        self.dir = path
        self.idf = self._build_idf(docs)
        self._spread = None
        self._load_english(english_dir)
        self._load_learned()
        self.mode = f"feynman_kb:{len(docs)}"
        if self.docs_en:
            self.mode += f"+en:{len(self.docs_en)}"
        if len(self.docs) > len(docs):
            self.mode += f"+learned:{len(self.docs) - len(docs)}"

    # -- learned overlay ---------------------------------------------------

    def _load_learned(self) -> None:
        """Append user-taught cards and aliases on top of the read-only corpus.

        Anything retrieved from here is still a normal row: same fields, same
        scoring, same answer builder. Removing ``data/kb_learned`` restores
        the exact pre-learning behaviour.
        """
        store = LearnedStore(self.learned_dir)
        self.learned = store
        known = {doc.id for doc in self.docs}
        added = 0
        for row in store.cards:
            if row.get("id") in known:
                continue
            doc = self._row_to_doc("learned_cards", row)
            if doc is None:
                continue
            self.docs.append(doc)
            known.add(doc.id)
            added += 1
        if added:
            self.idf = self._build_idf(self.docs)
        for card_id, phrase in store.alias_items():
            self._apply_alias(card_id, phrase)

    def _apply_alias(self, card_id: str, phrase: str) -> None:
        doc = next((d for d in self.docs if d.id == card_id), None)
        if doc is None:
            return
        phrases = self.aliases.setdefault(card_id, [])
        if phrase not in phrases:
            phrases.append(phrase)
        n = max(1, len(self.docs))
        for tok in _tokens(phrase):
            if doc.weights.get(tok, 0.0) < 3.0:
                doc.weights[tok] = 3.0
            if tok not in self.idf:
                self.idf[tok] = math.log(1.0 + n)

    def learn_alias(self, card_id: str, phrase: str,
                    persist: bool = True) -> bool:
        """Attach a phrasing to a card after the mind was praised for it."""
        if not phrase or not any(d.id == card_id for d in self.docs):
            return False
        self._apply_alias(card_id, phrase)
        if persist and self.learned is not None:
            self.learned.add_alias(card_id, phrase)
        return True

    def learn_card(self, text: str, origin: str = "conversation",
                   persist: bool = True, topic: str = "general") -> bool:
        """Turn a metaphor the user taught into a retrievable card.

        The gate lives in ``learner.card_row``: only comparisons become cards.
        A card that closely resembles an existing one becomes a revision, so
        the new wording replaces the old one without losing history. The new
        card joins ``self.docs`` immediately for later turns; it is not
        retroactively used to answer the turn that taught it.

        ``topic`` is coarser than a tag: it is the kind of corpus card the
        comparison is closest to, so a burst of physics metaphors cannot
        crowd art out of memory at the card cap.
        """
        row = card_row(text, origin=origin, topic=topic)
        if row is None:
            return False
        if persist and self.learned is not None:
            changed = self.learned.add_card(row)
            if not changed:
                return False
            row = next((c for c in self.learned.cards
                        if c.get("id") == row["id"]), row)
            # A revision may have merged into a *different* id.
            row = self._merged_row(row, text)
        doc = self._row_to_doc("learned_cards", row)
        if doc is None:
            return False
        for i, existing in enumerate(self.docs):
            if existing.id == doc.id:
                self.docs[i] = doc
                break
        else:
            self.docs.append(doc)
        self._drop_archived_docs()
        self.idf = self._build_idf(self.docs)
        return True

    def _merged_row(self, row: Dict[str, Any], text: str) -> Dict[str, Any]:
        """After a merge the new text lives on whichever card it resembled."""
        body = text.strip()
        for card in (self.learned.cards if self.learned else []):
            if card.get("body") == body or card.get("text") == row.get("text"):
                return card
        return row

    def _drop_archived_docs(self) -> None:
        if self.learned is None:
            return
        active = {c.get("id") for c in self.learned.cards}
        self.docs = [d for d in self.docs
                     if not d.id.startswith("learned-") or d.id in active]

    def reload_learned(self) -> None:
        """Rebuild the learned overlay after a batch change (sleep merges).

        Corpus docs are untouched; learned docs are re-derived from the store
        and every learned alias is re-applied. Aliases added in memory only
        (``persist=False``) are kept as they are.
        """
        if self.learned is None:
            return
        self.docs = [d for d in self.docs if not d.id.startswith("learned-")]
        known = {doc.id for doc in self.docs}
        for row in self.learned.cards:
            if row.get("id") in known:
                continue
            doc = self._row_to_doc("learned_cards", row)
            if doc is None:
                continue
            self.docs.append(doc)
            known.add(doc.id)
        for card_id, phrase in self.learned.alias_items():
            self._apply_alias(card_id, phrase)
        self.idf = self._build_idf(self.docs)
        self._spread = None

    def note_hit(self, card_id: str) -> None:
        if self.learned is not None and card_id.startswith("learned-"):
            self.learned.record_hit(card_id)

    def note_praise(self, card_id: str) -> None:
        if self.learned is not None and card_id.startswith("learned-"):
            self.learned.record_praise(card_id)

    # -- English overlay ---------------------------------------------------

    @staticmethod
    def _find_en_dir(explicit: Optional[str]) -> Optional[Path]:
        env = os.environ.get("OMEGA_KB_EN_PATH")
        candidates = [Path(p) for p in (explicit, env) if p]
        candidates.append(DEFAULT_EN_DIR)
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.jsonl")):
                return candidate
            if candidate.is_file() and candidate.suffix == ".jsonl":
                return candidate
        return None

    @staticmethod
    def _already_english(row: dict) -> bool:
        body = " ".join(str(row.get(k) or "") for k in ("text", "move"))
        return bool(body.strip()) and not _CJK.search(body)

    def _load_english(self, english_dir: Optional[str]) -> None:
        """Merge the translated overlay onto the Chinese rows.

        A row with no translation and no original English body is skipped:
        better an English mode that says less than one that leaks Chinese.
        """
        path = self._find_en_dir(english_dir)
        if path is None:
            self.docs_en = []
            self.idf_en = {}
            return
        files = [path] if path.is_file() else sorted(path.glob("*.jsonl"))
        translated: Dict[str, dict] = {}
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("id"):
                    translated[str(row["id"])] = row

        docs_en: List[Doc] = []
        for stem, row in self._source_rows:
            rid = str(row.get("id") or "")
            if rid and rid in translated:
                merged = dict(row)
                merged.update(translated[rid])
            elif self._already_english(row):
                merged = row
            else:
                continue
            doc = self._row_to_doc(stem, merged, language="en")
            if doc is not None:
                docs_en.append(doc)
        self.docs_en = docs_en
        self.idf_en = self._build_idf(docs_en)
        self.en_dir = path

    def _row_to_doc(self, stem: str, row: dict,
                    language: str = "zh") -> Optional[Doc]:
        kind = str(row.get("type") or STEM_KIND.get(stem, stem)).replace(" ", "_")
        answer = _answer_for(kind, row, language=language)
        if kind == "persona_memory":
            answer = _strip_persona_note(answer)
        if not answer:
            return None
        title = _join(row.get("title") or row.get("name_zh") or "")
        doc = Doc(
            id=str(row.get("id") or f"{stem}:{len(self.docs)}"),
            kind=kind,
            title=title or _join(row.get("word") or ""),
            answer=answer,
            row=row,
            language=language,
            unit=row if kind == "teachable_unit" else None,
        )
        # Title/tag/alias tokens are the card's subject; body tokens are
        # context. The weight ratio is what keeps "原子是什么" off a biography
        # that merely mentions atoms.
        self._add_weight(doc, title, 4.0, language=language)
        self._add_weight(doc, answer, 1.0, language=language)
        # The invariant is the card's concept sentence. In the English
        # overlay, physics_edge rows have no translated title, so without
        # this the concept word ("damping") would sit at body weight and fail
        # the precision gate.
        self._add_weight(doc, _join(row.get("invariant")), 2.5,
                         language=language)
        self._add_weight(doc, _join(row.get("tags")), 3.0, language=language)
        self._add_weight(doc, _join(row.get("aliases")), 3.0, language=language)
        entry_weight = 3.5 if language == "en" else 2.5
        self._add_weight(doc, _join(row.get("entry_question")), entry_weight,
                         language=language)
        self._add_weight(doc, _join(row.get("transfer")), 0.5, language=language)
        boost = row.get("retrieval_boost")
        if isinstance(boost, (int, float)) or (isinstance(boost, str) and boost.isdigit()):
            for tok in list(doc.weights):
                doc.weights[tok] *= 1.0 + min(float(boost), 50.0) / 100.0
        if language != "en":
            # Weak single-character evidence, below the 2.5 precision gate.
            for text, weight in ((title, 1.0), (_join(row.get("tags")), 0.8),
                                 (_join(row.get("aliases")), 1.0),
                                 (answer, 0.25)):
                for tok in _unigrams(text):
                    if doc.weights.get(tok, 0.0) < weight:
                        doc.weights[tok] = weight
        if not doc.weights:
            return None
        return doc

    @staticmethod
    def _add_weight(doc: Doc, text: str, weight: float,
                    language: str = "zh") -> None:
        if not text:
            return
        tokens = _tokens_en(text) if language == "en" else _tokens(text)
        for tok in tokens:
            if doc.weights.get(tok, 0.0) < weight:
                doc.weights[tok] = weight

    def _build_idf(self, docs: Optional[List[Doc]] = None) -> Dict[str, float]:
        docs = self.docs if docs is None else docs
        df: Dict[str, int] = {}
        for doc in docs:
            for tok in doc.weights:
                df[tok] = df.get(tok, 0) + 1
        n = max(1, len(docs))
        return {tok: math.log(1.0 + n / d) for tok, d in df.items()}

    def search(self, text: str, limit: int = 3,
               language: str = "zh",
               min_score: Optional[float] = None,
               prefer_kinds: Iterable[str] = ()) -> List[Tuple[float, Doc]]:
        docs = self.docs_en if language == "en" else self.docs
        idf = self.idf_en if language == "en" else self.idf
        if not docs:
            return []
        if language == "en" and (_EN_EPISODIC.search(text)
                                 or _EN_SMALLTALK.search(text)):
            return []
        if language != "en" and _ZH_EPISODIC.search(text):
            return []
        expanded = (_expand_en(text) if language == "en"
                    else _bridge_zh(_expand_zh(text)))
        query = set(_tokens_en(expanded) if language == "en"
                    else _tokens(expanded))
        if language != "en" and not query:
            query = set()
        if not query:
            return []
        uni_query = set(_unigrams(expanded)) if language != "en" else set()
        # Only tokens the corpus actually contains can match a card. Counting
        # out-of-vocabulary query tokens in the denominator punished natural
        # paraphrases ("费曼怎么看待权威") for words no card could ever have.
        qweight = {tok: idf[tok] ** 2 for tok in query if tok in idf}
        # Tokens that appear in more than a quarter of the cards carry no
        # subject information ("energy" in a physics corpus, "father" in a bio).
        idf_cut = math.log(1.0 + len(docs) / max(1.0, 0.25 * len(docs)))
        qweight = {t: w for t, w in qweight.items() if idf.get(t, 0.0) >= idf_cut}
        total = sum(qweight.values())
        if not total:
            return []
        wants_cjk = bool(_CJK.search(text))
        third_person_why = (language == "en" and "feynman" not in text.lower()
                            and bool(_EN_THIRD_PERSON_WHY.match(text)))
        prefer = set(prefer_kinds)
        scored: List[Tuple[float, Doc]] = []
        for doc in docs:
            if wants_cjk and not _CJK.search(doc.answer):
                continue  # English-only cards never answer a Chinese question
            if language == "en" and _CJK.search(doc.answer):
                continue  # a leaked Chinese row is not an English answer
            if (third_person_why
                    and doc.kind in ("biography", "persona_memory")):
                continue  # a story about Feynman is not a physics answer
            common = set(qweight) & doc.weights.keys()
            if not common:
                continue
            # Precision gate: the card must match the query on its own
            # subject (title / tags / aliases / entry question, weight 2.5+).
            # One stray shared body bigram is how "天气" used to reach a
            # cosmology card.
            if not any(doc.weights[tok] >= 2.5 for tok in common):
                continue
            # A second shape of the same failure: a question that names
            # several things but touches the card through exactly one token.
            # "为什么彩虹总是七种颜色呢" must not be answered by a van
            # anecdote merely because its tags include 颜色, nor by a damped-
            # oscillator card merely because its title contains the glue
            # bigram 总是. Only a one-subject question (query tokens all on
            # one subject, as in "What is damping?") may be answered on one
            # token. OOV words count here: 彩虹 is exactly the evidence that
            # the shared token is not what was asked about. Two-token
            # questions are exempt because a short compound gets split into
            # overlapping bigrams ("费曼图" -> 费曼 + 曼图).
            if len(common) == 1 and len(query) > 2:
                continue
            # Coverage: how much of the query the card accounts for. A long
            # card is not penalised for being long; a stray bigram is.
            weights = [qweight[tok] / total for tok in common]
            best = max(
                w * min(doc.weights[tok], 4.0) / 4.0
                for w, tok in zip(weights, common)
            )
            coverage = sum(
                w * min(doc.weights[tok], 4.0) / 4.0 for w, tok in zip(weights, common)
            )
            score = 0.7 * best + 0.3 * coverage
            # Two independent title tokens is a much stronger subject signal
            # than one, and it offsets the dilution from cross-word bigrams
            # ("曼是" in "费曼是谁") that no card can match.
            title_hits = sum(1 for tok in common if doc.weights[tok] >= 3.9)
            if title_hits >= 2:
                score = min(1.0, score + 0.06 * (title_hits - 1))
            if doc.kind in prefer:
                # A small, deterministic nudge: the right brain can prefer a
                # kind of card, but it cannot make an irrelevant card win.
                score *= 1.08
            scored.append((score, doc))
        # Weak unigram evidence tops the bigram score up; it can never open
        # the gate on its own because doc unigram weights stay below 2.5.
        if uni_query and scored:
            uni_idf = {t: idf.get(t, 0.0) for t in uni_query if t in idf}
            uni_total = sum(uni_idf.values()) or 1.0
            lifted: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                common_uni = set(uni_idf) & doc.weights.keys()
                if common_uni:
                    cov = sum(uni_idf[t] * min(doc.weights[t], 1.0)
                              for t in common_uni) / uni_total
                    score = min(1.0, score + 0.12 * cov)
                lifted.append((score, doc))
            scored = lifted
        if self.aliases:
            norm_text = normalize(text)
            boosted: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                if doc.id in self.aliases and any(
                        normalize(p) and normalize(p) in norm_text
                        for p in self.aliases[doc.id]):
                    score = min(1.0, score + 0.10)
                boosted.append((score, doc))
            scored = boosted
        if self.spreading and scored:
            index = self._spread_index(language, docs)
            subject = set(qweight)
            spread: List[Tuple[float, Doc]] = []
            for score, doc in scored:
                if spreading.precision_open(subject, doc):
                    score = min(1.0, score + spreading.spread_bonus(
                        subject, doc, index))
                spread.append((score, doc))
            scored = spread
        if language == "en" and _EN_MEMORY_QUERY.search(text):
            # A general memory question may only use a card whose subject is
            # memory; a body mention is not enough.
            scored = [(score, doc) for score, doc in scored
                      if doc.weights.get("memory", 0.0) >= 2.5]
        if min_score is not None:
            scored = [(score, doc) for score, doc in scored
                      if score >= min_score]
        # On an exact score tie, a physics card beats a story about Feynman.
        # "部分子是什么" should get the parton model, not the one-line
        # biography that merely names it; the biography still wins whenever
        # it scores higher or the question is about Feynman himself.
        narrative = ("biography", "persona_memory")
        # A one-word question ("what is polarization") scores 1.0 on every
        # card with that word in its title, so the vacuum-polarization card
        # tied with the optics card and won on file order. On an exact tie the
        # card that lists the question itself as an alias is the one written
        # to answer it. This is a tie-break only: it never lifts a lower score.
        asked = normalize(text)

        def _asked_verbatim(doc: Doc) -> int:
            aliases = doc.row.get("aliases") or ()
            if isinstance(aliases, str):
                aliases = (aliases,)
            return 1 if asked and any(
                isinstance(a, str) and normalize(a) == asked
                for a in aliases) else 0

        scored.sort(key=lambda pair: (pair[0], _asked_verbatim(pair[1]),
                                      0 if pair[1].kind in narrative else 1),
                    reverse=True)
        return scored[:max(1, limit)]

    def _spread_index(self, language: str, docs: List["Doc"]
                      ) -> Dict[str, List[Tuple[str, float]]]:
        if self._spread is None:
            self._spread = {}
        if language not in self._spread:
            self._spread[language] = spreading.npmi_neighbors(docs)
        return self._spread[language]

    def lookup(self, text: str, language: str = "zh",
               min_score: Optional[float] = None, limit: int = 3,
               prefer_kinds: Iterable[str] = (),
               detail: str = "normal",
               associate: bool = False) -> Lookup:
        """Retrieve one answer plus the score, runner-up margin and how many
        candidates were considered. ``limit`` is what makes the margin
        observable: with limit=1 there is no runner-up to compare against.

        ``associate=True`` appends at most one hand-confirmed association
        remark. It is chosen *after* the answer is fixed and never touches
        the source or the score.
        """
        found = self._lookup(text, language=language, min_score=min_score,
                             limit=limit, prefer_kinds=prefer_kinds,
                             detail=detail)
        if associate and found.text:
            return replace(found, association=self._association_for(
                found, language))
        return found

    def _association_for(self, found: Lookup,
                         language: str) -> Optional[str]:
        if ":" not in found.source:
            return None
        card_id = found.source.split(":", 1)[1]
        if card_id.startswith("learned-") or card_id == "no_match":
            return None
        graph = self._assoc_graph()
        if graph is None or not graph.edges:
            return None
        docs = self.docs_en if language == "en" else self.docs
        titles = {doc.id: doc.title for doc in docs}
        # Never explain a card the answer already names: the user asked
        # about it, and the runner-up may be the same subject restated.
        skip = {card_id}
        skip.update(other for other, _score in found.alternatives)
        return graph.sentence(card_id, language=language, titles=titles,
                              exclude=skip)

    def _assoc_graph(self) -> Optional[assoc.AssocGraph]:
        if self._assoc is None:
            self._assoc = assoc.load()
        return self._assoc

    def _lookup(self, text: str, language: str = "zh",
                min_score: Optional[float] = None, limit: int = 3,
                prefer_kinds: Iterable[str] = (),
                detail: str = "normal") -> Lookup:
        prefix = "feynman_kb_en" if language == "en" else "feynman_kb"
        threshold = self.MIN_SCORE if min_score is None else min_score
        # Search's own min_score would drop the soft band before lookup can
        # consider it. An explicit gate above the default is a deliberate
        # withholding ("be careful / say nothing"), so it stays strict.
        soft_ok = min_score is None or min_score <= self.MIN_SCORE
        floor = min(threshold, self.SOFT_SCORE) if soft_ok else threshold
        hits = self.search(text, language=language, limit=limit,
                           min_score=floor, prefer_kinds=prefer_kinds)
        alternatives = tuple((doc.id, score) for score, doc in hits[:3])
        if not hits or hits[0][0] < threshold:
            if hits:
                score, doc = hits[0]
                margin = score - hits[1][0] if len(hits) > 1 else None
                if (score >= self.SOFT_SCORE
                        and (margin is None or margin >= self.SOFT_MARGIN)):
                    return Lookup(
                        text=doc.answer_for(detail),
                        source=f"{prefix}_soft:{doc.id}",
                        score=score,
                        margin=margin,
                        considered=len(hits),
                        matched=self._matched_terms(text, doc, language),
                        alternatives=alternatives,
                    )
                # A user-taught restatement must not veto the corpus card it
                # restates: a near-tie across the learned/corpus boundary is
                # wording overlap, not subject doubt. Two cards of the same
                # class tied is still a near-tie, and still refused.
                tie = self._class_tie(hits, prefix, detail, text, language,
                                      alternatives=alternatives)
                if tie is not None:
                    return tie
            return Lookup(None, f"{prefix}:no_match", considered=len(hits),
                          alternatives=alternatives)
        score, doc = hits[0]
        margin = score - hits[1][0] if len(hits) > 1 else None
        return Lookup(
            text=doc.answer_for(detail),
            source=f"{prefix}:{doc.id}",
            score=score,
            margin=margin,
            considered=len(hits),
            matched=self._matched_terms(text, doc, language),
            alternatives=alternatives,
        )

    @staticmethod
    def _matched_terms(text: str, doc: Doc, language: str) -> Tuple[str, ...]:
        """Query tokens that hit this card — the evidence behind the answer."""
        if language == "en":
            tokens = set(_tokens_en(_expand_en(text)))
        else:
            tokens = set(_tokens(_bridge_zh(_expand_zh(text))))
        shared = sorted(tok for tok in tokens if tok in doc.weights)
        return tuple(shared[:10])

    def _class_tie(self, hits: List[Tuple[float, Doc]], prefix: str,
                   detail: str, text: str = "",
                   language: str = "zh",
                   alternatives: Tuple[Tuple[str, float], ...] = ()
                   ) -> Optional[Lookup]:
        """Resolve a learned/corpus near-tie in favour of the corpus card.

        The margin guard exists to avoid answering the wrong *subject*. A
        user-taught card that scores within the margin of a corpus card on
        the same question is a restatement of that subject, so the corpus
        answer is safe. Same-class ties are untouched and still refused.
        """
        if len(hits) < 2:
            return None
        (score0, doc0), (score1, doc1) = hits[0], hits[1]
        if score0 - score1 > self.SOFT_MARGIN:
            return None
        learned0 = doc0.id.startswith("learned-")
        learned1 = doc1.id.startswith("learned-")
        if learned0 == learned1:
            return None
        corpus, corpus_score = ((doc1, score1) if learned0
                                else (doc0, score0))
        if corpus_score < self.SOFT_SCORE:
            return None
        return Lookup(
            text=corpus.answer_for(detail),
            source=f"{prefix}_soft:{corpus.id}",
            score=corpus_score,
            margin=None,
            considered=len(hits),
            matched=self._matched_terms(text, corpus, language),
            alternatives=alternatives,
        )

    def answer(self, text: str, language: str = "zh",
               min_score: Optional[float] = None, limit: int = 3,
               prefer_kinds: Iterable[str] = (),
               detail: str = "normal") -> Tuple[Optional[str], str]:
        found = self.lookup(text, language=language, min_score=min_score,
                            limit=limit, prefer_kinds=prefer_kinds,
                            detail=detail)
        return found.text, found.source

    def teachable_units(self, language: str = "zh") -> List[Doc]:
        docs = self.docs_en if language == "en" else self.docs
        return [doc for doc in docs if doc.unit is not None]
