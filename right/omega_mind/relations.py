"""Relationship graph: who the mind knows, how close, and what connects them.

Nodes are people (the mind itself is ``self``). Edges are typed relations with
a weight. Per-user affinity lives here so switching interlocutors does not
carry one person's closeness over to another.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SELF = "self"

# "我妈妈叫王芳" / "我老板是张伟" — only named people become nodes. Unnamed
# roles stay out; inventing a node from every kinship word would be noise.
_MENTION = re.compile(
    r"我(?:的)?(妈|妈妈|母亲|爸|爸爸|父亲|哥|哥哥|姐|姐姐|弟|弟弟|妹|妹妹|"
    r"老板|同事|朋友|妻子|老公|丈夫|男朋友|女朋友)(?:叫|是|名叫)([一-鿿A-Za-z·]{1,6})"
)

# The same introduction in English. The role word is translated to the
# canonical Chinese tag so a person introduced in one language can be asked
# about in the other.
_MENTION_EN = re.compile(
    r"\bmy (mom|mother|dad|father|brother|sister|boss|wife|husband|"
    r"boyfriend|girlfriend)(?:'s name)? (?:is|called)\s+"
    r"([A-Za-z][\w .'\-]{0,30})",
    re.IGNORECASE,
)

# Every variant (and the English word) maps to one canonical tag, so
# "我妈" and "我妈妈" are the same relationship.
ROLE_ALIASES = {
    "妈": "妈妈", "mother": "妈妈", "mom": "妈妈",
    "母亲": "妈妈",
    "爸": "爸爸", "father": "爸爸", "dad": "爸爸",
    "哥": "哥哥", "brother": "哥哥",
    "姐": "姐姐", "sister": "姐姐",
    "弟": "弟弟", "妹": "妹妹",
    "老板": "老板", "boss": "老板",
    "妻子": "妻子", "wife": "妻子",
    "老公": "丈夫", "丈夫": "丈夫", "husband": "丈夫",
    "男朋友": "男朋友", "boyfriend": "男朋友",
    "女朋友": "女朋友", "girlfriend": "女朋友",
}

# A captured "name" that is really an interrogative ("我妈叫什么" -> "什么")
# is not a person, and neither is "a teacher" in "my mom is a teacher".
_NAME_STOP = {"什么", "啥", "谁", "哪", "哪儿", "哪里", "名字", "什么名字",
              "what", "who", "which", "where", "name"}
_EN_CLAUSE = re.compile(r"[,.;:!?\n]|\s+(?:and|but|because|so)\s+", re.I)


def canonical_role(role: str) -> str:
    return ROLE_ALIASES.get(role) or ROLE_ALIASES.get(role.lower(), role)


def _name_ok(name: str) -> bool:
    if not name or name in _NAME_STOP:
        return False
    first = name.split()[0].lower() if name.split() else ""
    return first not in ("a", "an", "the")


def _clean_en_name(name: str) -> str:
    return _EN_CLAUSE.split(name.strip(), maxsplit=1)[0].strip().strip(".'\"-")


@dataclass
class PersonNode:
    name: str
    affinity: float = 0.2
    trust: float = 0.5          # 0..1, how much the mind believes them
    interactions: int = 0
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    src: str
    dst: str
    kind: str
    weight: float = 1.0


_NODE_FIELDS = tuple(PersonNode.__dataclass_fields__)
_EDGE_FIELDS = tuple(Edge.__dataclass_fields__)


class RelationshipGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, PersonNode] = {
            SELF: PersonNode(name=SELF, affinity=1.0, trust=1.0),
        }
        self.edges: Dict[tuple, Edge] = {}

    def node(self, name: str) -> PersonNode:
        if name not in self.nodes:
            self.nodes[name] = PersonNode(name=name)
        return self.nodes[name]

    def link(self, src: str, dst: str, kind: str, delta: float = 0.0) -> Edge:
        key = (src, dst, kind)
        edge = self.edges.get(key)
        if edge is None:
            edge = self.edges[key] = Edge(src, dst, kind, weight=1.0)
        edge.weight = max(0.0, min(10.0, edge.weight + delta))
        return edge

    # -- per-turn bookkeeping ----------------------------------------------

    def touch(self, user: str, now: float, affinity_delta: float = 0.0,
              note: str = "") -> PersonNode:
        person = self.node(user)
        person.interactions += 1
        person.last_seen = now
        if person.first_seen is None:
            person.first_seen = now
        person.affinity = max(0.0, min(1.0, person.affinity + affinity_delta))
        # Trust follows closeness with a lag: it should not swing on one turn,
        # but it must not stay frozen at its initial 0.5 either.
        person.trust = max(0.0, min(1.0,
                                    person.trust + 0.05 * (person.affinity - person.trust)))
        self.link(SELF, user, "talks_to", 0.01)
        if note:
            tag = note[:12]
            if tag not in person.tags:
                person.tags.append(tag)
                person.tags = person.tags[-12:]
        return person

    def mentions(self, text: str, now: float, user: str) -> List[str]:
        """Register named relatives/colleagues mentioned in the text."""
        found = []
        pairs = [(role, name) for role, name in _MENTION.findall(text)]
        pairs += [(canonical_role(role), _clean_en_name(name))
                  for role, name in _MENTION_EN.findall(text)]
        for role, name in pairs:
            if not _name_ok(name):
                continue
            role = canonical_role(role)
            person = self.node(name)
            person.tags = sorted(set(person.tags + [role]))[-12:]
            person.last_seen = now
            self.link(user, name, role)
            found.append(name)
        return found

    def affinity(self, user: str, default: float = 0.2) -> float:
        node = self.nodes.get(user)
        return default if node is None else node.affinity

    def role_of(self, user: str, role: str) -> Optional[str]:
        """Name of the person the user introduced as ``role``, if known.

        Reads the tags written by ``mentions``; this is what makes the graph
        answer "what is my sister's name?" instead of being a write-only log.
        The query is normalised too, so "我妈" finds a "妈妈" tag and an
        English "mom" finds a relationship introduced in Chinese.
        """
        role = canonical_role(role)
        for (src, dst, kind), _edge in self.edges.items():
            if src != user or kind != role:
                continue
            node = self.nodes.get(dst)
            if node is not None and role in node.tags:
                return node.name
        return None

    # -- views -------------------------------------------------------------

    def about(self, user: str) -> Dict[str, Any]:
        node = self.nodes.get(user)
        edges = [
            asdict(e) for e in self.edges.values()
            if user in (e.src, e.dst)
        ]
        return {"person": node.snapshot() if node else None, "edges": edges}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.snapshot() for k, v in self.nodes.items()},
            "edges": [asdict(e) for e in self.edges.values()],
        }

    def restore(self, data: Dict[str, Any]) -> None:
        # Tolerate saves written by another version: extra keys are dropped,
        # entries without the required fields are skipped. A save must never
        # stop the app from starting.
        nodes = {}
        for name, node in data.get("nodes", {}).items():
            if isinstance(node, dict) and "name" in node:
                nodes[name] = PersonNode(
                    **{k: node[k] for k in _NODE_FIELDS if k in node})
        self.nodes = nodes or {SELF: PersonNode(name=SELF, affinity=1.0,
                                                trust=1.0)}
        edges = {}
        for e in data.get("edges", []):
            if isinstance(e, dict) and all(
                    key in e for key in ("src", "dst", "kind")):
                edge = Edge(**{k: e[k] for k in _EDGE_FIELDS if k in e})
                edges[(edge.src, edge.dst, edge.kind)] = edge
        self.edges = edges
