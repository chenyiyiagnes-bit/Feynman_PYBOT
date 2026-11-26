"""User-scoped topic references and conversation action phrases."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

_TOPIC_LIMIT = 8
_USER_LIMIT = 32
_RESUME = {
    "zh": re.compile(r"^(?:我们)?(?:能不能|可以|再)?(?:回到|接着聊|继续聊|再说说)(.+?)[。？?！!]*$"),
    "en": re.compile(r"^(?:(?:can|could) we |let's )?(?:go back to|back to|return to|revisit|continue with)\s+(.+?)[.?!]*$", re.I),
}
_HISTORICAL = {
    "zh": [(re.compile(r"我(?:之前|原来|以前|过去)住在?(?:哪|什么地方)"), "lives_in")],
    "en": [(re.compile(r"where (?:did i (?:use to )?live|(?:was i living|did i live) before)", re.I), "lives_in")],
}
RESUME_GAP = {
    "zh": "你想回到哪一个问题？可以再说一下主题。",
    "en": "Which earlier question did you want to return to?",
}
MEMORY_UNKNOWN = {
    "zh": "你还没告诉我现在的情况。",
    "en": "You haven't told me what the current answer is.",
}
MEMORY_PAST = {"zh": "你之前说住在{0}。", "en": "You told me you used to live in {0}."}


def normalized(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.casefold())


def card_revision(doc) -> str:
    payload = json.dumps([doc.id, doc.row, doc.answer], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def resume_target(text: str, language: str) -> str | None:
    match = _RESUME[language].fullmatch(text.strip())
    if not match:
        return None
    target = re.sub(r"^(?:(?:我们)?(?:最开始|一开始|开始|刚才|前面|之前|上次)(?:(?:刚)?说(?:过)?|提到)?的?|the (?:first |earlier |original )?(?:topic|question) (?:about|on) |what we (?:first )?(?:said|discussed) about )",
                    "", match[1].strip(), flags=re.I)
    target = re.sub(r"(?:的话题|的问题|那件事|这个问题|好吗|吧)$", "", target)
    return normalized(target)


def historical_kind(text: str, language: str) -> str | None:
    return next((kind for pattern, kind in _HISTORICAL[language] if pattern.search(text)), None)


@dataclass(frozen=True)
class Topic:
    card_id: str
    revision: str
    label: str
    query: str


class TopicHistory:
    def __init__(self):
        self.users: dict[str, list[Topic]] = {}

    def remember(self, user: str, topic: Topic) -> None:
        previous = self.users.pop(user, [])
        self.users[user] = [t for t in previous if t.card_id != topic.card_id][-(_TOPIC_LIMIT - 1):] + [topic]
        if len(self.users) > _USER_LIMIT:
            del self.users[next(iter(self.users))]

    def resolve(self, user: str, target: str, docs: dict):
        matches = []
        if not target:
            return None
        for topic in self.users.get(user, []):
            doc = docs.get(topic.card_id)
            if doc is None or topic.revision != card_revision(doc):
                continue
            labels = [normalized(topic.label), normalized(topic.query)]
            if target in labels:
                matches.append((2, doc))
            elif ((len(target) >= 2 or re.fullmatch(r"[一-鿿]", target))
                  and any(target in label for label in labels)):
                matches.append((1, doc))
        if matches:
            best = max(rank for rank, _ in matches)
            equally_good = [doc for rank, doc in matches if rank == best]
            if len(equally_good) == 1:
                return equally_good[0]
        return None

    def snapshot(self, package: str) -> dict:
        return {"package": package,
                "users": {user: [vars(topic) for topic in topics]
                          for user, topics in self.users.items()}}

    def restore(self, data: object, package: str) -> None:
        if not isinstance(data, dict) or data.get("package") != package:
            return
        users = data.get("users")
        if not isinstance(users, dict):
            return
        for user, topics in list(users.items())[-_USER_LIMIT:]:
            if not isinstance(user, str) or not isinstance(topics, list):
                continue
            for row in topics[-_TOPIC_LIMIT:]:
                fields = ("card_id", "revision", "label", "query")
                if isinstance(row, dict) and all(isinstance(row.get(f), str) for f in fields):
                    self.remember(user, Topic(*(row[f] for f in fields)))
