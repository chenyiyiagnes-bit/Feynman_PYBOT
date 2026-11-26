"""用户 overlay 字库：笔记里的概念 → 主库既有字粒的别名映射。

铁律不变：打印出来的每一个字粒都必须可溯源到费曼语料。overlay 不新增字粒，
只把用户笔记里的词（"集群""心跳""领导者"）映射到主库某个槽位里已经存在的
字粒（"振动""温度""电子"……）。检索、打分、渲染全程只用 target 原文；
alias 只出现在用户问句里，作为输入被替换掉。

绑定与失效：
- overlay 单独一个文件，不属于主库，不改主库文件、不改 bank_id；
- 文件里记编译时主库的 `bank_id`，加载时与当前字库核对，不符整份拒绝；
- target 必须已存在于所声明槽位，否则整份拒绝（不许借 overlay 夹带新词）。

用法：
    from plugins.user_overlay import load_overlay_or_none
    ov = load_overlay_or_none(vault)        # QBOT_USER_OVERLAY 未设 → None
    mapped, hits = ov.map_query(req.user_query)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from plugins.feynman_typesetter import SLOTS

OVERLAY_ENV = "QBOT_USER_OVERLAY"
OVERLAY_VERSION = 1
PROVENANCE = "user_note"
_MAX_ALIAS = 40
_MIN_ALIAS = 2


class OverlayError(ValueError):
    """overlay 文件不合规：路径、哈希绑定、词语出处或结构有问题。"""


@dataclass(frozen=True)
class OverlayEntry:
    alias: str
    target: str
    slot: str
    note_ref: str = ""
    provenance: str = PROVENANCE

    def as_dict(self) -> dict:
        return {"alias": self.alias, "target": self.target, "slot": self.slot,
                "note_ref": self.note_ref, "provenance": self.provenance}


@dataclass
class UserOverlay:
    path: str
    overlay_id: str
    bank_id: str
    entries: Tuple[OverlayEntry, ...]
    _by_alias: Dict[str, OverlayEntry] = field(default_factory=dict,
                                               repr=False)
    _pattern: Optional[re.Pattern] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._by_alias = {e.alias: e for e in self.entries}
        if self._by_alias:
            # 单遍替换：最长别名优先，替换结果不会被再次扫描，
            # 避免 alias→X 之后 X 又被另一条 alias 改写。
            aliases = sorted(self._by_alias, key=lambda a: (-len(a), a))
            self._pattern = re.compile(
                "|".join(re.escape(a) for a in aliases))

    def map_query(self, query: str) -> Tuple[str, List[dict]]:
        """把问句里的别名换成 target；返回 (新问句, 命中列表)。"""
        q = query or ""
        if not q or self._pattern is None:
            return q, []
        hits: List[dict] = []
        seen: set = set()

        def _sub(m: re.Match) -> str:
            entry = self._by_alias[m.group(0)]
            if entry.alias not in seen:
                seen.add(entry.alias)
                hits.append(entry.as_dict())
            return entry.target

        return self._pattern.sub(_sub, q), hits

    def info(self) -> dict:
        return {"overlay_id": self.overlay_id, "bank_id": self.bank_id,
                "path": self.path, "entries": len(self.entries)}


def validate_overlay(data, vault) -> List[str]:
    """逐条校验 overlay 数据，返回错误列表（空列表 = 可用）。"""
    errors: List[str] = []
    if not isinstance(data, dict):
        return ["overlay 根节点须为对象"]
    if data.get("version") != OVERLAY_VERSION:
        errors.append(f"version 须为 {OVERLAY_VERSION}")
    bank_id = data.get("bank_id")
    if bank_id != vault.bank_id:
        errors.append(f"bank_id 不符：文件 {bank_id!r} ≠ 当前字库 {vault.bank_id}")
    if not str(data.get("overlay_id") or "").strip():
        errors.append("缺少 overlay_id")
    entries = data.get("entries")
    if not isinstance(entries, list):
        return errors + ["entries 须为列表"]

    words = {s: {g.get("word") for g in vault.by_slot.get(s, [])}
             for s in SLOTS}
    seen: Dict[str, int] = {}
    for i, e in enumerate(entries):
        where = f"entries[{i}]"
        if not isinstance(e, dict):
            errors.append(f"{where} 须为对象")
            continue
        alias = e.get("alias")
        target = e.get("target")
        slot = e.get("slot")
        if not isinstance(alias, str) or not alias.strip():
            errors.append(f"{where} 缺少 alias")
            continue
        alias = alias.strip()
        if not (_MIN_ALIAS <= len(alias) <= _MAX_ALIAS):
            errors.append(f"{where} alias 长度须在 {_MIN_ALIAS}–{_MAX_ALIAS}：{alias!r}")
        if alias in seen:
            errors.append(f"{where} alias 重复：{alias!r}（首次出现于 entries[{seen[alias]}]）")
        else:
            seen[alias] = i
        if slot not in SLOTS:
            errors.append(f"{where} slot 须为 {SLOTS} 之一：{slot!r}")
            continue
        if not isinstance(target, str) or not target.strip():
            errors.append(f"{where} 缺少 target")
            continue
        target = target.strip()
        if target not in words[slot]:
            errors.append(
                f"{where} target {target!r} 不在字库 {slot} 槽位——"
                "overlay 只能映射既有字粒，不能新增")
        if target == alias:
            errors.append(f"{where} alias 与 target 相同，映射没有意义")
        prov = e.get("provenance", PROVENANCE)
        if prov != PROVENANCE:
            errors.append(f"{where} provenance 须为 {PROVENANCE!r}：{prov!r}")
        ref = e.get("note_ref", "")
        if not isinstance(ref, str):
            errors.append(f"{where} note_ref 须为字符串")
    return errors


def load_overlay(path: Path, vault) -> UserOverlay:
    """读取并校验一份 overlay；不合规抛 OverlayError。"""
    path = Path(path)
    if not path.is_file():
        raise OverlayError(f"overlay 文件不存在：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OverlayError(f"overlay JSON 解析失败：{path}：{exc}") from exc
    errors = validate_overlay(data, vault)
    if errors:
        raise OverlayError("overlay 校验失败：" + "；".join(errors))
    entries = tuple(
        OverlayEntry(alias=e["alias"].strip(), target=e["target"].strip(),
                     slot=e["slot"], note_ref=str(e.get("note_ref") or ""))
        for e in data["entries"])
    return UserOverlay(path=str(path), overlay_id=str(data["overlay_id"]),
                       bank_id=str(data["bank_id"]), entries=entries)


def overlay_from_env(vault) -> Optional[UserOverlay]:
    """QBOT_USER_OVERLAY 指向的文件存在就加载，未设置返回 None。"""
    ref = os.getenv(OVERLAY_ENV, "").strip()
    if not ref:
        return None
    return load_overlay(Path(ref), vault)


def load_overlay_or_none(vault) -> Optional[UserOverlay]:
    return overlay_from_env(vault)


def map_query(query: str, overlay: Optional[UserOverlay]
              ) -> Tuple[str, List[dict]]:
    if overlay is None:
        return query or "", []
    return overlay.map_query(query)
