"""领域扩展库：与费曼原典物理隔离的可写增量层。

`data/feynman_style_vault.json` 永久只读；爬虫/反馈学到的东西只写这里。
排版引擎读的时候做内存合并，删掉本文件即恢复出厂纯正状态。

三条契约：
- `ext_hash` 是规范化内容的 8 位指纹，覆盖原型、共现边、经验分与骨架分；
  时间戳不参与哈希，否则每次写入都会漂。
- 空库（或显式关闭）固定返回 "00000000"，种子链退化为旧格式，20 条基线零差异。
- 写入先写临时文件再 `os.replace`，并发读永远看不到半截 JSON。
"""

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Dict, Tuple

EXT_PATH = Path(os.getenv("QBOT_FEYNMAN_EXT", "data/domain_extensions.json"))
EMPTY_EXT_HASH = "00000000"

# 读-改-写与原子替换共用一把进程内锁：后台任务之间不丢更新
EXT_LOCK = threading.RLock()

_EMPTY = {"entity_prototypes": {}, "cooccurrence_edges": []}
_HASHED_FIELDS = ("entity_prototypes", "cooccurrence_edges", "edge_scores", "skeleton_scores")


def empty_extensions() -> dict:
    return {"entity_prototypes": {}, "cooccurrence_edges": []}


def compute_canonical_ext_hash(ext_data: Dict) -> str:
    """规范化内容 → 8 位 SHA-256。键序、边序、重复边都不影响结果。"""
    protos = ext_data.get("entity_prototypes") or {}
    edges = ext_data.get("cooccurrence_edges") or []
    scores = ext_data.get("edge_scores") or {}
    skeletons = ext_data.get("skeleton_scores") or {}
    if not (protos or edges or scores or skeletons):
        return EMPTY_EXT_HASH
    canonical = {
        "entity_prototypes": dict(sorted(protos.items())),
        "cooccurrence_edges": sorted({tuple(e) for e in edges if len(e) == 2}),
        "edge_scores": dict(sorted(scores.items())),
        "skeleton_scores": dict(sorted(skeletons.items())),
    }
    raw = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def load_domain_extensions_with_hash(enable_extensions: bool = True) -> Tuple[Dict, str]:
    """返回 (ext_data, ext_hash)。文件缺失/损坏/关闭扩展时返回空库与固定指纹。"""
    if not enable_extensions or not EXT_PATH.exists():
        return empty_extensions(), EMPTY_EXT_HASH
    try:
        data = json.loads(EXT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return empty_extensions(), EMPTY_EXT_HASH
    if not isinstance(data, dict):
        return empty_extensions(), EMPTY_EXT_HASH
    return data, compute_canonical_ext_hash(data)


def atomic_save_extensions(ext_data: Dict) -> str:
    """原子写入扩展库并返回新指纹。调用方须持有 EXT_LOCK（或用 update_extensions）。"""
    ext_data = {k: v for k, v in ext_data.items() if k != "ext_hash"}
    new_hash = compute_canonical_ext_hash(ext_data)
    ext_data["ext_hash"] = new_hash
    EXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(EXT_PATH.parent), prefix=".ext_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ext_data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, EXT_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return new_hash


def update_extensions(mutator) -> str:
    """在锁内完成一次读-改-写：mutator(ext_data) -> None，返回新指纹。

    后台任务并发更新同一文件时，串行执行，不丢更新。
    """
    with EXT_LOCK:
        data, _ = load_domain_extensions_with_hash(enable_extensions=True)
        mutator(data)
        return atomic_save_extensions(data)


def reset_extensions() -> str:
    """一键恢复出厂：删除扩展库文件。"""
    with EXT_LOCK:
        if EXT_PATH.exists():
            EXT_PATH.unlink()
    return EMPTY_EXT_HASH
