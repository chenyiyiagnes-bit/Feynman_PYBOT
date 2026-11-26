"""/solve 结果缓存：同一问题（规范化后）在 TTL 内直接复用上次响应。

- 缓存键包含运行模式标签（在线模型名 / 离线仿真），避免离线结果污染在线查询；
- 落盘 `QBOT_CACHE_PATH`（默认 data/solve_cache.json），进程重启后仍生效；
- `QBOT_CACHE_TTL_SEC` 默认 3600 秒，设为 0 关闭缓存；
- 条目数上限 200，超出按写入时间淘汰最旧条目。
"""

import copy
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional


def normalize_problem(problem: str) -> str:
    text = problem.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("，", ",").replace("。", ".").replace("？", "?")
    return text.lower()


class SolveCache:
    def __init__(self, storage_path: Optional[str] = None, ttl_sec: Optional[float] = None) -> None:
        self.storage_path = Path(storage_path or os.getenv("QBOT_CACHE_PATH", "data/solve_cache.json"))
        self.ttl_sec = float(os.getenv("QBOT_CACHE_TTL_SEC", "3600")) if ttl_sec is None else ttl_sec
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        if not self.storage_path.is_file():
            return
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._entries = raw
        except Exception:
            self._entries = {}

    def _flush(self) -> None:
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.storage_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.storage_path)
        except Exception:
            # 缓存写盘失败不影响主流程
            pass

    # ---- 键 ----
    @staticmethod
    def cache_key(problem: str, mode_tag: str) -> str:
        return hashlib.sha256(f"{mode_tag}::{normalize_problem(problem)}".encode("utf-8")).hexdigest()

    # ---- 读写 ----
    def get(self, key: str) -> Optional[dict]:
        if self.ttl_sec <= 0:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            age = time.time() - float(entry.get("stored_at", 0.0))
            if age > self.ttl_sec:
                self._entries.pop(key, None)
                self._flush()
                return None
            payload = copy.deepcopy(entry["response"])
            payload["cache_hit"] = True
            payload["cache_age_sec"] = round(age, 1)
            return payload

    def put(self, key: str, problem: str, response: dict) -> None:
        if self.ttl_sec <= 0:
            return
        with self._lock:
            stored = copy.deepcopy(response)
            stored.pop("cache_hit", None)
            stored.pop("cache_age_sec", None)
            self._entries[key] = {
                "problem": problem,
                "stored_at": time.time(),
                "response": stored,
            }
            if len(self._entries) > 200:
                oldest = sorted(self._entries, key=lambda k: self._entries[k].get("stored_at", 0.0))
                for stale in oldest[: len(self._entries) - 200]:
                    self._entries.pop(stale, None)
            self._flush()

    def clear(self) -> int:
        with self._lock:
            n = len(self._entries)
            self._entries = {}
            self._flush()
            return n


solve_cache = SolveCache()
