"""LLM 调用记账与预算闸门。

两层限制：
- 进程级硬上限 `QBOT_MAX_LLM_CALLS`（未设置或负数 = 不限）；
- 单次请求预算 `QBOT_MAX_LLM_CALLS_PER_SOLVE`（未设置或负数 = 不限），
  由 `request_budget()` 通过 contextvar 下发，`asyncio.to_thread` 会自动携带。

超预算抛 `BudgetExhausted`；调用方现有的 try/except 会回退到离线仿真内核，
因此预算耗尽不会让请求失败，只会降级并如实报告调用次数。
"""

import contextvars
import os
import threading
from typing import Optional


class BudgetExhausted(RuntimeError):
    """本轮/本进程 LLM 调用次数已达上限。"""


_scope: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "qbot_request_budget", default=None
)


def _env_limit(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


class Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.granted = 0
        self.denied = 0
        self.by_model: dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self.granted = 0
            self.denied = 0
            self.by_model = {}

    def grant(self, model: str) -> None:
        global_limit = _env_limit("QBOT_MAX_LLM_CALLS")
        with self._lock:
            scope = _scope.get()
            if global_limit is not None and self.granted >= global_limit:
                self.denied += 1
                raise BudgetExhausted(
                    f"进程级 LLM 调用预算已用尽（QBOT_MAX_LLM_CALLS={global_limit}）"
                )
            if scope is not None and scope["limit"] >= 0 and scope["used"] >= scope["limit"]:
                self.denied += 1
                raise BudgetExhausted(
                    f"本轮请求 LLM 调用预算已用尽（上限 {scope['limit']} 次）"
                )
            if scope is not None:
                scope["used"] += 1
            self.granted += 1
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "granted": self.granted,
                "denied": self.denied,
                "by_model": dict(self.by_model),
                "global_limit": _env_limit("QBOT_MAX_LLM_CALLS"),
                "per_solve_limit": _env_limit("QBOT_MAX_LLM_CALLS_PER_SOLVE"),
            }


ledger = Ledger()


def remaining_budget() -> Optional[int]:
    """当前请求/进程还可授予的调用次数；None = 不限。"""
    global_limit = _env_limit("QBOT_MAX_LLM_CALLS")
    scope = _scope.get()
    limits = []
    if global_limit is not None:
        limits.append(global_limit - ledger.granted)
    if scope is not None and scope["limit"] >= 0:
        limits.append(scope["limit"] - scope["used"])
    if not limits:
        return None
    return max(0, min(limits))


class request_budget:
    """with 块内统计并（可选）限制本轮请求的 LLM 调用次数。"""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = _env_limit("QBOT_MAX_LLM_CALLS_PER_SOLVE") if limit is None else limit
        self.used = 0
        self._state = None

    def __enter__(self):
        state = {"limit": -1 if self.limit is None else self.limit, "used": 0}
        self._state = state
        self._token = _scope.set(state)
        return self

    def __exit__(self, *exc_info):
        if self._state is not None:
            self.used = self._state["used"]
            _scope.reset(self._token)
        return False
