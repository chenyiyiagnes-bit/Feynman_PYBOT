"""滚动请求指标：只数事实（延迟、空白），不做任何解释。

窗口默认 500 条，给 /stats 用；进程重启即归零，不落盘。
"""

import threading
import time
from collections import deque


class RollingMetrics:
    def __init__(self, window: int = 500):
        self.window = max(1, int(window))
        self._lock = threading.Lock()
        self._latencies = deque(maxlen=self.window)
        self._blanks = deque(maxlen=self.window)
        self.total = 0
        self.blank_total = 0

    def record(self, latency_ms: float, blank: bool = False) -> None:
        with self._lock:
            self.total += 1
            self.blank_total += 1 if blank else 0
            self._latencies.append(float(latency_ms))
            self._blanks.append(1 if blank else 0)

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._blanks.clear()
            self.total = 0
            self.blank_total = 0

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            blanks = list(self._blanks)
            total = self.total
            blank_total = self.blank_total
        n = len(lat)
        p50 = round(lat[n // 2], 2) if n else None
        p95 = round(lat[min(n - 1, int(round(0.95 * (n - 1))))], 2) if n else None
        recent_rate = round(sum(blanks) / len(blanks), 3) if blanks else None
        return {
            "total": total,
            "blank_total": blank_total,
            "blank_rate": round(blank_total / total, 3) if total else None,
            "recent_window": n,
            "recent_blank_rate": recent_rate,
            "p50_ms": p50,
            "p95_ms": p95,
            "mean_ms": round(sum(lat) / n, 2) if n else None,
        }


def now_ms() -> float:
    return time.perf_counter() * 1000.0
