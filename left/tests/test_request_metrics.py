"""滚动指标回归：/stats 暴露空白率与延迟，单位逻辑独立可测。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_request_metrics.py
"""

import os

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"

from fastapi.testclient import TestClient

import bus.main as bm
from shared.metrics import RollingMetrics

client = TestClient(bm.app)


def test_rolling_window_trims_and_reports():
    m = RollingMetrics(window=3)
    for ms in (10.0, 20.0, 30.0, 40.0):
        m.record(ms, blank=(ms == 40.0))
    snap = m.snapshot()
    assert snap["total"] == 4 and snap["blank_total"] == 1
    assert snap["recent_window"] == 3
    assert abs(snap["recent_blank_rate"] - 1 / 3) < 1e-3
    assert snap["mean_ms"] is not None


def test_blank_and_empty_states():
    m = RollingMetrics()
    assert m.snapshot()["blank_rate"] is None
    m.record(5.0)
    assert m.snapshot()["blank_rate"] == 0.0


def test_stats_exposes_feynman_metrics_and_sessions():
    bm._feynman_metrics.reset()
    client.post("/feynman/print", json={"style_deg": 60.0})
    client.post("/feynman/paragraph", json={"user_query": "电子为什么会发光"})
    stats = client.get("/stats").json()
    assert stats["feynman"]["total"] == 2
    assert stats["feynman"]["p50_ms"] is not None
    assert 0.0 <= stats["feynman"]["blank_rate"] <= 1.0
    assert stats["sessions"]["active"] >= 0
    assert stats["sessions"]["max"] >= 1


def test_reset_clears_metrics():
    client.post("/feynman/print", json={"style_deg": 60.0})
    out = client.post("/counters/reset").json()
    assert out["feynman"]["total"] == 0
    assert client.get("/stats").json()["feynman"]["total"] == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 滚动指标 {len(tests)} 项回归全部通过")
