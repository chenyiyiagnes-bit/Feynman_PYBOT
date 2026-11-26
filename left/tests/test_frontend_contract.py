"""前端契约回归（P6.1）：大屏读到的每个字段都必须真实存在于后端响应里。

Streamlit 本身不在这里跑；只核对 dashboard.py 与 /feynman/print、/feynman/feedback
的字段契约，防止"前端按文档字段写、后端根本没返回"这类错位。
"""

import os
import re
import tempfile
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")

import plugins.domain_expander as de

DASH = Path("frontend/dashboard.py")

REQUIRED_IN_SOURCE = [
    '"user_query"', '"session_id"', '"operators"', '"style_deg"',
    "/feynman/feedback", '"vote"', "printed_collapse", "top_resonant_line",
    "ext_hash", "entity_weights", 'session"]', "pattern_type",
    # P7：段落面板与融合开关
    "/feynman/paragraph", '"fusion"', '"beam_width"', "sentences", "latency_ms",
    # P8：条件级联开关
    '"conditioned_chain"', "feynman_conditioned",
]


def _setup(tmpdir: Path):
    de.EXT_PATH = tmpdir / "domain_extensions.json"


def test_dashboard_uses_real_fields_only():
    src = DASH.read_text(encoding="utf-8")
    for token in REQUIRED_IN_SOURCE:
        assert token in src, f"dashboard 缺少 {token}"


def test_print_response_carries_every_field_frontend_reads():
    from fastapi.testclient import TestClient
    from bus.main import app

    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        r = client.post("/feynman/print", json={
            "style_deg": 60, "nonce": 0, "top_k": 10,
            "user_query": "股票回购是什么", "session_id": "ui-sess",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        for field in ("printed_collapse", "top_resonant_line", "paths", "seed",
                      "style_band", "purity", "killed", "skeleton",
                      "user_entities", "ext_hash", "entity_weights", "session"):
            assert field in body, field
        assert body["session"]["turns"] >= 1
        assert body["ext_hash"] == "00000000"

        # 前端点赞的 payload 必须被后端接受
        fb = client.post("/feynman/feedback", json={
            "vote": "up",
            "entities": body["user_entities"],
            "action": body["paths"][0]["glyphs"]["action"],
            "skeleton": body["skeleton"]["pattern_type"],
            "session_id": "ui-sess",
        })
        assert fb.status_code == 200, fb.text
        assert fb.json()["ext_hash"] != "00000000"

        # 点完之后同一请求必须能读到新的经验乘子
        again = client.post("/feynman/print", json={
            "style_deg": 60, "nonce": 0, "top_k": 10,
            "user_query": "股票回购是什么", "session_id": "ui-sess",
        }).json()
        assert again["ext_hash"] == fb.json()["ext_hash"]
        assert any(v > 1.0 for v in again["entity_weights"].values())


def test_paragraph_response_carries_every_field_frontend_reads():
    """P7 段落面板读的字段必须真在 /feynman/paragraph 响应里。"""
    from fastapi.testclient import TestClient
    from bus.main import app

    with tempfile.TemporaryDirectory() as d:
        _setup(Path(d))
        client = TestClient(app)
        r = client.post("/feynman/paragraph", json={
            "user_query": "为什么天空是蓝的", "style_deg": 70, "nonce": 1,
            "beam_width": 6,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        for field in ("paragraph", "sentences", "complete", "pool_size",
                      "style_band", "seed", "latency_ms", "ext_hash"):
            assert field in body, field
        assert len(body["sentences"]) == 3
        for s in body["sentences"]:
            for field in ("stage", "kind", "connector", "line"):
                assert field in s, field
        # 前端段落请求里的 fusion 字段必须被接受
        ok = client.post("/feynman/paragraph", json={
            "user_query": "为什么天空是蓝的", "fusion": "boltzmann"}).json()
        assert "paragraph" in ok
        # P8：conditioned_chain 开关两端点都必须接受，且默认关
        cond = client.post("/feynman/paragraph", json={
            "user_query": "为什么天空是蓝的", "conditioned_chain": True}).json()
        assert "paragraph" in cond
        assert cond.get("sentences")
        pr = client.post("/feynman/print", json={
            "user_query": "为什么天空是蓝的", "conditioned_chain": True,
            "top_k": 5}).json()
        assert pr["conditioned"] is True
        pr_off = client.post("/feynman/print", json={
            "user_query": "为什么天空是蓝的", "top_k": 5}).json()
        assert pr_off["conditioned"] is False


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 前端字段契约 {len(tests)} 项回归全部通过")
