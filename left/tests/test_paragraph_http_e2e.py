"""段落接口端到端回归：走真实 HTTP 栈（TestClient），锁住三件此前没人测的事。

1. 同一请求（无 session）两次返回的段落与三句逐字节一致——排版可复现。
2. 带 session_id 连续追问：会话振幅从 1.0 起、单调升、封顶 2.0；
   turn 计数每次 +1；switching 语体后相位朝新方向移动。
3. 不传 session_id 时响应里没有 session 字段。

只读字库；扩展库强制指向临时文件，避免隐式信号后台写生产 ext。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_paragraph_http_e2e.py
"""

import os
import tempfile

TMP = tempfile.mkdtemp()
os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"
os.environ["QBOT_FEYNMAN_EXT"] = os.path.join(TMP, "domain_extensions.json")
# 会话库从空开始，避免进程内其它测试残留
os.environ.pop("QBOT_SESSION_MAX", None)
os.environ.pop("QBOT_SESSION_TTL_SEC", None)

from fastapi.testclient import TestClient

import bus.main as bm
from plugins.feynman_vault import DEATH_MARKERS

client = TestClient(bm.app)

QUERY = "电子为什么会发光"
STYLE = 70.0


def _post(**kw):
    body = {"user_query": QUERY, "style_deg": STYLE, "nonce": 7,
            "beam_width": 5}
    body.update(kw)
    r = client.post("/feynman/paragraph", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _lines(body):
    return [s.get("line") or "" for s in body["sentences"]]


def test_same_request_is_byte_identical():
    a, b = _post(), _post()
    assert a["paragraph"] == b["paragraph"], "同请求两次段落不一致"
    assert _lines(a) == _lines(b), "同请求两次分句不一致"
    assert a["seed"] == b["seed"]
    assert a["ext_hash"] == b["ext_hash"]
    assert len(a["sentences"]) >= 1
    assert "session" not in a, "未传 session_id 不应返回 session"


def test_no_death_content_and_sentences_distinct():
    body = _post()
    text = body["paragraph"]
    for m in DEATH_MARKERS:
        assert m not in text, f"出现死亡相关字粒 {m!r}：{text}"
    lines = [x for x in _lines(body) if x]
    assert len(lines) == len(set(lines)), f"分句重复：{lines}"


def test_session_amplitude_grows_and_caps():
    bm._sessions.clear()
    sid = "e2e-grow"
    amps, turns, phases = [], [], []
    for _ in range(4):
        body = _post(session_id=sid)
        snap = body["session"]
        amps.append(float(snap["amplitude"]))
        turns.append(int(snap["turns"]))
        phases.append(float(snap["phase_deg"]))
    assert turns == [1, 2, 3, 4], turns
    assert amps[0] == 1.0, f"首轮振幅应为 1.0，实际 {amps}"
    assert all(y >= x for x, y in zip(amps, amps[1:])), f"振幅不单调：{amps}"
    assert max(amps) <= 2.0, f"振幅超过封顶：{amps}"
    assert amps[-1] == 2.0, f"同相位追问 4 轮应触顶 2.0，实际 {amps}"
    # 同语体连续追问，相位维持在请求语体附近
    assert all(abs(p - STYLE) <= 15.0 for p in phases), phases
    snap = body["session"]
    assert abs(float(snap["bias_deg"])) <= 15.0
    # 相位偏置把 Δθ 压在 ~65° 内，0.85·cos(65°) ≈ 0.36 > 0.15，单轮换题剪不掉
    assert snap["pruned"] is False
    assert snap["session_id"] == sid


def test_topic_switch_moves_phase_toward_new_style():
    bm._sessions.clear()
    sid = "e2e-switch"
    for _ in range(3):
        _post(session_id=sid)
    before = _post(session_id=sid)["session"]
    assert float(before["amplitude"]) == 2.0
    # 换到低相位语体：相位应朝新方向下移，而不是原地不动
    after = _post(session_id=sid, style_deg=15.0)["session"]
    assert float(after["turns"]) == int(before["turns"]) + 1
    assert float(after["phase_deg"]) < float(before["phase_deg"]), (
        f"换题后相位未移动：{before['phase_deg']} → {after['phase_deg']}")
    assert float(after["phase_deg"]) >= 0.0
    assert abs(float(after["bias_deg"])) <= 15.0


def test_session_isolated_per_id():
    bm._sessions.clear()
    a = _post(session_id="e2e-a")["session"]
    b = _post(session_id="e2e-b")["session"]
    assert a["turns"] == 1 and b["turns"] == 1
    assert a["phase_deg"] == b["phase_deg"] == STYLE


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 段落 HTTP e2e {len(tests)} 项回归全部通过")
