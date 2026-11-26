"""会话库有界 + 进化库并发写不丢更新。

会话库此前是无上限 dict，probe 脚本或 Discord 网关这类长跑客户端
一换 session_id 就多一条，进程只增不减。这里锁死三条：
LRU 上限、空闲 TTL、触达即刷新；进化库的读-改-写在 EXT_LOCK 内，
多线程并发更新不同实体必须全部落盘。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_session_store_bound.py
"""

import os
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ["QBOT_CACHE_TTL_SEC"] = "0"

from fastapi.testclient import TestClient

import bus.main as bm

client = TestClient(bm.app)


def _reset():
    bm._sessions.clear()
    bm._session_last_seen.clear()
    for k in ("QBOT_SESSION_MAX", "QBOT_SESSION_TTL_SEC"):
        os.environ.pop(k, None)


def _post(sid: str):
    r = client.post("/feynman/print",
                    json={"style_deg": 60.0, "session_id": sid})
    assert r.status_code == 200, r.text


def test_lru_cap_evicts_oldest():
    _reset()
    os.environ["QBOT_SESSION_MAX"] = "3"
    for sid in ("s1", "s2", "s3", "s4", "s5"):
        _post(sid)
        assert len(bm._sessions) <= 3
    assert set(bm._sessions) == {"s3", "s4", "s5"}
    # 命中旧会话要刷新次序，不能按创建时间淘汰
    _post("s3")
    _post("s6")
    assert set(bm._sessions) == {"s3", "s5", "s6"}, set(bm._sessions)


def test_idle_ttl_expires_and_touch_refreshes():
    _reset()
    os.environ["QBOT_SESSION_TTL_SEC"] = "0.4"
    _post("hot")
    time.sleep(0.45)
    _post("hot")  # 触达刷新
    _post("new")  # 刚触达过，不该被淘汰
    assert "hot" in bm._sessions
    time.sleep(0.45)
    _post("newer")
    assert "hot" not in bm._sessions


def test_defaults_are_bounded_without_env():
    _reset()
    os.environ["QBOT_SESSION_MAX"] = ""
    os.environ["QBOT_SESSION_TTL_SEC"] = ""
    for i in range(3):
        _post(f"d{i}")
    assert len(bm._sessions) == 3  # 默认值远大于 3，不该误伤


def test_concurrent_evolution_updates_keep_every_edge():
    import plugins.domain_expander as de
    from plugins.evolution_loop import apply_evolution_update

    orig = de.EXT_PATH
    de.EXT_PATH = Path(tempfile.mkdtemp(prefix="ext_conc_")) / "ext.json"
    try:
        ents = [f"实体{i:02d}" for i in range(8)]
        threads = [threading.Thread(
            target=apply_evolution_update,
            args=([e], "旋转", "", 0.3, "concurrency-test")) for e in ents]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        data, _ = de.load_domain_extensions_with_hash()
        for e in ents:
            assert data["edge_scores"].get(f"{e}|旋转") == 1.3, (e, data["edge_scores"])
    finally:
        de.EXT_PATH = orig


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 会话库有界 + 进化并发写 {len(tests)} 项回归全部通过")
