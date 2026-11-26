"""段落模式与前缀同一套产品默认档（P15 + 专属从句优先）。

问题：单句 /feynman/print 有 _with_product_defaults，段落 /feynman/paragraph
的 ParagraphRequest 却没有这五个字段，weave_paragraph 构造 PrintRequest 时
一个都不传 → 段落一直走老检索排序，P15 的专属从句在段落里被通用原子压过。

用法：
    PYTHONPATH=. .venv/bin/python -B tests/test_p15_paragraph_defaults.py
"""

import os
from types import SimpleNamespace

os.environ.setdefault("DEEPSEEK_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", "")
os.environ.setdefault("QBOT_CACHE_TTL_SEC", "0")

import plugins.paragraph_weaver as pw
from plugins.paragraph_weaver import ParagraphRequest, weave_paragraph


def _capture_print_request(req: ParagraphRequest):
    """拦下 weave_paragraph 里构造的 PrintRequest，不跑真排版。"""
    captured = {}
    orig = pw.run_feynman_print

    def stub(vault, print_req, **kw):
        captured["req"] = print_req
        return {"paths": [], "seed": None, "ext_hash": "",
                "user_entities": [], "user_verbs": []}

    pw.run_feynman_print = stub
    try:
        weave_paragraph(SimpleNamespace(skeletons=[]), req)
    finally:
        pw.run_feynman_print = orig
    return captured["req"]


def test_flags_reach_print_request():
    req = ParagraphRequest(user_query="线圈为什么会转动", topic_lock=True,
                           retrieve_consequence=True, retrieval_gates=True,
                           specific_pref=True)
    out = _capture_print_request(req)
    assert out.topic_lock is True
    assert out.retrieve_consequence is True
    assert out.retrieval_gates is True
    assert out.specific_pref is True


def test_flags_default_off_without_product_layer():
    """段落自己不做默认档；默认档只由总线补齐，避免两处配置打架。"""
    out = _capture_print_request(ParagraphRequest(user_query="线圈为什么会转动"))
    assert out.specific_pref is False
    assert out.retrieve_consequence is False


def test_endpoint_applies_product_defaults():
    from fastapi.testclient import TestClient
    from bus.main import app

    client = TestClient(app)
    captured = {}
    orig = pw.weave_paragraph

    def fake_weave(vault, req, **kw):
        captured["req"] = req
        return {"paragraph": "", "sentences": [], "complete": False,
                "pool_size": 0, "anchor": {}, "user_entities": [], "user_verbs": []}

    pw.weave_paragraph = fake_weave
    try:
        resp = client.post("/feynman/paragraph", json={"user_query": "线圈为什么会转动"})
        assert resp.status_code == 200, resp.text
        req = captured["req"]
        assert req.specific_pref is True
        assert req.retrieve_consequence is True
        assert req.retrieval_gates is True
        assert req.topic_lock is True

        captured.clear()
        resp = client.post("/feynman/paragraph",
                           json={"user_query": "线圈为什么会转动", "specific_pref": False})
        assert resp.status_code == 200, resp.text
        assert captured["req"].specific_pref is False
        assert captured["req"].retrieve_consequence is True
    finally:
        pw.weave_paragraph = orig


def test_endpoint_env_can_disable():
    from fastapi.testclient import TestClient
    from bus.main import app

    client = TestClient(app)
    captured = {}
    orig = pw.weave_paragraph

    def fake_weave(vault, req, **kw):
        captured["req"] = req
        return {"paragraph": "", "sentences": [], "complete": False,
                "pool_size": 0, "anchor": {}, "user_entities": [], "user_verbs": []}

    saved = os.environ.get("QBOT_DEFAULT_SPECIFIC")
    os.environ["QBOT_DEFAULT_SPECIFIC"] = "0"
    pw.weave_paragraph = fake_weave
    try:
        client.post("/feynman/paragraph", json={"user_query": "线圈为什么会转动"})
        assert captured["req"].specific_pref is False
    finally:
        pw.weave_paragraph = orig
        if saved is None:
            os.environ.pop("QBOT_DEFAULT_SPECIFIC", None)
        else:
            os.environ["QBOT_DEFAULT_SPECIFIC"] = saved


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"✅ 段落产品默认档 {len(tests)} 项回归全部通过")
