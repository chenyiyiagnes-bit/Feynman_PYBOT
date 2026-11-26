"""产品默认档：P15 + 专属从句检索优先。

2026-11-25 盲评（110 行、p10 vs p15+specific）配对差值 +1.07，95% CI
[0.73, 1.42]，据此把四档检索开关设为产品默认；客户端显式传值仍然优先。

用法（测试）：
    PYTHONPATH=. .venv/bin/python -B tests/test_p15_product_defaults.py
"""

import os

from plugins.feynman_typesetter import PrintRequest
from bus.main import _product_defaults, _with_product_defaults

FLAGS = ("topic_lock", "retrieve_consequence", "retrieval_gates",
         "specific_pref", "use_topology")


def test_defaults_apply_when_client_silent():
    req = PrintRequest(user_query="线圈为什么会转动")
    out = _with_product_defaults(req)
    assert all(getattr(out, f) is True for f in FLAGS)


def test_explicit_false_is_respected():
    req = PrintRequest(user_query="线圈为什么会转动", specific_pref=False)
    out = _with_product_defaults(req)
    assert out.specific_pref is False
    assert out.topic_lock is True


def test_env_can_disable_each_flag(monkeypatch):
    monkeypatch.setenv("QBOT_DEFAULT_SPECIFIC", "0")
    monkeypatch.setenv("QBOT_DEFAULT_TOPOLOGY", "0")
    d = _product_defaults()
    assert d["specific_pref"] is False
    assert d["use_topology"] is False
    assert d["retrieve_consequence"] is True


def test_non_flag_fields_untouched():
    req = PrintRequest(user_query="曲线为什么弯曲", style_deg=77.0, top_k=7)
    out = _with_product_defaults(req)
    assert out.style_deg == 77.0 and out.top_k == 7
    assert out.user_query == "曲线为什么弯曲"


class _Monkey:
    def __init__(self):
        self._saved = {}

    def setenv(self, k, v):
        self._saved.setdefault(k, os.environ.get(k))
        os.environ[k] = v

    def __del__(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in tests:
        if "monkeypatch" in fn.__code__.co_varnames:
            fn(_Monkey())
        else:
            fn()
    print(f"✅ 产品默认档 {len(tests)} 项回归全部通过")
