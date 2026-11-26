"""结果缓存 + 调用预算回归（全程离线/假 Key，不产生真实网络请求）。"""
import os
import tempfile

os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["QBOT_CACHE_PATH"] = os.path.join(tempfile.mkdtemp(prefix="qbot_cache_"), "cache.json")
os.environ["QBOT_CACHE_TTL_SEC"] = "3600"
# 显式置空而不是 pop：bus.main 的 load_dotenv 在导入期先于插件执行，
# pop 掉的话 .env 里的 QBOT_MAX_LLM_CALLS 又会被填回来。空值 = 不限。
os.environ["QBOT_MAX_LLM_CALLS"] = ""
os.environ["QBOT_MAX_LLM_CALLS_PER_SOLVE"] = ""

from fastapi.testclient import TestClient

from bus.main import app
from shared.solve_cache import solve_cache, normalize_problem
from shared.llm_ledger import ledger, request_budget, BudgetExhausted

client = TestClient(app)

SEALED = "一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？"
AMBIGUOUS = "一架无人机在卡车里飞起来，卡车会变轻吗？"


def test_cache_miss_then_hit():
    solve_cache.clear()
    first = client.post("/solve", json={"problem": SEALED}).json()
    assert first["cache_hit"] is False
    assert first["llm_calls"] == 0  # 离线模式没有可授予的调用

    second = client.post("/solve", json={"problem": SEALED}).json()
    assert second["cache_hit"] is True
    assert second["cache_age_sec"] >= 0
    assert second["collapse_status"] == first["collapse_status"]
    assert second["spectrum"] == first["spectrum"]


def test_cache_normalizes_whitespace_and_punctuation():
    solve_cache.clear()
    client.post("/solve", json={"problem": SEALED})
    variant = "  一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？  "
    assert client.post("/solve", json={"problem": variant}).json()["cache_hit"] is True
    assert normalize_problem(SEALED) == normalize_problem(variant)


def test_force_refresh_bypasses_cache():
    solve_cache.clear()
    client.post("/solve", json={"problem": AMBIGUOUS})
    data = client.post("/solve", json={"problem": AMBIGUOUS, "force_refresh": True}).json()
    assert data["cache_hit"] is False


def test_different_problem_is_a_miss():
    solve_cache.clear()
    client.post("/solve", json={"problem": SEALED})
    assert client.post("/solve", json={"problem": AMBIGUOUS}).json()["cache_hit"] is False


def test_ttl_zero_disables_cache():
    solve_cache.clear()
    client.post("/solve", json={"problem": SEALED})
    old_ttl = solve_cache.ttl_sec
    solve_cache.ttl_sec = 0
    try:
        key = solve_cache.cache_key(SEALED, "offline")
        assert solve_cache.get(key) is None
    finally:
        solve_cache.ttl_sec = old_ttl


def test_request_budget_denies_extra_calls():
    state = request_budget(limit=1)
    with state:
        ledger.grant("deepseek-chat")
        try:
            ledger.grant("deepseek-chat")
            raised = False
        except BudgetExhausted:
            raised = True
    assert raised and state.used == 1


def test_budget_exhaustion_degrades_to_offline_instead_of_failing():
    """假 Key + 单请求预算 0：所有 LLM 调用被闸门拦下，请求仍须正常坍缩。"""
    solve_cache.clear()
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-budget-only"
    os.environ["QBOT_MAX_LLM_CALLS_PER_SOLVE"] = "0"
    try:
        before = ledger.denied
        data = client.post("/solve", json={"problem": SEALED}).json()
        assert data["llm_calls"] == 0
        assert data["cache_hit"] is False
        assert data["collapse_status"] == "COLLAPSED_VALID", data["spectrum"]
        assert data["spectrum"][0]["probability"] >= 60.0
        assert ledger.denied > before  # 闸门确实拦截过
        assert data["llm_budget"]["per_solve_limit"] == 0
    finally:
        os.environ["DEEPSEEK_API_KEY"] = ""
        os.environ.pop("QBOT_MAX_LLM_CALLS_PER_SOLVE", None)


def test_stats_reports_ledger_cache_and_lean():
    stats = client.get("/stats").json()
    assert "llm_ledger" in stats and "cache_entries" in stats and "lean" in stats
    assert stats["lean"]["available"] in (True, False)


if __name__ == "__main__":
    for fn in [
        test_cache_miss_then_hit,
        test_cache_normalizes_whitespace_and_punctuation,
        test_force_refresh_bypasses_cache,
        test_different_problem_is_a_miss,
        test_ttl_zero_disables_cache,
        test_request_budget_denies_extra_calls,
        test_budget_exhaustion_degrades_to_offline_instead_of_failing,
        test_stats_reports_ledger_cache_and_lean,
    ]:
        fn()
    print("✅ 缓存与调用预算 8 项回归全部通过")
