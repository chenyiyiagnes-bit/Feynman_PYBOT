import os

# 离线模式：强制清空 Key，避免真实调用并保证断言针对内置仿真内核
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient

from bus.main import app

client = TestClient(app)

SEALED_PROBLEM = "一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？"
AMBIGUOUS_PROBLEM = "一架无人机在卡车里飞起来，卡车会变轻吗？"


def test_ambiguity_triggers_second_notebook():
    data = client.post("/solve", json={"problem": AMBIGUOUS_PROBLEM}).json()
    assert data["collapse_status"] == "SECOND_NOTEBOOK_PENDING", data["spectrum"]
    assert data["skeptic_verdict"] is None
    assert data["second_notebook"] is not None
    assert data["second_notebook"]["clarifying_questions"]
    # 两个假设与「条件缺失」三派都应低于 60% 门限
    assert all(s["probability"] < 60.0 for s in data["spectrum"]), data["spectrum"]


def test_sealed_problem_collapses():
    data = client.post("/solve", json={"problem": SEALED_PROBLEM}).json()
    assert data["collapse_status"] == "COLLAPSED_VALID", data["spectrum"]
    assert data["spectrum"][0]["probability"] >= 60.0
    assert data["skeptic_verdict"]


if __name__ == "__main__":
    test_ambiguity_triggers_second_notebook()
    test_sealed_problem_collapses()
    print("✅ 第二笔记本离线回归通过：模糊题不坍缩、密封题坍缩")
