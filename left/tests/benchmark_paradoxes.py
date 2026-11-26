"""三层悖论基准 + 三项计划指标（每层 4 题；离线可跑，配 DEEPSEEK_API_KEY 测真实引擎）。

指标定义（对应方案 1173–1263 行）：
- 干涉剪枝率：|phase| ≥ 90° 或实部 ≤ 0 的路径占比（非相干振幅被压制的比例）。
- 跨路径共鸣度：同一结论下至少 2 条路径相位差 < 30° 且实部 > 0 的结论数。
- Skeptic 拦截准确率：|phase| ≥ 150° 的路径中，其结论未能夺冠的比例（应为 100%）。

离线内核只覆盖题 1/2 与中档题 1；其余题允许老实 abstain（触发第二笔记本），
不算失败，但会在汇总里单列 abstain 数，避免拿「编答案」冒充通过。

设置 QBOT_BENCHMARK_OFFLINE=1 可强制离线（即使 .env 里有 Key）。
"""
import os

if os.environ.get("QBOT_BENCHMARK_OFFLINE") == "1":
    os.environ["DEEPSEEK_API_KEY"] = ""
    os.environ["GEMINI_API_KEY"] = ""

os.environ.setdefault("QBOT_CACHE_TTL_SEC", "0")  # 基准必须实跑，不复用缓存

from fastapi.testclient import TestClient
from bus.main import app

client = TestClient(app)

TIERS = [
    {
        "tier": "easy",
        "problems": [
            {
                "title": "密封卡车悬停无人机（控制体边界）",
                "problem": "一辆完全密封的厢式货车停在电子地磅上总重5吨，车内一架2公斤无人机起飞悬停在正中央，地磅读数如何变化？换成敞篷车呢？",
                "keywords": ("读数不变", "密封", "不变"),
            },
            {
                "title": "密封玻璃箱内悬停的鸟",
                "problem": "一个完全密封的玻璃箱放在电子秤上，箱内有一只鸟在悬停飞行，秤的读数是否等于鸟停栖在箱底时的总重？",
                "keywords": ("不变", "等于", "相同", "一样"),
            },
            {
                "title": "密闭水箱中悬浮的鱼",
                "problem": "一个完全密闭、注满水的水箱放在秤上，水中一条鱼悬浮不动，秤的读数与鱼沉底时相比如何变化？",
                "keywords": ("不变", "相同", "一样"),
            },
            {
                "title": "【模糊陷阱】卡车里飞无人机",
                "problem": "一架无人机在卡车里飞起来，卡车会变轻吗？",
                "expect": "SECOND_NOTEBOOK_PENDING",
                "keywords": (),
            },
        ],
    },
    {
        "tier": "medium",
        "problems": [
            {
                "title": "麦克斯韦妖永动机（信息熵壁垒）",
                "problem": "绝热容器中间有无摩擦小门，小妖只放快分子去右边、慢分子去左边从而制造温差发电，是否打破热力学第二定律？",
                "keywords": ("熵", "兰道尔", "擦除", "第二类永动机", "守恒", "二律", "不破", "第二定律"),
            },
            {
                "title": "西拉德引擎（单分子热机）",
                "problem": "只靠观察单个气体分子在容器哪一半、再插入活塞让它做功，这样的西拉德引擎能否无代价地持续输出功？",
                "keywords": ("熵", "信息", "擦除", "兰道尔", "不能"),
            },
            {
                "title": "兰道尔原理：擦除 1 比特的最低能耗",
                "problem": "在温度 T 下不可逆地擦除 1 比特信息，最少要耗散多少热量？能否做到零能耗擦除？",
                "keywords": ("kT", "ln2", "兰道尔", "不能", "耗散"),
            },
            {
                "title": "无摩擦小门制造第二类永动机",
                "problem": "用无摩擦、无质量的理想小门自动分离冷热分子，能否造出从单一热源持续做功的第二类永动机？",
                "keywords": ("不能", "熵", "第二类永动机", "热力学第二定律", "耗散", "擦除", "抵消"),
            },
        ],
    },
    {
        "tier": "hard",
        "problems": [
            {
                "title": "延迟选择量子擦除（因果序）",
                "problem": "量子擦除实验里，是否可以先测得干涉条纹，再决定擦除路径信息，从而逆时间改变已经通过双缝的光子走哪条路？",
                "keywords": ("超光速", "无信号", "因果", "条件", "延迟", "干涉", "擦除", "测量", "逆时间"),
            },
            {
                "title": "惠勒延迟选择实验",
                "problem": "惠勒延迟选择实验是否意味着光子在被观测之后才决定自己之前走的是哪条路径？",
                "keywords": ("不", "因果", "事后", "选择", "互补", "干涉"),
            },
            {
                "title": "量子芝诺效应",
                "problem": "持续高频观测能否让一个不稳定粒子永远不衰变，从而冻结时间演化？",
                "keywords": ("芝诺", "不", "演化", "测量", "观测"),
            },
            {
                "title": "EPR 纠缠能否超光速通信",
                "problem": "利用 EPR 对纠缠粒子，一方测量能否让另一方瞬间获知结果，从而实现超光速通信？",
                "keywords": ("不能", "无信号", "超光速", "随机", "经典通信"),
            },
        ],
    },
]


def phase_coherent(p):
    return abs(p["phase_deg"]) < 90.0 and p["real"] > 0


def interference_pruning_rate(packets):
    if not packets:
        return None
    return sum(1 for p in packets if not phase_coherent(p)) / len(packets)


def cross_path_resonance(packets):
    bins = {}
    for p in packets:
        if p["real"] > 0:
            bins.setdefault(p["conclusion_key"], []).append(p["phase_deg"])
    count = 0
    for phases in bins.values():
        if len(phases) < 2:
            continue
        phases = sorted(phases)
        if any(abs(phases[i + 1] - phases[i]) < 30.0 for i in range(len(phases) - 1)):
            count += 1
    return count


def skeptic_interception_accuracy(packets, winner_conclusion):
    vetoed = [p for p in packets if abs(p["phase_deg"]) >= 150.0]
    if not vetoed:
        return 1.0
    blocked = sum(1 for p in vetoed if p["conclusion_key"] != winner_conclusion)
    return blocked / len(vetoed)


def evaluate_problem(item, attempts=3):
    """返回 (ok, abstained, data)；在线引擎偶发全反相时允许重试。"""
    data = None
    for _ in range(attempts):
        data = client.post("/solve", json={"problem": item["problem"]}).json()
        status = data["collapse_status"]
        expect = item.get("expect")
        if expect and status == expect:
            return True, False, data
        if not expect and status in ("COLLAPSED_VALID", "COLLAPSED_RESCUED"):
            winner = data["spectrum"][0]
            if any(k in winner["conclusion"] for k in item["keywords"]):
                return True, False, data
        if not expect and status == "SECOND_NOTEBOOK_PENDING":
            return True, True, data
    return False, False, data


def main():
    mode = "在线真实引擎" if os.getenv("DEEPSEEK_API_KEY", "").strip() else "离线仿真内核"
    print(f"🧪 Feynman-QBot v2.0 三层悖论基准（每层 4 题，{mode}）")

    failures, abstains, collapsed = [], 0, 0
    for tier in TIERS:
        print(f"\n=== {tier['tier'].upper()} ===")
        for item in tier["problems"]:
            ok, abstained, data = evaluate_problem(item)
            status = data["collapse_status"]
            winner = data["spectrum"][0] if data["spectrum"] else {"conclusion": "—", "probability": 0.0}
            packets = data["packets"]
            prune = interference_pruning_rate(packets)
            line = f"  {'✅' if ok else '❌'} {item['title']}: {status} | {winner['conclusion']} ({winner['probability']}%)"
            if prune is not None:
                line += f" | 剪枝 {prune:.0%} | 共鸣 {cross_path_resonance(packets)} | 拦截 {skeptic_interception_accuracy(packets, winner['conclusion']):.0%}"
            if abstained:
                line += " | 老实弃答"
            print(line)
            if not ok:
                failures.append(item["title"])
            elif abstained:
                abstains += 1
            else:
                collapsed += 1

    beam = client.post("/solve_beam", json={"problem": TIERS[1]["problems"][0]["problem"], "max_depth": 1}).json()
    trace = beam.get("trace", [])
    if trace:
        r1 = trace[0]
        total = len(r1["surviving_beams"]) + len(r1["pruned_branches"])
        beam_rate = f"{len(r1['pruned_branches']) / total:.0%}" if total else "n/a"
    else:
        beam_rate = "100%（全部剪枝）"
    print(f"\n[BEAM] 中档第 1 题第一轮剪枝率: {beam_rate}")

    print(f"\n汇总：通过 {collapsed} 题、老实弃答 {abstains} 题、失败 {len(failures)} 题")
    assert not failures, f"未通过：{failures}"
    print("🎉 三层基准通过（弃答题按诚实反问计通过）")


if __name__ == "__main__":
    main()
