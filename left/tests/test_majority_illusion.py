import asyncio
from shared.protocol import RawWorkerDraft
from evaluator.mapreduce_auditor import mapreduce_evaluate_drafts
from bus.main import compute_damped_collapse_spectrum

EXTREME_PROBLEM = "假设在绝对光滑无摩擦的水平冰面上，一个人站在冰面正中央，手里拿着一块石头，不借助任何外部工具，他能否移动到冰面边缘？"

async def run_stress_test():
    drafts = [
        RawWorkerDraft(bot_id="PyBot-F1", wave_direction="RETARDED_FORWARD", perspective="直觉派A", raw_reasoning="【盲从】冰面无摩擦，无法蹬地获得摩擦力，原地不动。"),
        RawWorkerDraft(bot_id="PyBot-F2", wave_direction="RETARDED_FORWARD", perspective="直觉派B", raw_reasoning="【盲从】没有外部水平合外力，白白制造不出位移。"),
        RawWorkerDraft(bot_id="PyBot-F3", wave_direction="RETARDED_FORWARD", perspective="直觉派C", raw_reasoning="【盲从】零摩擦条件下任何动作都只能原地打滑。"),
        RawWorkerDraft(bot_id="PyBot-F4", wave_direction="RETARDED_FORWARD", perspective="直觉派D", raw_reasoning="【盲从】摆动手臂只会原地晃动，无法到达边缘。"),
        RawWorkerDraft(bot_id="PyBot-B1", wave_direction="ADVANCED_BACKWARD", perspective="逆向边界破坏者", raw_reasoning="【逆向证伪】若零摩擦绝对无法移动成立，宇航员在太空中也永远无法移动！直接撞碎动量守恒——向反方向扔出石头即可反冲滑向边缘！"),
        RawWorkerDraft(bot_id="PyBot-F5", wave_direction="RETARDED_FORWARD", perspective="微观零件拆解师", raw_reasoning="【孤立真相】人把石头向后抛出，石头对人施加向前反冲力，系统总动量为零但人获得速度滑向边缘。")
    ]

    packets = await mapreduce_evaluate_drafts(EXTREME_PROBLEM, drafts)
    spectrum = compute_damped_collapse_spectrum(packets)
    winner = spectrum[0]
    print(f"🏆 极端 4:1 多数派幻觉压力测试胜出：{winner['conclusion']} -> 坍缩概率: {winner['probability']}%")
    # 在线引擎的归并标签是自由措辞（如「抛石反冲可滑向冰面边缘」），离线 mock 是「假设B：……」，两者都必须认定为正确答案。
    correct = any(k in winner["conclusion"] for k in ("假设B", "反冲", "扔", "抛", "石头"))
    assert correct and winner["probability"] >= 95.0, winner["conclusion"]

if __name__ == "__main__":
    asyncio.run(run_stress_test())
