import os
import asyncio
from typing import List
from shared.protocol import RawWorkerDraft
from shared.deepseek_client import chat_completion, client_available


BIDIRECTIONAL_FLEET = [
    {
        "bot_id": "PyBot-F1",
        "wave_direction": "RETARDED_FORWARD",
        "perspective": "正向滞后波·微观拆解派 (t₀→t_mid)",
        "rule": "【时间顺推】从题目给出的初始条件出发，只用微观粒子碰撞、动量与能量传递向前推演，寻找必然触发的中间物理机制。"
    },
    {
        "bot_id": "PyBot-F2",
        "wave_direction": "RETARDED_FORWARD",
        "perspective": "正向滞后波·跨界同构派 (t₀→t_mid)",
        "rule": "【时间顺推】从初始条件出发，将系统映射为流体力学、电路或信息论模型，顺向推演系统最终会演化到哪种稳态。"
    },
    {
        "bot_id": "PyBot-F3",
        "wave_direction": "RETARDED_FORWARD",
        "perspective": "正向滞后波·直觉常规派 (t₀→t_mid)",
        "rule": "【时间顺推】从初始条件出发，按人类日常经验直觉向前推演最直观的结果。"
    },
    {
        "bot_id": "PyBot-B1",
        "wave_direction": "ADVANCED_BACKWARD",
        "perspective": "逆向超前波·最小作用量侦探 (t_end→t_mid)",
        "rule": "【时间反演】假设最终观测结果已经发生，把时间轴倒着放！逆向追问：为了让终点成立，中间环节必须满足哪个不可绕开的必要条件？"
    },
    {
        "bot_id": "PyBot-B2",
        "wave_direction": "ADVANCED_BACKWARD",
        "perspective": "逆向超前波·极端边界破坏者 (t_end→t_mid)",
        "rule": "【时间反演】假设错误直觉结论在终点成立，将其参数推向无穷大(∞)或绝对零度(0)倒推回起点，检查是否会撞碎守恒律！"
    }
]


async def _single_wave_think(cfg: dict, problem: str) -> RawWorkerDraft:
    if client_available():
        try:
            resp_text = await asyncio.to_thread(
                chat_completion,
                os.getenv("WORKER_MODEL", "deepseek-chat"),
                (
                    f"【波函数演化方向】：{cfg['perspective']}\n"
                    f"【推演铁律】：{cfg['rule']}\n"
                    f"【探索课题】：{problem}\n"
                    "请输出150字以内的纯逻辑推演，明确指出中间核心枢纽机制，不要打分。"
                ),
            )
            return RawWorkerDraft(
                bot_id=cfg["bot_id"],
                wave_direction=cfg["wave_direction"],
                perspective=cfg["perspective"],
                raw_reasoning=resp_text
            )
        except Exception:
            # API 出错或预算耗尽：不产出残缺草稿，直接跌落内置仿真内核
            pass

    await asyncio.sleep(0.1)
    domain = offline_domain(problem)

    # 针对刻意缺失封闭条件的模糊问题，模拟波函数分裂（触发《费曼第二笔记本》）
    if domain == "sealed_drone" and ("密封" not in problem and "敞篷" not in problem):
        ambiguous_map = {
            "PyBot-F1": "若卡车是密封车厢，螺旋桨向下气流撞击车厢底板，地磅读数不变。",
            "PyBot-F2": "若卡车是敞篷无顶结构，向下气流逸散到车外大地，地磅读数减少2公斤。",
            "PyBot-F3": "直觉上离开地板就不再压秤，读数变轻。",
            "PyBot-B1": "逆推发现：最终读数取决于控制体边界是否封闭（题目未说明车厢是否密封）。",
            "PyBot-B2": "边界条件缺失：无法在开放与封闭两种截然相反的物理边界间做单一判定。"
        }
        return RawWorkerDraft(
            bot_id=cfg["bot_id"],
            wave_direction=cfg["wave_direction"],
            perspective=cfg["perspective"],
            raw_reasoning=ambiguous_map[cfg["bot_id"]]
        )

    if domain == "sealed_drone":
        mock_responses = {
            "PyBot-F1": "【顺推】螺旋桨向下加速空气分子，在密封车厢内，向下运动的空气分子必然撞击车厢底板并传递全部动量。",
            "PyBot-F2": "【顺推】将密封车厢类比为密闭水箱，无人机类比悬浮潜艇，流体静压差将重力完整传导至箱体底部。",
            "PyBot-F3": "【顺推】无人机既然已经离开地板悬在半空，就没有接触秤面，直觉上地磅读数应该减少2公斤。",
            "PyBot-B1": "【逆推】若要使无人机静止悬停，空气必须施加向上升力；根据牛顿第三定律，无人机必然对封闭箱内空气施加等大向下的反作用力，由底板承重。",
            "PyBot-B2": "【逆推证伪】若‘离地即减重’成立，在密封货箱内悬停100吨无人机即可实现反重力悬浮，直接撞碎质心守恒定律！"
        }
    elif domain == "maxwell_demon":
        mock_responses = {
            "PyBot-F1": "【顺推】小妖测量分子速度并切换开关状态，微观上必须通过光子散射与物理触发器翻转，必然伴随不可逆能量耗散。",
            "PyBot-F2": "【顺推】把小妖大脑映射为有限容量的存储器，根据兰道尔原理，擦除旧比特信息必然向环境释放 kT*ln2 热量。",
            "PyBot-F3": "【顺推】只要把小门做得极其轻巧且无摩擦，小妖只放行快分子就能白白制造出温差用于发电。",
            "PyBot-B1": "【逆推】要使热力学第二定律在全宇宙尺度不被破坏，小妖在周期性循环工作后必须恢复初始空白状态——‘清空记忆’是不可绕开的必要熵增代价。",
            "PyBot-B2": "【逆推证伪】若‘无摩擦小门可零能耗制造温差’成立，将无限个小妖串联即可建成第二类永动机，彻底撞碎热力学公理！"
        }
    else:
        # 离线内核没有覆盖的题：不编答案，明确标注无法判定
        mock_responses = {
            bid: f"【离线内核未覆盖】该问题不在内置物理仿真域内，本路径无法给出可靠推演，需接入模型或补充边界条件后重试。"
            for bid in ("PyBot-F1", "PyBot-F2", "PyBot-F3", "PyBot-B1", "PyBot-B2")
        }

    return RawWorkerDraft(
        bot_id=cfg["bot_id"],
        wave_direction=cfg["wave_direction"],
        perspective=cfg["perspective"],
        raw_reasoning=mock_responses[cfg["bot_id"]]
    )


OFFLINE_DOMAINS = {
    "sealed_drone": ("无人机", "卡车", "地磅", "车厢", "敞篷"),
    "maxwell_demon": ("妖", "温差", "热力学第二定律", "分子", "熵", "绝热"),
}


def offline_domain(problem: str) -> str | None:
    """离线仿真内核覆盖的题目域；None 表示未覆盖，调用方应老实反问而不是编答案。"""
    for name, keywords in OFFLINE_DOMAINS.items():
        if any(k in problem for k in keywords):
            return name
    return None


async def run_parallel_workers(problem: str) -> List[RawWorkerDraft]:
    tasks = [_single_wave_think(cfg, problem) for cfg in BIDIRECTIONAL_FLEET]
    return await asyncio.gather(*tasks)
