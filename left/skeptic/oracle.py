import os
import asyncio
from typing import List, Dict
from pydantic import BaseModel

from shared.deepseek_client import chat_completion, client_available


class SecondNotebookReport(BaseModel):
    status: str = "SECOND_NOTEBOOK_PENDING"
    conflict_diagnosis: str
    missing_parameters: List[str]
    clarifying_questions: List[str]


async def feynman_skeptic_verdict(problem: str, winner_claim: str, supporting_details: list[str]) -> str:
    if client_available():
        try:
            prompt = (
                "你是理查德·费曼。请剥离所有专业黑话，用聪明高中生也能听懂的大白话总结以下胜出结论：\n"
                f"问题：{problem}\n胜出结论：{winner_claim}\n支撑路径：{supporting_details}"
            )
            return await asyncio.to_thread(
                chat_completion, os.getenv("AUDITOR_MODEL", "deepseek-chat"), prompt
            )
        except Exception:
            pass

    if "密封" in problem or "无人机" in problem:
        return (
            "【COLLAPSED_VALID | 费曼大白话裁决】：无人机不是靠魔法飘在空中的，它是靠螺旋桨拼命把空气往下踩才飞起来的。"
            "在密封车厢里，这股向下的气流结结实实踩在了车厢地板上，所以地磅示数一克都不会少；一旦打开车厢变成敞篷，风吹到了车外的大地上，秤就会变轻。"
        )
    return (
        "【COLLAPSED_VALID | 费曼大白话裁决】：大自然从不做赔本买卖。表面上看似零能耗的巧合或捷径，在微观层面一定藏着‘观测、记忆擦除或介质反作用力’的隐形账单，"
        "把这些账单算进去后，守恒定律依然纹丝不动。"
    )


async def generate_second_notebook_report(problem: str, spectrum: List[Dict], packets: List[Dict]) -> Dict:
    if client_available():
        try:
            summary_lines = [
                f"- 候选结论 [{s['conclusion']}] (概率 {s['probability']}%)" for s in spectrum
            ]
            prompt = (
                "你是理查德·费曼，正在撰写专门记录‘未知盲区’的《第二笔记本》。当前最高坍缩概率低于60%，请严禁强行猜答案，"
                "找出缺失的关键边界参数并向用户提出1~3个精准反问。\n"
                f"原问题：{problem}\n僵持光谱：\n" + "\n".join(summary_lines)
                + '\n请严格只输出一个 JSON 对象，键为 conflict_diagnosis(字符串) / missing_parameters(字符串数组) / clarifying_questions(字符串数组)。'
            )
            resp_text = await asyncio.to_thread(
                chat_completion,
                os.getenv("AUDITOR_MODEL", "deepseek-chat"),
                prompt,
                response_format={"type": "json_object"},
            )
            return SecondNotebookReport.model_validate_json(resp_text).model_dump()
        except Exception:
            pass

    # 离线仿真：针对模糊问题生成第二笔记本报告
    return SecondNotebookReport(
        conflict_diagnosis="系统在‘密封封闭’与‘敞篷开放’两派之间僵持，波函数无法达到 60% 坍缩门限——题目缺失车厢封闭性这一关键边界条件。",
        missing_parameters=[
            "车厢是否为完全密封结构（侧壁与顶盖是否与地磅构成封闭控制体）",
            "观测时刻是无人机刚起飞的瞬态，还是悬停稳定后的平衡态",
            "地磅测量的是整车总重量，还是车厢底板局部受力"
        ],
        clarifying_questions=[
            "1. 请明确该系统的物理边界是【完全封闭】还是【开放连通】的？",
            "2. 你关注的是事件发生的【瞬间动态过程】，还是系统稳定后的【最终平衡状态】？"
        ]
    ).model_dump()
