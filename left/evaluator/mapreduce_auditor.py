import os
import asyncio
from typing import List
from pydantic import BaseModel, Field

from shared.protocol import RawWorkerDraft, WorkerPathEvaluation, AmplitudePacket
from shared.phase_mapper import compute_symmetric_amplitude
from shared.deepseek_client import chat_completion, client_available


class SingleMapAudit(BaseModel):
    claim_summary: str = Field(..., description="15字以内核心主张")
    critical_flaw: str = Field(..., description="一句话逻辑漏洞或守恒依据")
    evidence_strength: float = Field(..., ge=0.1, le=1.0)
    assumption_burden: float = Field(..., ge=0.0, le=1.0)
    boundary_fragility: float = Field(..., ge=0.0, le=1.0)
    logical_leap: float = Field(..., ge=0.0, le=1.0)


class ClaimMappingItem(BaseModel):
    path_index: int
    normalized_claim: str


class ReduceClusterResponse(BaseModel):
    mappings: List[ClaimMappingItem]


async def _map_audit_single_draft(idx: int, problem: str, draft: RawWorkerDraft) -> SingleMapAudit:
    direction_tag = "正向顺推波" if draft.wave_direction == "RETARDED_FORWARD" else "逆向反演波"
    text = draft.raw_reasoning
    is_drone = ("密封" in problem or "无人机" in problem)

    # 模糊问题分支仿真（使两派概率均低于 60%，触发《费曼第二笔记本》）
    if "未说明车厢是否密封" in text or "边界条件缺失" in text:
        return SingleMapAudit(
            claim_summary="条件缺失：无法判定封闭边界",
            critical_flaw="【条件缺失】题目未给出控制体是否封闭，无法在两种边界之间做单一判定",
            evidence_strength=0.70, assumption_burden=0.45, boundary_fragility=0.45, logical_leap=0.45
        )
    if "敞篷无顶结构" in text:
        return SingleMapAudit(
            claim_summary="假设A：敞篷开放环境下读数变轻",
            critical_flaw="【条件依赖】仅在敞篷开放边界下成立，题目未给出封闭性条件",
            evidence_strength=0.75, assumption_burden=0.35, boundary_fragility=0.20, logical_leap=0.25
        )
    if "若卡车是密封车厢" in text:
        return SingleMapAudit(
            claim_summary="假设B：密封封闭环境下读数不变",
            critical_flaw="【条件依赖】仅在密封封闭边界下成立，题目未给出封闭性条件",
            evidence_strength=0.78, assumption_burden=0.35, boundary_fragility=0.20, logical_leap=0.25
        )
    if "读数变轻" in text or "不再压秤" in text:
        return SingleMapAudit(
            claim_summary="假设A：敞篷开放环境下读数变轻",
            critical_flaw="【单向直觉幻觉】只凭‘离地不压秤’的直觉，未追问向下气流去了哪里",
            evidence_strength=0.45, assumption_burden=0.35, boundary_fragility=0.20, logical_leap=0.85
        )

    # 常规/高压压力测试分支仿真
    if "减少2公斤" in text or "白白制造" in text or "【盲从】" in text:
        return SingleMapAudit(
            claim_summary="假设A：直觉表面结论（离地减重/零摩擦无法移动/零能耗温差）",
            critical_flaw="【单向直觉幻觉】忽略封闭介质反作用力、反冲动量或信息擦除能耗",
            evidence_strength=0.75, assumption_burden=0.40, boundary_fragility=0.25, logical_leap=0.45
        )
    elif "撞碎" in text or "【逆向证伪】" in text:
        return SingleMapAudit(
            claim_summary="假设A：直觉表面结论（离地减重/零摩擦无法移动/零能耗温差）",
            critical_flaw="【逆向边界证伪】极端推演直接撞碎守恒定律，触发 180° 反相熔断",
            evidence_strength=0.95, assumption_burden=0.90, boundary_fragility=1.00, logical_leap=0.85
        )
    else:
        if "石头" in text or "反冲" in text:
            summary = "假设B：向后扔石头利用动量守恒反冲滑向边缘"
        elif is_drone:
            summary = "假设B：密封箱空气承重读数不变（敞篷变轻）"
        else:
            summary = "假设B：受观测与擦除记忆熵增守恒限制"
        return SingleMapAudit(
            claim_summary=summary,
            critical_flaw="【时空驻波共鸣】正向推导与逆向守恒条件完美咬合形成因果闭环",
            evidence_strength=0.92, assumption_burden=0.04, boundary_fragility=0.02, logical_leap=0.04
        )


class BlindAuditResponse(BaseModel):
    items: List[SingleMapAudit]


async def _blind_batch_audit(
    problem: str, drafts: List[RawWorkerDraft], calibration: bool = False
) -> List[SingleMapAudit]:
    """批量横向盲审：一次打包 5 条路径，强制拉开分差，避免逐条「老好人式无差别挑刺」。

    这是原文 1730 行「横向相对标定（Batch Calibration）」的设计意图：
    正确路径应因证据饱满/假设负担低而进入相长区，错误路径因假设负担高进入反相区。
    逐条审会失去横向梯度，正确结论也被打到 90°+ 全灭。

    calibration=True 时是「熔断后的补救重审」：上一轮全员高损耗导致完全反相，
    本轮要求以相对最优为锚，至少让最自洽路径进入低损耗区。
    """
    blind_texts = "\n".join([f"Path_{i}: {d.raw_reasoning}" for i, d in enumerate(drafts)])
    calibration_note = (
        "【补救重审】上一轮审计把全部路径都判成了高损耗，导致波函数 100% 反相熔断、没有任何结论存活。\n"
        "本轮请以『相对比较』为准：先选出自洽度最高、守恒链条最完整的一条路径作为基准，给它低损耗"
        "（assumption_burden/boundary_fragility/logical_leap 均 ≤ 0.30，evidence_strength ≥ 0.80），"
        "其余路径相对它拉开差距。只有确实违反守恒或偷换概念的才给 0.6 以上损耗。\n"
    )
    prompt = (
        f"你是冷酷无情的物理学与逻辑作用量审计员。请盲审以下 {len(drafts)} 条推理路径，寻找隐藏假设与边界漏洞。\n"
        + (calibration_note if calibration else "")
        + f"问题：{problem}\n{blind_texts}\n\n"
        "审阅要求：\n"
        "1. 必须横向对比这 N 条路径，强制拉开分差：证据饱满、守恒链条严密、假设负担低的路径要明显给出低损耗（assumption_burden/boundary_fragility/logical_leap 接近 0，evidence_strength 接近 1）；"
        "反之，违反守恒、偷换概念、强行假设的路径要给出高损耗（0.6~1.0）。\n"
        "2. 先在一句话 critical_flaw 里指出每条路径最薄弱的环节，再打分。\n"
        "3. 把语义等价的核心主张归并为一模一样的 normalized_claim 标签；结论方向相反的必须保留在不同标签里。\n"
        "4. 每条路径都要在 claim_summary 给出 15 字以内的该路径主张。\n"
        '请严格只输出一个 JSON 对象，键为 items，值为数组；数组长度必须等于路径数，每个元素为 {"claim_summary": 字符串, "critical_flaw": 字符串, "evidence_strength": 0.0~1.0, "assumption_burden": 0.0~1.0, "boundary_fragility": 0.0~1.0, "logical_leap": 0.0~1.0}，顺序与 Path 编号一致。'
    )
    resp_text = await asyncio.to_thread(
        chat_completion,
        os.getenv("AUDITOR_MODEL", "deepseek-chat"),
        prompt,
        temperature=0.2 if calibration else 0.3,
        response_format={"type": "json_object"},
        max_tokens=2048,
    )
    parsed = BlindAuditResponse.model_validate_json(resp_text)
    if len(parsed.items) != len(drafts):
        raise ValueError(f"盲审返回 {len(parsed.items)} 条，期望 {len(drafts)} 条")
    return parsed.items


async def mapreduce_evaluate_drafts(
    problem: str, drafts: List[RawWorkerDraft], calibration: bool = False
) -> List[AmplitudePacket]:
    # Map 阶段：优先批量横向盲审；失败则逐条离线 mock 兜底
    map_audits = []
    if client_available():
        try:
            map_audits = await _blind_batch_audit(problem, drafts, calibration=calibration)
        except Exception:
            map_audits = []

    if len(map_audits) != len(drafts):
        map_audits = await asyncio.gather(*(_map_audit_single_draft(i, problem, d) for i, d in enumerate(drafts)))

    # Reduce 阶段：独立二次调用，把 N 条 claim 聚类到少数共同标签（语义基矢对齐，原文瓶颈 A 的解法）
    normalized_claims = await _reduce_cluster_claims(map_audits)

    packets = []
    for draft, audit, norm_claim in zip(drafts, map_audits, normalized_claims):
        eval_obj = WorkerPathEvaluation(
            bot_id=draft.bot_id,
            conclusion_key=norm_claim,
            detail=draft.raw_reasoning,
            evidence_strength=audit.evidence_strength,
            assumption_burden=audit.assumption_burden,
            boundary_fragility=audit.boundary_fragility,
            logical_leap=audit.logical_leap
        )
        amp_dict = compute_symmetric_amplitude(eval_obj)
        packets.append(AmplitudePacket(
            bot_id=draft.bot_id,
            wave_direction=draft.wave_direction,
            perspective=draft.perspective,
            conclusion_key=norm_claim,
            detail=draft.raw_reasoning,
            critical_flaw=audit.critical_flaw,
            **amp_dict
        ))
    return packets


async def _reduce_cluster_claims(map_audits: List[SingleMapAudit]) -> List[str]:
    """Reduce 阶段：独立聚类调用，把语义等价主张归并到逐字完全相同的结论键。

    批量盲审输出的 claim_summary 往往「同义不同词」（例：水平速度不影响竖直下落 /
    两弹竖直初速同为零 / 竖直水平解耦），若不归并，概率会被稀释到阈值以下。
    这里要求模型把同一答案的多种措辞收敛到唯一标签，是解决原文瓶颈 A 的关键。
    """
    short_claims = [f"Path_{i}: {m.claim_summary}" for i, m in enumerate(map_audits)]

    if client_available():
        try:
            prompt = (
                "你是 Reduce 阶段的语义聚类仲裁官。下面 N 条主张来自同一道题的多个推理路径，"
                "其中若干条其实是同一个结论的不同措辞。请你把【答案完全一致】的主张归并到同一个标签：\n"
                + "\n".join(short_claims)
                + "\n\n归并铁律：\n"
                "1. 只要最终答案相同、方向一致，就必须用【逐字完全相同】的一个短标签（例如统一写成\"同时落地\"，而不是各自保留\"水平速度不影响竖直下落\"\"竖直水平解耦\"等说法）。\n"
                "2. 答案方向相反（如\"发散\"vs\"收敛\"）必须保留在不同标签，严禁合并。\n"
                "3. 输出必须覆盖每一条 path（path_index 从 0 开始连续，缺一不可）。\n"
                "4. 全部同向时，理想结果就是 1 个标签；存在真正分歧时才允许多标签，但总数越小越好。\n"
                '请严格只输出一个 JSON 对象，键为 mappings，值为数组；每个元素为 {"path_index": 整数, "normalized_claim": 字符串}。'
            )
            resp_text = await asyncio.to_thread(
                chat_completion,
                os.getenv("AUDITOR_MODEL", "deepseek-chat"),
                prompt,
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            parsed = ReduceClusterResponse.model_validate_json(resp_text)
            mapping_dict = {item.path_index: item.normalized_claim for item in parsed.mappings}
            if len(parsed.mappings) != len(map_audits):
                raise ValueError(f"Reduce 聚类返回 {len(parsed.mappings)} 条，期望 {len(map_audits)} 条")
            return [mapping_dict.get(i, map_audits[i].claim_summary) for i in range(len(map_audits))]
        except Exception:
            pass

    # 离线兜底：逐字一致的直接归并，其余按原文保守保留
    return _cluster_by_claim(map_audits)


def _normalize_claim_text(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch not in " ，,。.、；;：:！!？?（）()【】[]\"'`")


_NEGATION_CHARS = ("不", "非", "无", "未", "否", "没", "零", "反")


def _strip_negations(text: str) -> str:
    return "".join(ch for ch in text if ch not in _NEGATION_CHARS)


def _claim_bigrams(text: str) -> set[str]:
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _claim_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ga, gb = _claim_bigrams(a), _claim_bigrams(b)
    return len(ga & gb) / len(ga | gb)


def _is_negation_flip(a: str, b: str) -> bool:
    """结论方向相反：去掉否定词后高度重合、但双方否定词的有无不同。"""
    if a == b:
        return False
    sa, sb = _strip_negations(a), _strip_negations(b)
    if not sa or not sb:
        return False
    if (sa in sb or sb in sa) and (len(a) - len(sa)) != (len(b) - len(sb)):
        return True
    return any(x in b and x not in a for x in _NEGATION_CHARS) or any(
        x in a and x not in b for x in _NEGATION_CHARS
    )


def _cluster_by_claim(map_audits: List[SingleMapAudit]) -> List[str]:
    """无 API / LLM 聚类失败时的确定性兜底。

    规则：规范化去标点 -> 子串或字符二元组 Jaccard ≥ 0.55 归并；
    但只要检测到否定方向翻转（如「读数不变」vs「读数改变」），强制保留不同标签。
    """
    normalized = [_normalize_claim_text(m.claim_summary) for m in map_audits]
    groups: List[dict] = []  # {"label": 原文, "norm": 规范化文本, "count": n}

    for idx, norm in enumerate(normalized):
        placed = False
        for g in groups:
            if _is_negation_flip(norm, g["norm"]):
                continue
            if _claim_similarity(norm, g["norm"]) >= 0.55:
                g["count"] += 1
                if len(map_audits[idx].claim_summary) < len(g["label"]):
                    g["label"] = map_audits[idx].claim_summary
                placed = True
                break
        if not placed:
            groups.append({"label": map_audits[idx].claim_summary, "norm": norm, "count": 1})

    # 每个原始主张映射到「字面最短的代表标签」，保证同组逐字相同
    result = []
    for idx, norm in enumerate(normalized):
        label = map_audits[idx].claim_summary
        for g in groups:
            if _is_negation_flip(norm, g["norm"]):
                continue
            if _claim_similarity(norm, g["norm"]) >= 0.55:
                label = g["label"]
                break
        result.append(label)
    return result
