"""费曼语料铸字：从本地真实语料蒸馏四槽位费曼字库（带出处）。

语料来源（默认）：
- `/Users/yiyichen/Desktop/费曼物理_文字版/corpus/cpt_zh.jsonl` 中费曼本人署名的作品
  （讲义中译、别闹了费曼先生、发现的乐趣、费曼讲系列、QED 等）；
- `~/Projects/feynman-open/knowledge/*.jsonl` 的结构化思维动作与品味判据（由引擎读取）。

本模块只做统计与测量，不生成费曼没说过的话：
1. 句首起手式、实体、动作、后果短语，每一项记录 `source｜section` 出处；
2. 用句子特征（问句、人称、想象词、抽象后缀密度）测语体相位，不手填；
3. 槽位相位偏移也是量的：首小句 / 次小句 / 倒数第二 / 末小句 与整句相位之差；
4. 句法骨架从语料里挖，带真实出处与频次，渲染时才套用；
5. 字粒共现表（同句 ±5 字窗口）给"反直觉"算子提供"陌生但有据"的判据。

词性净化：装了 jieba 就用 `jieba.posseg` 判实体/动作；没装则只用后缀规则，
构建期可选、运行期零依赖。
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CORPUS_PATH = Path(
    os.getenv("QBOT_FEYNMAN_CORPUS", "/Users/yiyichen/Desktop/费曼物理_文字版/corpus/cpt_zh.jsonl")
)
KNOWLEDGE_DIR = Path(
    os.getenv("QBOT_FEYNMAN_KNOWLEDGE", str(Path.home() / "Projects/feynman-open/knowledge"))
)
VAULT_PATH = Path(os.getenv("QBOT_FEYNMAN_VAULT", "data/feynman_style_vault.json"))

AUTHORED_SOURCE_MARKERS = (
    "费恩曼物理学讲义",
    "别闹了，费曼先生",
    "发现的乐趣",
    "永远的费曼",
    "费曼讲",
    "QED",
    "物理定律的本性",
    "Surely you're joking",
    "费曼经典",
    "六堂 Easy",
    "Six Easy",
    "费曼物理学讲义英文版",
)
EXCLUDED_SOURCE_MARKERS = (
    "费曼传", "书评", "讣告", "读者来信", "百年诞辰", "纪念", "学习法", "思考法", "导读",
)
NOISE_PATTERNS = (
    "ISBN", "图书在版编目", "出版社", "印刷", "定价", "字数", "开本", "印张", "责任编辑",
    "版权所有", "译者序", "作者简介", "目录", "http", "www.", "第 1 页", "— 1 —",
)
# 与费曼去世、悼念、身后评价有关的段落一律不进字库：
# 字粒要能当"他本人会说的话"用，讣告与追忆是别人在说他。
DEATH_MARKERS = (
    "讣告", "去世", "逝世", "病逝", "死于", "临终", "弥留", "追悼", "悼念", "哀悼",
    "葬礼", "墓地", "墓碑", "遗孀", "遗作", "生前好友", "最后的日子", "生命最后",
    "1988年2月15日", "与世长辞", "永别", "缅怀", "纪念费曼", "费曼逝世", "纪念文集",
    "他死后", "身后", "葬礼上", "悼词", "盖棺",
)

FUNCTION_CHARS = set(
    "的了着过是在和与或把被将从向于就都而及其之为以对由并把吗呢吧啊呀嘛么这那我你他她它们个些很太更最又还只才再刚正不没别从沿向往朝跟给让使到却竟已曾要会能想该应须可每各当用条张片颗种次遍场座辆架艘层里中上下外前后"
)
VERB_SEEDS = (
    "穿过", "穿越", "行走", "震荡", "振荡", "坍缩", "唤醒", "撞上", "撞到", "渗透", "绕过",
    "拽住", "拉动", "推动", "吸住", "排斥", "吸引", "发热", "发光", "燃烧", "冷却", "融化",
    "蒸发", "沉淀", "漂浮", "旋转", "晃动", "跳动", "振动", "波动", "散开", "聚集", "排队",
    "碰撞", "弹开", "测量", "计数", "擦掉", "记住", "忘记", "想象", "观察", "盯着", "看见",
    "听见", "感觉到", "变成", "形成", "产生", "出现", "消失", "停止", "开始", "继续",
)
OBSERVABLE_MARKERS = (
    "读数", "指针", "发光", "发热", "火花", "条纹", "声音", "温度", "颜色", "味道", "影子",
    "痕迹", "烧焦", "变冷", "变热", "响", "亮", "升高", "降低", "变大", "变小", "移动",
)
ABSTRACT_SUFFIX = ("性", "化", "度", "机制", "效应", "系统", "范式", "表征", "耦合", "维度")

# 语篇连接词/代词：不是费曼的起手式，是任何中文都有的胶水
DISCOURSE_STOP = {
    "因此", "然而", "例如", "事实上", "现在", "所以", "但是", "而且", "并且", "不过",
    "首先", "其次", "最后", "总之", "换言之", "也就是说", "于是", "然后", "同时", "此外",
    "另外", "当然", "其实", "那么", "这样", "如果", "因为", "由于", "虽然", "尽管",
    "我们", "你们", "他们", "它们", "这个", "那个", "这些", "那些", "这里", "那里",
    "什么", "怎么", "为什么", "如何", "一个", "一种", "一面", "不是", "就是", "还有",
    "可以", "可能", "应该", "必须", "已经", "正在", "一直", "总是", "只是", "而是",
    "对于", "关于", "根据", "通过", "按照", "随着", "为了", "除了", "比如", "好像",
    "换句话说", "另一方面", "实际上", "在这种情况下", "这就是说", "举例来说", "反过来",
    "与此同时", "由此可见", "一般来说", "一般说来", "总的来说", "总而言之", "正是",
    "由此", "反之", "相反", "显然", "同样", "事实", "实际", "比方", "问题是", "举例", "也就是说",
}
# 元话语名词：句子里在谈论"话本身"而不是世界里可测的东西。
# 它们可以出现在任何作者笔下，不构成费曼字粒；混进实体槽会让排版句变成
# "角度穿过概率"这种空转句，所以从实体/起手式/后果三处一并拦掉。
META_DISCOURSE_NOUNS = (
    "白话", "比方", "例子", "举例", "话题", "情况", "意思", "角度", "细节", "思路",
    "结论", "说法", "东西", "事情", "时候", "地方", "部分", "整体", "核心", "关键",
    "重点", "难点", "问题", "事实", "内容", "方面", "方式", "方法", "结果", "过程",
    "状态", "概念", "答案", "意义", "关系", "影响", "作用", "条件", "基础", "本质",
    "结构", "性质", "特征", "定义", "规则", "观念", "形式", "现象", "情形", "事物",
    "证明", "说明", "解释", "讨论", "观点", "看法", "想法", "尝试", "步骤", "章节",
    "世界", "知识", "理论", "数学", "科学", "方程", "公式", "理由", "原因", "目的",
)
# 书页脚手架式开场：谈的是"接下来要讲什么/这本书怎么组织"，不是现象本身
META_HOOK_STOP = {
    "答案是", "问题在于", "事实证明", "结果表明", "结果发现", "这样看来", "由此可见",
    "接下来", "类似地", "特别地", "同样地", "到目前为止", "迄今为止", "很久以前",
    "在上一章中", "在下一章中", "作为一个例子", "比如说", "比方说", "这意味着",
    "很清楚", "很明显", "遗憾的是", "幸运的是", "有趣的是", "特别是", "也就是说",
}
META_HOOK_PREFIXES = ("在量子力学中", "在相对论中", "在经典力学中", "在上一章", "在下一章",
                      "在这一章", "在这一节", "在本书", "在这一部分")

# 费曼式起手式的特征词：出现则加分
RHETORICAL_MARKERS = (
    "你", "想", "假设", "假想", "想象", "记住", "注意", "看", "听", "别", "不要",
    "让", "如果", "为什么", "试", "不妨", "瞧", "猜", "比方", "举例",
)
ENTITY_STOP_SUFFIX = ("的", "了", "是", "有", "在", "和", "与", "或", "及", "等", "们")
# 动词/泛化词不可能做实体槽；段内出现这些切字也切开
ENTITY_BREAK_CHARS = set("有一另知说做想觉得很太更最多少好坏新旧这点那个些每各真假")
# 实体切分不动情态/使役字：否则"想象力"会被切成"象力"
ENTITY_FUNCTION_CHARS = FUNCTION_CHARS - set("想要会能可应须该使让给")
GENERIC_WORDS = {
    "知道", "发现", "开始", "出现", "观察", "测量", "产生", "移动", "变成", "形成", "问题",
    "情况", "事实", "实际", "理解", "认为", "看到", "觉得", "意思", "事情", "东西", "时间",
    "地方", "方式", "方法", "结果", "原因", "过程", "状态", "例子", "答案", "说法", "概念",
    "时候", "相同", "不同", "有些", "所有", "一切", "部分", "整个",
    "变化", "定律", "方面", "告诉", "消失", "存在", "作用", "影响", "关系", "意义",
    "内容", "任何", "简单", "复杂", "讨论", "发生", "理论", "知识", "整个世界", "自然界",
    "象力", "方式", "条件", "基础", "本质", "结构", "性质", "特征",
    "注意", "考虑", "意味", "完全", "确定", "自己", "对方", "大家",
}
VERB_WORDS = set(VERB_SEEDS) | {
    "知道", "发现", "开始", "出现", "观察", "测量", "产生", "移动", "变成", "形成", "认为",
    "觉得", "看到", "理解", "研究", "解释", "描述", "推", "拉", "转动", "放在", "拿起",
}
# --- P8 铸字净化（对照 PYBOT 4.1 文档，按本仓库数据重写）--------------------
# 悬垂从句尾巴：带这些尾巴的片段不是完整实体/从句，是切句切在中途。
DANGLING_TAILS = (
    "的时候", "的话", "之后", "以后", "之前", "以前", "之时", "来说", "来讲",
    "来看", "而言", "对于", "关于", "通过", "根据", "按照", "随着", "为了",
    "除了", "并", "而", "把", "将", "向", "的",
)
# 动作槽里收尾在名词上的片段（"观察水""旋转木马""振荡电荷"）是"动词+宾语"，
# 不是动作本身；句法上会让"把{E}{A}"读成两个宾语。
ACTION_OBJECT_HEADS = ("n", "m", "q")
# 后果从句不许这些词开头：切剩下的连接词残片
CONSEQUENCE_BAD_HEADS = (
    "然", "就", "还", "也", "都", "又", "便", "所以", "因此", "但是", "不过",
    "而且", "并且", "于是", "因为", "然后", "如果", "假如", "随着", "通过",
    "为了", "根据", "由于", "当", "只要", "类似", "顺便", "有一天", "简单",
    "作为", "人们", "无论如何", "不管", "比方", "比如", "举个例子", "举例",
)
CONSEQUENCE_BAD_INNER = ("起见", "提一", "举个例子", "比如说", "我们来看", "你会发现")
# 时间/方位收尾：教材段按逗号切开后常见的伪后果（"事故发生前不久""爆炸三分钟后"）
CONSEQUENCE_TEMPORAL_TAILS = (
    "前不久", "不久前", "之前", "以前", "之后", "以后", "的时候", "期间", "以来",
    "前后", "那时", "此时", "当时", "过后", "之后", "三分钟后", "分钟后", "小时后",
)

# P8：后果槽只收"发生的事情/产生的变化"，句中必须有一个变化类动词。
# 光靠词性会把"严格地说""老实说""必须承认"这类口头语收进来。
EFFECT_VERB_SEEDS = (
    "变", "成为", "升高", "降低", "增加", "减少", "增大", "减小", "出现", "消失",
    "发生", "产生", "得到", "获得", "形成", "达到", "移动", "转动", "旋转", "振动",
    "发光", "发热", "冷却", "加热", "膨胀", "收缩", "断裂", "停止", "停下", "加速",
    "减速", "弯曲", "增强", "减弱", "损失", "失去", "保持", "相等", "平衡", "偏离",
    "偏转", "散射", "反射", "折射", "吸收", "发射", "辐射", "传导", "冻结", "融化",
    "燃烧", "爆炸", "上升", "下降", "落下", "流动", "改变", "转化", "抵消", "放大",
    "缩小", "扩散", "穿过", "击中", "碰撞", "受到", "作用", "推动", "拉动",
)
PROPER_TAGS = ("nr", "ns", "nz", "nx")
# 后果句不能以这些开头：残片、疑问、指示代词、连词
P8_CONSEQUENCE_BAD_HEADS = (
    "使得", "使", "是", "是否", "如何", "怎样", "现在", "这里", "那里", "它", "他", "她",
    "这", "那", "所以", "因此", "于是", "但", "而", "或", "并且", "即使", "只有",
    "无论", "不管", "为", "把", "将", "被", "给", "以及", "并", "若", "如果", "假如",
    "一旦", "除非", "越", "问题", "实际", "说的是", "这就是说",
    "即", "结果", "很少", "假定", "知道", "假如",
)
# 条件从句/疑问残留不是已发生的后果
P8_CONDITIONAL_INNER = ("如果", "假如", "一旦", "即使", "除非", "倘若", "为什么", "怎么",
                        "而非", "使得", "以及", "并且")
# 这些词性开头说明分句被切在半途（助词、副词、代词、时间词、介词、方位词、连词）
BAD_FIRST_TAGS = ("h", "y", "d", "t", "r", "p", "f", "c", "vg", "ud", "ug", "ul", "uv", "uj", "uz")
# 这些词性收尾说明尾巴被切掉（区别词、前接成分、数词/量词/代词残片）
BAD_TAIL_TAGS = ("b", "h", "y", "d", "c", "p", "f", "m", "q", "r", "t")
# 反复重建时稳定复现、但语义上不是"后果"的条目：人工核对后显式排除
P8_CONSEQUENCE_DENY = {
    "要做的事情就是改变", "怀疑的看法逐渐地出现", "用来求出台球的碰撞规则",
    "头脑中自然就会产生好奇心", "要通过在另一个地方缓慢移动来弥补",
    "没有一个得到成功", "发射一个频率", "温度等变化", "我会停下来",
}
# 这些开头是祈使/意图，不是已经发生的变化
P8_IMPERATIVE_HEADS = ("要", "做", "怀疑", "让", "用来", "没有", "头脑", "试图", "设法")

# 因果标记：既是后果候选的切分点，也是 P8 里 A→C 边的唯一来源
P8_CAUSAL_MARKERS = ("所以", "因此", "于是", "结果", "使得", "从而使",
                     "导致", "造成", "引起")


# jieba 会把少数副词/动词误标成名词（无法→n、明白→nr）：P8 实体门里显式拦掉
P8_ENTITY_STOP = {"无法", "明白", "办法", "论证", "错误", "我会", "我们", "你们", "他们",
                  "时会"}  # jieba 把"…时会…"切成一个 n 词，实际是"时+会"的残片

# 残余边界抽象词：jieba 判 n、语法合法，但没有可操作的具体画面。
# 只收整词，不按"度/量/率/性"宽后缀拦——质量/速度/温度/动量/频率/强度/距离
# 是费曼原话里的常规主语（"温度升高"），宽后缀会误伤基线。
GENERIC_ENTITY_WORDS = {
    "材料", "原料", "资料", "燃料", "物体", "物品", "事物", "产物", "装置", "设备",
    "仪器", "器具", "器件", "工具", "介质", "媒介", "载体", "样本", "样品", "零件",
    "元件", "整体", "个体", "主体", "客体", "实体", "事物", "物质",
}
METRIC_ABSTRACT_WORDS = {
    "分量", "矢量", "标量", "张量", "矩阵", "坐标", "投影", "参数", "变量", "常量",
    "系数", "指数", "维度", "自由度", "比率", "比例", "数值", "读数", "差值", "总和",
    "均值", "数字", "大量", "程度",
}
FRAMEWORK_ABSTRACT_WORDS = {
    "方向", "位置", "地点", "方位", "表面", "内部", "外部", "中心", "终点", "起点",
    "轨迹", "路径", "区域", "范围", "领域", "原理", "模型", "假设", "名称", "符号",
    "现象", "事件", "规律", "时刻", "历史", "技术", "原则", "贡献", "基本", "意识",
    "微子",
}
ABSTRACT_ENTITY_WORDS = (GENERIC_ENTITY_WORDS | METRIC_ABSTRACT_WORDS
                         | FRAMEWORK_ABSTRACT_WORDS)
# 复合词的窄语素尾：只收"料/品/件/域/式/型"这类集合或框架收尾，
# 不收"体/物/度/量"（超导体/障碍物/重物/质量/速度要靠整词判定）。
ABSTRACT_MORPHEME_TAILS = ("材料", "原料", "介质", "装置", "仪器", "器件", "元件",
                           "样品", "物品", "物体", "模型", "规律", "原理", "轨迹",
                           "路径", "区域", "范围", "方向")
ABSTRACT_COMPOUND_TAILS = ("料", "品", "件", "域", "式", "型")
# 具体器物豁免：尾字可能落在上面规则里，但语义上是能摸到的物件
CONCRETE_ENTITY_WHITELIST = {"重物", "障碍物", "超导体", "导体", "晶体", "流体",
                             "液体", "气体", "固体", "磁铁", "弹珠", "弹簧", "线圈", "狭缝"}

# 第二梯队实体：边界抽象词不判死刑，允许入槽但单独配额、单独排序。
# 它们的搭配里常有真东西（"物体|碰撞 85""矢量|旋转 31"），只是不能像具体物那样
# 挤占前 60 颗的位置。入槽条件：和至少 TIER2_MIN_SPECIFIC_EDGES 个非相位动作同句共现。
TIER2_ENTITY_SLOTS = 6
TIER2_MIN_SPECIFIC_EDGES = 3
PHASE_ACTION_WORDS = {"产生", "出现", "开始", "形成", "变成", "消失", "停止", "继续", "存在"}
# 第二梯队只收"能当话题的名词"：单字形容词素（基本）、量词（大量）、切词残片（微子）
# 不进兜底队列，它们连边界词都算不上
TIER2_INELIGIBLE = {"基本", "大量", "微子", "数字", "时刻", "程度"}


def entity_tier(word: str) -> int:
    """实体槽分梯队：1=具体实体，2=有搭配证据才进槽的边界抽象词，0=不收。"""
    if is_pure_noun_phrase(word):
        return 1
    if word in ABSTRACT_ENTITY_WORDS and word not in TIER2_INELIGIBLE:
        return 2
    return 0

# 及物动词：排版时放在实体前面（"观察电子"），否则跟实体后面（"电子旋转"）
TRANSITIVE_SEEDS = (
    "观察", "测量", "吸引", "推动", "拉动", "拽住", "吸住", "排斥", "唤醒", "记住",
    "忘记", "想象", "盯着", "看见", "听见", "感觉到", "擦掉", "计数", "研究", "解释", "描述",
)
# 后果必须落在可观测变化上，不能是条件从句
CONSEQUENCE_HEAD_STOP = ("如果", "假设", "随着", "通过", "为了", "根据", "由于", "当", "只要", "因为")
CONSEQUENCE_TAIL = tuple(OBSERVABLE_MARKERS) + (
    "升高", "降低", "变大", "变小", "变亮", "变暗", "变热", "变冷", "加快", "减慢",
    "移动", "停止", "继续", "出现", "消失", "相同", "不同", "相等",
)
# 后果片段常见的书页脚手架/人称残留：带这些的一律不收
CONSEQUENCE_BAD = ("图中", "如图", "见表", "我们看到", "我们", "你们", "他们", "这一个", "那个", "漂亮的")
CONSEQUENCE_LEAD = (
    "曾经", "经有过", "所以", "因此", "但是", "不过", "而且", "并且", "于是", "因为",
)
CONSEQUENCE_TAIL_TRIM = ("的时候", "时候", "之后", "以后", "之前", "以前", "来说", "来讲", "来看", "而言", "时")


def _clean_consequence(w: str) -> str:
    """去掉尾部"…时/的时候/来说"与句首连接词；清不干净就返回空串（丢弃）。"""
    for lead in CONSEQUENCE_LEAD:
        if w.startswith(lead) and len(w) - len(lead) >= 4:
            w = w[len(lead):]
    for suf in CONSEQUENCE_TAIL_TRIM:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            break
    return w


def _consequence_ok(w: str) -> bool:
    if any(bad in w for bad in CONSEQUENCE_BAD):
        return False
    # 元话语/残片：谈"说法、角度、问题"的片段不是可测后果
    if w.startswith(("一定的", "这个说法", "这种情况")) or w.endswith(tuple(META_DISCOURSE_NOUNS)):
        return False
    # 开在动词上的片段是动作不是后果（"移动探测器""说出成分"）
    if w.startswith(tuple(VERB_WORDS)) or w.startswith(("说出", "提高", "降低", "改变", "类似", "变成")):
        return False
    if w.endswith(("影响", "来说", "来讲", "来看", "而言", "漂亮")):
        return False
    # 单字观测词（亮/响/热）只认句尾，避免"漂亮的椭圆"这种假命中
    if w.endswith(CONSEQUENCE_TAIL):
        return True
    return any(t in w[-6:] for t in CONSEQUENCE_TAIL if len(t) >= 2)

# --- 词性净化 -------------------------------------------------------------
# 名词化/抽象化后缀：结尾是它们、又不在白名单里的片段不进实体槽。
ABSTRACT_ENTITY_SUFFIX = ("性", "化", "机制", "效应", "系统", "范式", "表征", "耦合", "维度", "主义", "体系")
ABSTRACT_ENTITY_SUFFIX_3 = ("度", "率")
# 动作槽的伪动词后缀（名词化动词/量词化）：不收
NOMINAL_ACTION_SUFFIX = (
    "性", "度", "率", "力", "者", "器", "方程", "场", "子", "时", "面", "结果", "过程",
    "关系", "效应", "机制", "主义", "方式", "状态", "情况", "系统", "理论", "周期",
    "模式", "时间", "距离", "方向",
)
# 贪心截出来的半截词（"出现什""排斥作""振荡器频"）：最后一个字是这些的一律不收
ACTION_FRAGMENT_SUFFIX = ("某", "什", "磁", "作", "频", "方")

# --- 句法骨架 -------------------------------------------------------------
# 模板是给四槽位填空的框；每种框必须在本语料里真实出现过（freq≥3）才进字库，
# 并记录第一条匹配句与出处。整句是"拼装句"，不是费曼原句——出处只对字粒负责。
SKELETON_DEFS = (
    ("declarative", "{hook}，{verb}，{consequence}。"),
    ("conditional", "{hook}，假设{verb}，{consequence}。"),
    ("temporal", "{hook}，{verb}的时候，{consequence}。"),
    ("causal", "{hook}，{verb}，所以{consequence}。"),
    ("contrast", "{hook}，{verb}，但是{consequence}。"),
    ("retro", "{consequence}——之所以会这样，是因为{hook}，{verb}。"),
    ("analogy", "{hook}，{verb}，就像{consequence}。"),
)
SKELETON_PATTERNS = {
    "conditional": re.compile(r"^(如果|假如|假设|设想|想象一下)"),
    "temporal": re.compile(r"(的时候|之时)"),
    "causal": re.compile(r"(所以|因此|于是|这就是为什么|正因为)"),
    "contrast": re.compile(r"(但是|不过|可是|然而)"),
    "retro": re.compile(r"^(之所以|为什么|原因是|这是因为)"),
    "analogy": re.compile(r"(就像|好比|好像|如同)"),
}
DEFAULT_SLOT_OFFSETS = {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0}

STYLE_LECTURE, STYLE_DIALOGUE, STYLE_SPEECH = "lecture", "dialogue", "speech"

_POSSEG = None


def _posseg():
    """jieba 可选：装了就用词性，没装返回 None（只用后缀规则）。"""
    global _POSSEG
    if _POSSEG is None:
        try:
            import jieba.posseg as pseg  # type: ignore
            pseg.cut("测试")  # 预热词典，失败则降级
            _POSSEG = pseg
        except Exception:
            _POSSEG = False
    return _POSSEG or None


def purify_entity_word(word: str) -> bool:
    """实体槽只要名词性、去抽象化：吸引力/重要性这类不收，电子/水面/咖啡杯收。"""
    if len(word) < 2:
        return False
    if word in META_DISCOURSE_NOUNS:
        return False
    if word.endswith(ABSTRACT_ENTITY_SUFFIX):
        return False
    if len(word) >= 3 and word.endswith(ABSTRACT_ENTITY_SUFFIX_3):
        return False
    pseg = _posseg()
    if pseg:
        pairs = [(w, f) for w, f in pseg.cut(word) if w.strip()]
        if not pairs:
            return False
        return all(f.startswith("n") or f in ("j", "l", "eng", "nz") for _, f in pairs)
    return True


def purify_action_word(word: str) -> bool:
    """动作槽要真动词：结尾名物化的（…性/…度/…结果）不收。"""
    if len(word) < 2:
        return False
    if word.endswith(NOMINAL_ACTION_SUFFIX) or word.endswith(ACTION_FRAGMENT_SUFFIX):
        return False
    pseg = _posseg()
    if pseg:
        head = word[:2]
        flags = [f for _, f in pseg.cut(head)]
        return any(f.startswith("v") for f in flags)
    return True


def _seg(word: str) -> List[Tuple[str, str]]:
    pseg = _posseg()
    if not pseg:
        return []
    return [(w, f) for w, f in pseg.cut(word) if w.strip()]


def is_pure_noun_phrase(word: str) -> bool:
    """实体槽只要纯名词短语：末词性必须名词，动词性短语、悬垂尾巴一律不收。

    例：电子 ✅；天空变成 ❌（动词收尾）；的时候 ❌（悬垂）；无法 ❌（副词）。
    """
    if not (2 <= len(word) <= 8) or not _is_cjk(word):
        return False
    if word in META_DISCOURSE_NOUNS or word in GENERIC_WORDS or word in DISCOURSE_STOP:
        return False
    if word in P8_ENTITY_STOP:
        return False
    if word not in CONCRETE_ENTITY_WHITELIST:
        if word in ABSTRACT_ENTITY_WORDS:
            return False
        if word.endswith(ABSTRACT_MORPHEME_TAILS) or word.endswith(ABSTRACT_COMPOUND_TAILS):
            return False
    else:
        # 白名单里的具体器物不受词性噪声影响（"狭缝"被 jieba 标成区别词 b）
        return True
    if word.endswith(DANGLING_TAILS):
        return False
    if any(word.endswith(s) for s in ABSTRACT_ENTITY_SUFFIX):
        return False
    if any(word.endswith(v) for v in VERB_SEEDS):
        return False
    pairs = _seg(word)
    if not pairs:
        return False
    nominal = ("n", "j", "l", "eng", "nz")
    if not all(f.startswith("n") or f in nominal for _, f in pairs):
        return False
    return pairs[-1][1].startswith("n") or pairs[-1][1] in nominal


def is_clean_action(word: str) -> bool:
    """动作槽只收"动词(+一个动词补语)"：动词+宾语、碎片、名词化一律不收。

    例：观察 ✅；测量出 ✅；开始旋转 ✅；观察水 ❌；旋转木马 ❌；震荡电荷 ❌。
    """
    if not (2 <= len(word) <= 5) or not _is_cjk(word):
        return False
    if any(ch in FUNCTION_CHARS for ch in word):
        return False
    if word.endswith(NOMINAL_ACTION_SUFFIX) or word.endswith(ACTION_FRAGMENT_SUFFIX):
        return False
    pairs = _seg(word)
    if not pairs:
        return word in VERB_SEEDS
    if not (pairs[0][1].startswith("v") or word in VERB_SEEDS):
        return False
    rest = pairs[1:]
    if len(rest) > 1:
        return False
    for w, f in rest:
        if f.startswith(ACTION_OBJECT_HEADS) or f.startswith(("d", "p", "c", "r", "u", "a")):
            return False
    return True


def classify_action_valency(word: str, order: Optional[str] = None) -> str:
    """vo = 能带宾语（观察/测量出）；sv = 不及物（出现/开始旋转）。

    判据按可靠度排序：词典种子 > 派生形态（动词+补语） > 现有 order 标注。
    """
    if word in TRANSITIVE_SEEDS or (len(word) >= 2 and word[:2] in TRANSITIVE_SEEDS):
        return "vo"
    if word.endswith(("出", "到", "住", "上", "下", "开", "起", "进")):
        return "vo"
    return "vo" if order == "pre" else "sv"


def _p8_consequence_ok(word: str) -> bool:
    """P8 后果门：完整从句 + 含变化类动词 + 不是切半残片/人名。

    比 is_complete_consequence 更严：后者只保证"是个从句"，
    这里再要求句子真的在讲一件发生变化的事。
    """
    if not is_complete_consequence(word):
        return False
    if len(word) >= 16:                        # 卡在上限＝多半被截断（"…会发生什"）
        return False
    if word in META_HOOK_STOP or any(b in word for b in CONSEQUENCE_BAD_INNER):
        return False
    if word in P8_CONSEQUENCE_DENY:
        return False
    if word.startswith(P8_IMPERATIVE_HEADS):
        return False
    if word.startswith(P8_CONSEQUENCE_BAD_HEADS):
        return False
    if any(c in word for c in P8_CONDITIONAL_INNER):
        return False
    if not any(s in word for s in EFFECT_VERB_SEEDS):
        return False
    pairs = _seg(word)
    if not pairs:
        return False
    head_word, head_tag = pairs[0]
    tail_word, tail_tag = pairs[-1]
    if head_tag.startswith(PROPER_TAGS):       # 人名/专名开头：费曼、洛伦兹
        return False
    if head_tag.startswith(BAD_FIRST_TAGS):    # 切在半途：助词/副词/代词/时间词
        return False
    if head_tag.startswith("v") and len(pairs) > 2:
        # 动词开头且超过两词：多半是把前半句动作带进来了（"看见那道白光变成黄色"）
        return False
    if tail_tag.startswith(BAD_TAIL_TAGS):     # 尾巴被切掉：电磁、声响的…
        return False
    if tail_tag.startswith(PROPER_TAGS):       # 尾巴是专名："场发射显微镜"
        return False
    if any(w0 == "等" for w0, _ in pairs[:-1]):   # "温度等变化"：列举尾巴
        return False
    if len(head_word) == 1 and head_tag.startswith("n"):   # "量增加"：切半
        return False
    if len(tail_word) == 1 and tail_tag.startswith("v"):   # "方程变"：切半
        return False
    if len(pairs) >= 2 and pairs[-2][1] == "uj" and tail_tag.startswith("v"):
        return False                           # "…的分析"：的+动词＝名词化
    if len(pairs) == 2 and tail_tag.startswith("n") and head_tag.startswith("v"):
        return False                            # "振动系统"：动词+名词是名词短语
    if (len(pairs) == 3 and tail_tag.startswith("n") and head_tag.startswith("n")
            and pairs[1][1].startswith("v")):
        return False                            # "场发射显微镜"：名词复合
    # 扩大语料后暴露的碎片形态（R14 教材段）：形容词/数量词开头没有主语、
    # 动词叠动词没有宾语、名词串收尾、简称收尾。这些是名词短语或残片，不是后果。
    if len(pairs) == 2 and head_tag.startswith(("a", "m", "d")):
        return False                            # "大爆炸""第一次发生"
    if len(pairs) == 2 and head_tag.startswith("v") and tail_tag.startswith("v"):
        return False                            # "会发生""引发爆炸"
    if tail_tag == "j":
        return False                            # "反射波"：以简称/缩略收尾
    if (len(pairs) >= 3 and all(t.startswith("n") for _, t in pairs[:-1])
            and not any(t in ("uj", "p", "d", "v", "c") for _, t in pairs[1:-1])):
        return False                            # "宇宙微波背景辐射"：名词串当从句
    return True


def is_complete_consequence(word: str) -> bool:
    """后果槽要完整从句：有谓词；以名词收尾时，前面必须是"动词+宾语"而不是"…的+名词"。

    例：温度升高 ✅；水平移动一个重物 ✅；霓虹灯的颜色 ❌（无谓词）；
    响声出现的速率 ❌（"的"把动词短语变成了名词短语）；然会发出声音 ❌（残片）。
    """
    if not (3 <= len(word) <= 16) or not _is_cjk(word):
        return False
    if word.endswith(DANGLING_TAILS):
        return False
    # 时间/方位收尾是教材段切出来的残片（"事故发生前不久"），不是后果
    if word.endswith(CONSEQUENCE_TEMPORAL_TAILS):
        return False
    # 只留下"变成/成为"没有宾语：句子被切断了
    if word.endswith(("变成", "成为")):
        return False
    if word.startswith(CONSEQUENCE_BAD_HEADS):
        return False
    if any(bad in word for bad in CONSEQUENCE_BAD):
        return False
    pairs = _seg(word)
    if not pairs:
        return False
    flags = [f for _, f in pairs]
    if not any(f.startswith("v") or f.startswith("a") for f in flags):
        return False
    if pairs[-1][1].startswith("n"):
        verb_idx = max((i for i, f in enumerate(flags) if f.startswith("v")), default=-1)
        if verb_idx < 0:
            return False
        between = pairs[verb_idx + 1:-1]
        if not all(f.startswith(("m", "q")) for _, f in between):
            return False
    return True


@dataclass
class Item:
    word: str
    slot: str
    count: int = 1
    amp: float = 1.0
    domain: str = "everyday"
    style: str = STYLE_LECTURE
    phase_deg: float = 15.0
    sources: List[str] = field(default_factory=list)
    example: str = ""

    def to_dict(self) -> dict:
        return {
            "word": self.word, "slot": self.slot, "count": self.count, "amp": self.amp,
            "domain": self.domain, "style": self.style, "phase_deg": round(self.phase_deg, 1),
            "sources": self.sources[:3], "example": self.example[:80],
        }


def _clean_source(src: str) -> bool:
    low = src.lower()
    if any(m.lower() in low for m in EXCLUDED_SOURCE_MARKERS):
        return False
    return any(m.lower() in low for m in AUTHORED_SOURCE_MARKERS)


def iter_sections(corpus_path: Path = CORPUS_PATH, trusted: bool = False):
    """(source, section, text)；只保留费曼署名作品的干净段落。

    trusted=True 用于已人工登记来源的集成语料：跳过"署名作品"白名单，
    但排除词、噪声、去世词与长度/Latin 过滤照旧。
    """
    if not corpus_path.is_file():
        return
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            src = str(rec.get("source", ""))
            section = str(rec.get("section", ""))
            text = str(rec.get("text", "")).strip()
            if not text or (not trusted and not _clean_source(src)):
                continue
            if any(p in text for p in NOISE_PATTERNS):
                continue
            if any(m in text or m in src or m in section for m in DEATH_MARKERS):
                continue
            if len(text) < 30 or len(text) > 4000:
                continue
            latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
            if latin > len(text) * 0.6:
                continue
            yield src, section, text


def sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？；])|\n+", text)
    return [p.strip(" \t\r\n“”\"'·-—") for p in parts if 15 <= len(p.strip()) <= 120]


def clauses(sentence: str) -> List[str]:
    return [c.strip() for c in re.split(r"[，、：,]", sentence) if len(c.strip()) >= 2]


def _style_of(sentence: str) -> Tuple[str, float]:
    """从句子本身量语体相位：约 0–40° 讲义，40–80° 聊天，80–120° 演讲哲理。"""
    q = sentence.count("？")
    first_person = sentence.count("我") + sentence.count("你")
    exclaim = sentence.count("！")
    abstract = sum(sentence.count(s) for s in ABSTRACT_SUFFIX)
    imaginal = sum(sentence.count(w) for w in ("想象", "假设", "比如", "好像", "像", "假想"))
    score = (
        0.9 * q + 0.5 * first_person + 0.7 * exclaim + 0.9 * imaginal
        - 1.4 * abstract - 0.02 * max(0, len(sentence) - 40)
    )
    if score >= 2.5:
        return STYLE_SPEECH, round(min(120.0, 95.0 + min(20.0, score)), 1)
    if score >= 0.5:
        return STYLE_DIALOGUE, round(55.0 + (score - 0.5) * 10, 1)
    return STYLE_LECTURE, round(max(5.0, 25.0 + score * 8), 1)


def _is_cjk(word: str) -> bool:
    return bool(word) and all("一" <= ch <= "鿿" for ch in word)


def _trim(word: str) -> str:
    w = word.strip()
    while w and (w[0] in FUNCTION_CHARS or w[-1] in FUNCTION_CHARS):
        if w[0] in FUNCTION_CHARS:
            w = w[1:]
        if w and w[-1] in FUNCTION_CHARS:
            w = w[:-1]
    return w


def _trim_tail(word: str) -> str:
    """只去尾虚词，保留句首——P8 后果要用完整开头判断是不是残片。"""
    w = word.strip()
    while w and w[-1] in FUNCTION_CHARS:
        w = w[:-1]
    return w


def _usable(word: str, min_len: int = 2, max_len: int = 8) -> bool:
    if not (min_len <= len(word) <= max_len):
        return False
    if not _is_cjk(word):
        return False
    if word in DISCOURSE_STOP:
        return False
    if any(word.endswith(s) for s in ENTITY_STOP_SUFFIX):
        return False
    return True


# 日常物的形态线索：讲义译文中日常实体少，共现统计会把它们误判成物理词，
# 所以先用词形兜底，再退回共现统计。
EVERYDAY_MORPHEMES = (
    "球", "水", "杯", "桌", "椅", "门", "窗", "车", "猫", "狗", "手", "眼", "嘴", "冰",
    "火", "风", "雨", "绳", "弹簧", "钟", "表", "镜", "照片", "纸", "笔", "山", "河",
    "树", "花", "鸟", "鱼", "人", "饭", "茶", "咖啡", "苹果", "蚂蚁", "鞋", "衣服",
    "房间", "街", "路", "孩子", "朋友", "家", "床", "灯", "声音", "颜色", "影子",
)


def _entity_domain(word: str, stats: Dict[str, Counter]) -> str:
    if any(m in word for m in EVERYDAY_MORPHEMES):
        return "everyday"
    phys = sum(stats[word][k] for k in ("仪器", "测量", "粒子", "原子", "电子", "能量", "实验", "光谱", "磁场", "力"))
    life = sum(stats[word][k] for k in ("看见", "摸", "像", "比如", "想象", "吃饭", "走路", "家里", "朋友", "手", "眼睛"))
    return "physics" if phys > life else "everyday"


def build_vault(corpus_path: Path = CORPUS_PATH, limit_per_slot: int = 60,
                p8: bool = False, trusted: bool = False,
                slot_limits: Optional[Dict[str, int]] = None,
                consequence_inventory: Optional[Path] = None,
                edge_supplement: Optional[Path] = None) -> dict:
    """consequence_inventory：P9 策展库存文件。给了就整槽替换后果字粒
    （条目须带 slot=consequence、word、来源与出处），其余槽与默认路径不变。

    edge_supplement：JSON 文件，{"实体|后果": 次数}。给来自语料之外的书里
    观察到的 E→C 同句共现补边——这些句子不在语料里，第一遍统计看不见。
    只做加法：键必须指向存在且带 slot 的词，次数必须为正整数；语料里已有
    的计数原样保留，补的是增量。不给时与旧行为逐字节一致。"""
    hook_counts: Counter = Counter()
    hook_src: Dict[str, List[str]] = defaultdict(list)
    hook_meta: Dict[str, Tuple[str, float, str]] = {}

    entity_counts: Counter = Counter()
    entity_ctx: Dict[str, Counter] = defaultdict(Counter)
    entity_src: Dict[str, List[str]] = defaultdict(list)
    entity_meta: Dict[str, Tuple[str, float, str]] = {}

    action_counts: Counter = Counter()
    action_src: Dict[str, List[str]] = defaultdict(list)
    action_meta: Dict[str, Tuple[str, float, str]] = {}
    action_order: Dict[str, str] = {}

    cons_counts: Counter = Counter()
    cons_src: Dict[str, List[str]] = defaultdict(list)
    cons_meta: Dict[str, Tuple[str, float, str]] = {}

    sections = sents_total = 0
    offset_samples: Dict[str, List[float]] = {s: [] for s in DEFAULT_SLOT_OFFSETS}
    skeleton_hits: Dict[str, dict] = {
        kind: {"freq": 0, "phases": [], "provenance": []} for kind, _ in SKELETON_DEFS
    }

    for src, section, text in iter_sections(corpus_path, trusted=trusted):
        sections += 1
        src_tag = f"{src}｜{section[:24]}"
        for sent in sentences(text):
            sents_total += 1
            style, phase = _style_of(sent)

            # 起手式：句首到第一个停顿；要求是费曼式引导语而非逻辑连接词
            head = re.split(r"[，,？！。]", sent)[0].strip(" \t“”\"'·-—")
            head_ok = (
                2 <= len(head) <= 10 and _is_cjk(head) and head not in DISCOURSE_STOP
                and head not in META_HOOK_STOP and not head.startswith(META_HOOK_PREFIXES)
                and head not in META_DISCOURSE_NOUNS
                and not head.startswith(("第", "图", "表", "§", "公式", "在本", "本章", "这一章", "这一节",
                                          "下面", "上述", "前面", "后面", "如图", "见表", "正如"))
                and (len(head) >= 3 or any(m in head for m in RHETORICAL_MARKERS))
                and not (head.startswith("我们") and not any(v in head for v in ("看", "知道", "假设", "可以", "发现", "想", "说")))
            )
            if head_ok:
                bonus = 2.0 if any(m in head for m in RHETORICAL_MARKERS) else 1.0
                hook_counts[head] += bonus
                hook_meta.setdefault(head, (style, phase, sent[:60]))
                if src_tag not in hook_src[head] and len(hook_src[head]) < 3:
                    hook_src[head].append(src_tag)

            # 实测槽位偏移：首小句≈起手式，次小句≈实体，倒数第二≈动作，末小句≈后果
            cls = clauses(sent)
            if len(cls) >= 3 and len(sent) >= 20:
                picks = {"hook": cls[0], "entity": cls[1],
                         "action": cls[-2], "consequence": cls[-1]}
                for slot, clause_text in picks.items():
                    if 4 <= len(clause_text) <= 60:
                        offset_samples[slot].append(_style_of(clause_text)[1] - phase)

            # 句法骨架：按句式关键词计数，存第一条真实匹配句与出处
            for kind, pattern in SKELETON_PATTERNS.items():
                if pattern.search(sent):
                    rec = skeleton_hits[kind]
                    rec["freq"] += 1
                    rec["phases"].append(phase)
                    if len(rec["provenance"]) < 2:
                        rec["provenance"].append(f"{src_tag}｜{sent[:60]}")

            # P8：因果标记后面的分句也是后果候选（"…，所以温度升高"）
            extra_cons: List[str] = []
            if p8:
                for marker in P8_CAUSAL_MARKERS:
                    if marker in sent:
                        extra_cons.extend(clauses(sent.split(marker, 1)[1]))

            for ci, clause in enumerate(cls):
                # 实体：按虚词切段后的完整名词短语，再按动词边界切一次
                segs = re.split("[" + "".join(ENTITY_FUNCTION_CHARS | ENTITY_BREAK_CHARS) + "]", clause)
                pieces: List[str] = []
                for seg in segs:
                    seg = seg.strip()
                    if not seg:
                        continue
                    pieces.append(seg)
                    for seed in VERB_SEEDS:
                        if seed in seg[1:]:
                            pieces.extend(seg.split(seed))
                tokens = set()
                for piece in pieces:
                    w = _trim(piece)
                    if p8:
                        if (not entity_tier(w) or w in GENERIC_WORDS
                                or w in VERB_WORDS):
                            continue
                    elif not _usable(w, 2, 6) or w in GENERIC_WORDS or w in VERB_WORDS:
                        continue
                    tokens.add(w)
                    entity_counts[w] += 1
                    entity_meta.setdefault(w, (style, phase, clause[:50]))
                    if src_tag not in entity_src[w] and len(entity_src[w]) < 3:
                        entity_src[w].append(src_tag)
                    # 域统计：同分句共现
                    for m in ("仪器", "测量", "粒子", "原子", "电子", "能量", "实验", "光谱", "磁场", "力",
                              "看见", "摸", "像", "比如", "想象", "吃饭", "走路", "家里", "朋友", "手", "眼睛"):
                        if m in clause:
                            entity_ctx[w][m] += 1

                # 动作：动词种子 + 紧邻补语（去虚词尾巴）
                for seed in VERB_SEEDS:
                    idx = clause.find(seed)
                    if idx >= 0:
                        w = seed + clause[idx + len(seed):idx + len(seed) + 2]
                        while w and w[-1] in FUNCTION_CHARS | set("一两个几"):
                            w = w[:-1]
                        ok = (is_clean_action(w) if p8 else (
                            _usable(w, 2, 6) and not any(ch in FUNCTION_CHARS for ch in w)
                            and not w.endswith(("者", "器", "方程", "时", "面"))))
                        if ok:
                            action_counts[w] += 1
                            action_order.setdefault(w, "pre" if seed in TRANSITIVE_SEEDS else "post")
                            action_meta.setdefault(w, (style, phase, clause[:50]))
                            if src_tag not in action_src[w] and len(action_src[w]) < 3:
                                action_src[w].append(src_tag)

                # 后果：分句尾部、落在可观测变化上，不能是条件从句
                if p8:
                    # 只看末两个分句：句首分句多是起手式/铺垫，不是后果
                    if ci >= len(cls) - 2:
                        w = _clean_consequence(_trim_tail(clause))
                        if _p8_consequence_ok(w):
                            cons_counts[w] += 1
                            cons_meta.setdefault(w, (style, phase, clause[:50]))
                            if src_tag not in cons_src[w] and len(cons_src[w]) < 3:
                                cons_src[w].append(src_tag)
                elif any(m in clause for m in OBSERVABLE_MARKERS):
                    w = _clean_consequence(_trim(clause)) if len(_trim(clause)) >= 4 else ""
                    w_ok = (
                        _usable(w, 4, 12)
                        and not w.startswith(CONSEQUENCE_HEAD_STOP)
                        and _consequence_ok(w)
                    )
                    if w_ok:
                        cons_counts[w] += 1
                        cons_meta.setdefault(w, (style, phase, clause[:50]))
                        if src_tag not in cons_src[w] and len(cons_src[w]) < 3:
                            cons_src[w].append(src_tag)

                # 跨域纠缠对不再在这里收：那是分句里的候选名词，
                # 与槽位最终收的字粒不是一回事，打分配对永远对不上。
                # 改由 _entangled_from_cooc 在共现建好后从 E→C 边生成。

            if p8:
                for clause in extra_cons:
                    w = _clean_consequence(_trim_tail(clause))
                    if _p8_consequence_ok(w):
                        cons_counts[w] += 1
                        cons_meta.setdefault(w, (style, phase, clause[:50]))
                        if src_tag not in cons_src[w] and len(cons_src[w]) < 3:
                            cons_src[w].append(src_tag)

    vocab: Dict[str, List[dict]] = {}
    if p8:
        # 切半/多带前缀的去重："量增加"、"脑细胞一定也有变化" 都只留更完整的那条
        cons_words = list(cons_counts)
        for w in cons_words:
            if any(z != w and 0 < len(z) - len(w) <= 3 and z.endswith(w)
                   and cons_counts[z] >= cons_counts[w] for z in cons_words
                   if z != w):
                del cons_counts[w]
    entity_gate = entity_tier if p8 else (lambda w: 1 if purify_entity_word(w) else 0)
    action_gate = is_clean_action if p8 else purify_action_word
    cons_gate = _p8_consequence_ok if p8 else None
    for slot, counter, src_map, meta_map, min_count, min_breadth, domain_fn, purifier in (
        ("hook", hook_counts, hook_src, hook_meta, 3, 2, None, None),
        ("entity", entity_counts, entity_src, entity_meta, 5, 3,
         lambda w: _entity_domain(w, entity_ctx), entity_gate),
        ("action", action_counts, action_src, action_meta, 3, 2, None, action_gate),
        ("consequence", cons_counts, cons_src, cons_meta, 2, 2, lambda w: "physics", cons_gate),
    ):
        max_count = max(counter.values()) if counter else 1
        slot_limit = int((slot_limits or {}).get(slot, limit_per_slot))
        items = []
        tier2: List[dict] = []
        for word, count in counter.most_common(slot_limit * 8):
            if count < min_count or len(src_map.get(word, [])) < min_breadth:
                continue
            tier = 1
            if purifier is not None:
                verdict = purifier(word)
                if verdict is True:
                    tier = 1
                elif verdict is False or not verdict:
                    continue
                else:
                    tier = int(verdict)
            style, phase, example = meta_map.get(word, (STYLE_LECTURE, 20.0, ""))
            amp = round(0.94 + 0.06 * math.log1p(count) / math.log1p(max_count), 3)
            item = Item(
                word=word, slot=slot, count=int(round(count, 1)), amp=amp,
                domain=domain_fn(word) if domain_fn else "everyday",
                style=style, phase_deg=phase, sources=src_map.get(word, []), example=example,
            ).to_dict()
            if slot == "entity":
                item["tier"] = tier
                if tier == 2:
                    # 第二梯队只占配额，不挤第一梯队；概率权降一档，锚点仍先选具体物
                    item["amp"] = round(amp * 0.85, 3)
                    if len(tier2) < TIER2_ENTITY_SLOTS:
                        tier2.append(item)
                    continue
            items.append(item)
            if len(items) >= slot_limit:
                break
        vocab[slot] = items + tier2 if slot == "entity" else items

    if consequence_inventory:
        inv = json.loads(Path(consequence_inventory).read_text(encoding="utf-8"))
        cons_items = []
        for it in inv:
            if it.get("slot") != "consequence" or not it.get("word"):
                raise ValueError(f"后果库存条目非法：{it!r}")
            cons_items.append(it)
        if cons_items:
            # 整槽替换：策展库存才是这个模式下后果槽的完整定义
            vocab["consequence"] = cons_items

    # 动作的语序：及物动词放实体前（"观察电子"），其余放实体后（"电子旋转"）
    for it in vocab["action"]:
        it["order"] = action_order.get(it["word"], "post")
        it["valency"] = classify_action_valency(it["word"], it["order"])

    # 实测偏移：样本足够才信数据，否则回退手编常量
    slot_offsets = {}
    for slot, default in DEFAULT_SLOT_OFFSETS.items():
        samples = offset_samples[slot]
        if len(samples) >= 50:
            mean = sum(samples) / len(samples)
            slot_offsets[slot] = round(max(-25.0, min(25.0, mean)), 1)
        else:
            slot_offsets[slot] = default
    # 强制单调：起手式在前、后果在后，避免样本噪声把弧翻过来
    slot_offsets["hook"] = min(slot_offsets["hook"], slot_offsets["entity"] - 2.0)
    slot_offsets["action"] = max(slot_offsets["action"], slot_offsets["entity"] + 2.0)
    slot_offsets["consequence"] = max(slot_offsets["consequence"], slot_offsets["action"] + 2.0)

    skeletons = []
    for kind, template in SKELETON_DEFS:
        rec = skeleton_hits[kind]
        if kind != "declarative" and rec["freq"] < 3:
            continue
        phases = rec["phases"] or [45.0]
        provenance = rec["provenance"][:2] or ["通用陈述框（语料高频四段结构，无单一匹配句）"]
        skeletons.append({
            "pattern_type": kind,
            "template": template,
            "freq": rec["freq"],
            "phase_deg": round(sum(phases) / len(phases), 1),
            "provenance": provenance,
        })

    # 第二遍：同句 ±5 字窗口的字粒共现，只留实词对
    cooc = _build_cooccurrence(corpus_path, vocab, p8=p8, trusted=trusted)

    # 语料外书籍补边：只加不减，键必须落在已入库的字粒上
    if edge_supplement is not None:
        ent_words = {g["word"] for g in vocab["entity"]}
        act_words = {g["word"] for g in vocab["action"]}
        cons_words = {g["word"] for g in vocab["consequence"]}
        known = ent_words | act_words | cons_words
        sup = json.loads(Path(edge_supplement).read_text(encoding="utf-8"))
        for pair, delta in sup.items():
            if "|" not in pair:
                raise ValueError(f"edge_supplement 键须为 实体|后果：{pair!r}")
            a, b = pair.split("|", 1)
            if a not in known or b not in cons_words:
                raise ValueError(f"edge_supplement 指向未入库字粒：{pair!r}")
            if not isinstance(delta, int) or isinstance(delta, bool) or delta <= 0:
                raise ValueError(f"edge_supplement 次数须为正整数：{pair!r}={delta!r}")
            cooc[pair] = cooc.get(pair, 0) + delta

    # 第二梯队实体只留"真有搭配"的：至少 TIER2_MIN_SPECIFIC_EDGES 个非相位动作同句共现，
    # 否则"材料""原理"这类只是占位，不配进槽
    if p8 and vocab["entity"]:
        kept = []
        for it in vocab["entity"]:
            if it.get("tier", 1) == 1:
                kept.append(it)
                continue
            specific = sum(
                1 for a in vocab["action"]
                if a["word"] not in PHASE_ACTION_WORDS
                and cooc.get(f"{it['word']}|{a['word']}", 0) > 0
            )
            if specific >= TIER2_MIN_SPECIFIC_EDGES:
                it["specific_actions"] = specific
                kept.append(it)
        vocab["entity"] = kept

    vault = {
        "corpus": str(corpus_path),
        "sections": sections,
        "sentences": sents_total,
        "slot_offsets": slot_offsets,
        "skeletons": skeletons,
        "hooks": vocab["hook"],
        "entities": vocab["entity"],
        "actions": vocab["action"],
        "consequences": vocab["consequence"],
        "entangled_pairs": _entangled_from_cooc(cooc, vocab),
        "cooccurrence": cooc,
    }
    return vault


def _entangled_from_cooc(cooc: Dict[str, int], vocab: Dict[str, List[dict]],
                         top: int = 80, min_count: int = 3) -> List[dict]:
    """纠缠对 = 语料里真的同句出现过的（实体, 后果），与打分配对的键完全一致。

    历史版本用的是分句候选名词对，和槽位字粒不是同一批，80 对里 0 对能命中
    `(entity, consequence)`，加成从未生效。字段名 everyday/physics 沿用旧格式，
    语义改为实体词/后果词；按共现次数取前 top 条。
    """
    ent = {g["word"] for g in vocab.get("entity", [])}
    cons = {g["word"] for g in vocab.get("consequence", [])}
    merged: Dict[Tuple[str, str], int] = defaultdict(int)
    for key, n in cooc.items():
        a, _, b = str(key).partition("|")
        if not a or not b:
            continue
        if a in ent and b in cons:
            merged[(a, b)] += int(n)
        elif b in ent and a in cons:
            merged[(b, a)] += int(n)
    pairs = sorted(((a, b, n) for (a, b), n in merged.items()),
                   key=lambda x: (-x[2], x[0], x[1]))
    return [{"everyday": a, "physics": b, "count": n}
            for a, b, n in pairs[:top] if n >= min_count]


def _build_cooccurrence(corpus_path: Path, vocab: Dict[str, List[dict]],
                        limit: int = 8000, window: int = 5,
                        p8: bool = False, trusted: bool = False) -> Dict[str, int]:
    """entity×action / entity×consequence / action×consequence。

    旧库：同句 ±window 字内的无向共现。
    P8：E→A、E→C 仍按同句距离；A→C 只认**因果标记两侧的谓词对**——
    动作在标记前的紧邻分句、后果是标记后的第一分句，否则不算"动作导致后果"。
    """
    ent = [g["word"] for g in vocab["entity"]]
    act = [g["word"] for g in vocab["action"]]
    cons = [g["word"] for g in vocab["consequence"]]
    if not (ent and act and cons):
        return {}
    ent_set, act_set, cons_set = set(ent), set(act), set(cons)
    counts: Counter = Counter()
    for _src, _section, text in iter_sections(corpus_path, trusted=trusted):
        for sent in sentences(text):
            positions: Dict[str, int] = {}
            for word in ent + act + cons:
                idx = sent.find(word)
                if idx >= 0:
                    positions[word] = idx
            if len(positions) < 2:
                continue
            if not p8:
                ents = [(w, i) for w, i in positions.items() if w in ent_set]
                for a, i in ents:
                    for b, j in positions.items():
                        if b in (a,) or b not in act_set and b not in cons_set:
                            continue
                        if abs(j - i) <= window + 2:
                            counts[tuple(sorted((a, b)))] += 1
                continue
            # P8：按槽位方向记边，级联打分要的是 P(动作|实体)、P(后果|动作)
            found_e = [(w, i) for w, i in positions.items() if w in ent_set]
            found_a = [(w, i) for w, i in positions.items() if w in act_set]
            found_c = [(w, i) for w, i in positions.items() if w in cons_set]
            for e, ei in found_e:
                for a, ai in found_a:
                    if abs(ai - ei) <= window + 2:
                        counts[(e, a)] += 1
                for c, ci in found_c:
                    if abs(ci - ei) <= 60:  # 同句、宽窗：实体到后果通常跨小句
                        counts[(e, c)] += 1
            # A→C 两级权重：同句宽窗是弱边（+1），因果标记两侧的谓词对是强边（再 +2）。
            # 只认因果对会让边数掉到 1（这份语料标记句里能过门的后果只有 89 条），
            # 所以弱边保留覆盖、强边负责排序，条件提升比会偏向真因果。
            for a, ai in found_a:
                for c, ci in found_c:
                    if abs(ci - ai) <= 60:
                        counts[(a, c)] += 1
            for marker in P8_CAUSAL_MARKERS:
                idx = sent.find(marker)
                if idx < 0:
                    continue
                pre, post = sent[:idx], sent[idx + len(marker):]
                pre_tail = (clauses(pre) or [pre])[-1]
                post_clause = (clauses(post) or [post])[0]
                acts_here = [w for w in act if w in pre_tail]
                if not acts_here:
                    continue
                cons_here = [w for w in cons if w in post_clause]
                for a in acts_here:
                    for c in cons_here:
                        counts[(a, c)] += 2
    # 只保留下限以上的对，避免文件膨胀；"从未共现"由缺席表示
    kept = [(pair, n) for pair, n in counts.most_common(limit) if n >= (1 if p8 else 2)]
    return {f"{a}|{b}": int(n) for (a, b), n in kept}


def dump_vault(vault: dict, path: Path = VAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vault, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    p8 = "--p8" in sys.argv
    trusted = "--trusted" in sys.argv
    corpus = CORPUS_PATH
    if "--corpus" in sys.argv:
        corpus = Path(sys.argv[sys.argv.index("--corpus") + 1])
    out = VAULT_PATH
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
    v = build_vault(corpus_path=corpus, p8=p8, trusted=trusted)
    p = dump_vault(v, out)
    print(f"✅ 铸字完成：{p}")
    print(f"   段落 {v['sections']}，句子 {v['sentences']}")
    print(f"   槽位偏移 {v['slot_offsets']}")
    print(f"   骨架 {len(v['skeletons'])} 条，共现对 {len(v['cooccurrence'])} 条")
    for slot in ("hooks", "entities", "actions", "consequences"):
        print(f"   {slot}: {len(v[slot])} 颗")
        print("     ", [h["word"] for h in v[slot][:8]])
