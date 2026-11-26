"""费曼心智活字：四槽位相位排版（起手式 × 实体 × 动作 × 可观测后果）。

与原版量子活字印刷机的区别：
- 字粒不是现场从任意语料抽的，而是 `feynman_vault` 从费曼本人语料里铸好的，每颗带出处；
- 槽位从三扩到四，相位轴是"语体"：约 15° 讲义、60° 对话、105° 演讲／哲理；
- 槽位偏移不再手编：字库里 `slot_offsets` 是逐句量出来的（首/次/倒数第二/末小句）；
- 四道闸门：术语黑名单（taste_discriminants 的 contempt）、跨域纠缠加成、
  语体相干度、句法骨架（骨架来自语料，带出处与频次）；
- 五个心智算子（微观拆解 / 跨界类比 / 极端边界 / 时间反演 / 反直觉）0–2 可调，
  默认全 0 = 基线行为，乘数只做路径间的差异放大；
- 证据不足时印不出字，只落一颗留白字粒，并说明缺什么。

排版仍是确定性采样：seed = sha256(bank_id|style_deg|nonce)。
"""

import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba.posseg as pseg
from pydantic import BaseModel, Field

from plugins.feynman_vault import (
    DISCOURSE_STOP,
    GENERIC_WORDS,
    META_DISCOURSE_NOUNS,
    _posseg,
)

VAULT_PATH = Path(os.getenv("QBOT_FEYNMAN_VAULT", "data/feynman_style_vault.json"))
KNOWLEDGE_DIR = Path(
    os.getenv("QBOT_FEYNMAN_KNOWLEDGE", str(Path.home() / "Projects/feynman-open/knowledge"))
)

SLOTS = ("hook", "entity", "action", "consequence")
SLOT_LABELS = {"hook": "起手式", "entity": "实体", "action": "动作", "consequence": "后果"}
SLOT_KEYS = {"hook": "hooks", "entity": "entities", "action": "actions", "consequence": "consequences"}
# 兜底偏移：字库里没有实测值时用；起手式略早于旋钮，后果略晚，形成一条"开口朝说话人"的弧
SLOT_PHASE_OFFSET = {"hook": -12.0, "entity": 0.0, "action": 10.0, "consequence": 20.0}
# 句式在语体弧上的目标位置：低相位讲义 → 中相位对话 → 高相位演讲/回溯
SKELETON_TARGET_PHASE = {
    "conditional": 15.0, "causal": 25.0, "contrast": 35.0,
    "declarative": 55.0, "temporal": 68.0, "analogy": 92.0, "retro": 110.0,
}
SLOT_AMP = {"hook": 0.92, "entity": 1.0, "action": 0.86, "consequence": 0.9}
DEFAULT_TEMPLATE = "{hook}，{entity}{action}，{consequence}。"

STYLE_BANDS = (
    (0.0, 40.0, "讲义", "像在黑板前，把现象一步步摆出来"),
    (40.0, 80.0, "对话", "像坐在桌边，用日常东西打比方"),
    (80.0, 120.0, "演讲", "像在礼堂里，把话说到哲学那一层"),
)

# 术语闸门：从品味判据的 contempt 里自动抽，再补一小撮手工词。
# 这里的词不是"禁词"，而是"只报名字不作解释"的信号，扣权不封杀。
DEFAULT_JARGON = (
    "范式", "表征", "耦合", "赋能", "抓手", "闭环", "底层逻辑", "方法论", "认知维度",
    "重构", "对齐", "涌现", "元认知", "系统性", "结构性", "机制性", "话语体系",
)
BLANK_GLYPH = "【此处没有可测的东西】"

# 问句动词里的"提问方式"：可以注入动作槽，但不拿来做坍缩偏好，
# 否则"磁场能看见吗"会印成"磁场看见"。
PERCEPTION_VERBS = {
    "看见", "看到", "听懂", "理解", "明白", "知道", "认识", "觉得", "感觉",
    "发现", "解释", "描述", "观察", "研究", "讨论", "记得", "想", "说",
}

# 五个心智算子：默认 0 = 完全等于基线；预设才把权重拉起来。
# 每个算子只回答一件事："这条路径在多大程度上像这种想问题的方式"，分数 ∈[0,1]。
class OperatorWeights(BaseModel):
    micro: float = Field(0.0, ge=0.0, le=2.0)
    cross: float = Field(0.0, ge=0.0, le=2.0)
    boundary: float = Field(0.0, ge=0.0, le=2.0)
    reversal: float = Field(0.0, ge=0.0, le=2.0)
    anti_intuition: float = Field(0.0, ge=0.0, le=2.0)
    jargon_strip: float = Field(0.0, ge=0.0, le=2.0)
    least_action: float = Field(0.0, ge=0.0, le=2.0)
    cargo_cult: float = Field(0.0, ge=0.0, le=2.0)


OPERATOR_LABELS = {
    "micro": "微观拆解",
    "cross": "跨界类比",
    "boundary": "极端边界",
    "reversal": "时间反演",
    "anti_intuition": "反直觉",
    "jargon_strip": "剥掉术语",
    "least_action": "最小作用量",
    "cargo_cult": "货物崇拜检验",
}
OPERATOR_PRESETS: Dict[str, Dict[str, float]] = {
    "均衡费曼全脑": {"micro": 1.0, "cross": 1.0, "boundary": 1.0, "reversal": 0.5, "anti_intuition": 0.8},
    "黑板物理推演": {"micro": 1.6, "cross": 0.3, "boundary": 1.2, "reversal": 0.4, "anti_intuition": 0.6},
    "酒吧科普神比喻": {"micro": 0.8, "cross": 1.8, "boundary": 0.6, "reversal": 0.2, "anti_intuition": 1.0},
    "侦探由果溯因": {"micro": 0.6, "cross": 0.6, "boundary": 0.8, "reversal": 1.8, "anti_intuition": 1.2},
    "剥掉术语": {"micro": 0.6, "cross": 0.8, "jargon_strip": 1.8},
    "最小作用量": {"micro": 0.8, "boundary": 0.8, "least_action": 1.8},
    "货物崇拜检验": {"reversal": 0.8, "anti_intuition": 0.6, "cargo_cult": 1.8},
}
BOUNDARY_TOKENS = (
    "假设", "假如", "如果", "想象", "设想", "无穷", "无限", "永远", "从不", "绝对",
    "极限", "光速", "绝对零度", "全部", "最大", "最小", "极端", "零",
)

# --- 用户实体注入口 ---------------------------------------------------------
# 用户问句里的名词不是语料铸出来的字粒，但排版必须落到他问的那个东西上。
# 只收名词性、非元话语、非术语的词；最多 3 个，按出现顺序。
USER_ENTITY_STOP = set(META_DISCOURSE_NOUNS) | set(GENERIC_WORDS) | set(DISCOURSE_STOP) | {
    "什么", "怎么", "为什么", "如何", "东西", "事情", "时候", "地方", "一下", "一直",
    "可以", "可能", "应该", "必须", "已经", "正在", "我们", "你们", "他们", "它们",
    "怎么办", "怎么样", "怎么回事", "是什么", "咋办", "怎样", "多少", "是否",
    "时会",  # "测量电子时会发生什么"：jieba 把"时+会"并成一个 n 词，抽出来是残片
    # 招呼语/客套话不是实体：问"你好，你是谁"时不能拿"你好"去排版
    "你好", "您好", "谢谢", "再见", "请问", "麻烦", "老师", "朋友",
}
# 抽取残片：情态词打头的两字片段（"会生""能变"）是"情态+动词"被切开的产物
_LEAD_MODALS = "会能要想该应可得了"
_VERB_CHARS = set("生热说做去来看走变叫成到给动转停消增念吃喝烧冻浮沉吸排开合好坏")
def _junk_fragment(word: str) -> bool:
    if len(word) == 2 and word[0] in _LEAD_MODALS and word[1] in _VERB_CHARS:
        return True
    return word.startswith(("的", "地", "得"))
# jieba 把一批物理过程名词标成 vn（摩擦/振动/辐射），它们正是问句最该抓的实体
VN_ENTITY_WHITELIST = {
    "摩擦", "振动", "振荡", "碰撞", "膨胀", "收缩", "流动", "辐射", "加热", "冷却",
    "燃烧", "弯曲", "衰变", "折射", "反射", "衍射", "发光", "发热", "导电", "导热",
    "对流", "蒸发", "沸腾", "凝固", "熔化", "磁化", "电离", "聚变", "裂变", "传导",
}
# 单字物理名词也常是问句主语（"冰为什么会浮"），只认这一小张表，不放开单字抽取
SINGLE_CHAR_ENTITY = set("冰水光电磁力风声波气核")


# --- 用户动词注入口 ---------------------------------------------------------
# 问句的谓语（发光/吸/浮）才是"要说的事"；语料 60 个动作字里没有时，直接拿它当动作字。
# 系动词、情态词、认知词不收——"天空为什么是蓝的"不能印成"天空是"。
VERB_STOP = {
    "是", "会", "能", "要", "想", "有", "在", "去", "来", "说", "做", "看", "让", "被",
    "给", "使", "用", "把", "知道", "觉得", "认为", "发现", "相信", "希望", "需要",
    "应该", "可以", "能够", "可能", "变得", "变成", "成为", "发生", "存在", "属于",
    "叫作", "称为", "如果", "假设", "解释", "描述", "问", "帮", "麻烦",
}
# 单字过程动词白名单："吸铁"被 jieba 整体标成 n，按首字拆出来
SINGLE_CHAR_VERB = set("吸浮沉烧冻转推拉碰撞滑滚跳飞流亮暗响碎弯弹漂化熔凝蒸沸落升降涨跌飘晃震扭挤压吹")
# 过程动词小词表：jieba 偶尔把连写切开（"会生热"→"会生|热"），词表扫描兜底
QUERY_VERB_LEXICON = (
    "发光", "发热", "生热", "散热", "导电", "导热", "膨胀", "收缩", "融化", "熔化",
    "蒸发", "沸腾", "凝固", "溶解", "燃烧", "生锈", "带电", "放电", "折射", "反射",
    "衍射", "漂浮", "下沉", "上升", "下降", "转动", "震动", "滚动", "滑动", "碰撞",
    "摩擦", "振动", "爆炸", "辐射", "冷却", "加热", "磁化", "电离", "衰变",
)


def extract_query_verbs(query: str, exclude: Optional[set] = None,
                        limit: int = 2) -> List[str]:
    """抽问句里可当动作字印的过程动词（最多 limit 个，按出现顺序）。

    exclude 给字库已有的动作词与已抽出的实体词：重复注入没有意义，
    实体词当动作会印成"摩擦摩擦"。
    """
    q = (query or "").strip()
    if not q:
        return []
    exclude = exclude or set()
    pseg = _posseg()
    if not pseg:
        return []
    # (位置, 动词)：先扫过程动词小词表——jieba 会把"会生热"切成"会生|热"，
    # 词表扫描不受分词影响；其余交给词性
    found: List[tuple] = []
    for v in QUERY_VERB_LEXICON:
        i = q.find(v)
        if i >= 0:
            found.append((i, v))
    cursor = 0
    for word, flag in pseg.cut(q):
        pos, cursor = cursor, cursor + len(word)
        word = word.strip()
        if not word:
            continue
        verb = ""
        if flag.startswith("v") and not _junk_fragment(word):
            verb = word
        elif flag == "n" and 2 <= len(word) <= 4 and word[0] in SINGLE_CHAR_VERB:
            verb = word[0]          # "吸铁" → 吸
        if verb:
            found.append((pos, verb))
    out: List[str] = []
    for _, verb in sorted(found):
        if len(verb) > 4 or verb in VERB_STOP or verb in exclude:
            continue
        if len(verb) == 1 and verb not in SINGLE_CHAR_VERB:
            continue
        if verb not in out:
            out.append(verb)
        if len(out) >= limit:
            break
    return out


# --- 招呼语模板 -------------------------------------------------------------
# 纯招呼/自报家门不进语料排版：拿"你好"去铸字只会印出与对话无关的物理句。
GREETING_REPLY = (
    "你好。我是费曼——一台拿费曼的字排版的小机器。"
    "你随便问一个物理问题，我就试着用活字拼一句。"
)
IDENTITY_REPLY = (
    "我是费曼的活字印刷机：字粒全部铸自语料，句子是拼出来的，不是编出来的。"
    "问一个物理问题试试，比如'冰为什么会浮在水面上'。"
)
_GREETINGS = {"你好", "您好", "嗨", "哈喽", "喂", "在吗", "早上好", "上午好", "中午好",
              "下午好", "晚上好", "晚安", "hello", "hi", "hey"}
_IDENTITY_ASKS = {"你是谁", "你是哪位", "你叫什么", "你叫什么名字", "你是干什么的",
                  "你能做什么", "你会什么"}


def detect_smalltalk(query: str) -> Optional[str]:
    """整句只是招呼/自报家门的询问时返回模板回复；其余返回 None。"""
    q = (query or "").strip().strip("，。！？!?,.~～ \t")
    q = q.rstrip("啊呀哈嘞呢吧")
    if not q:
        return None
    if q.lower() in _GREETINGS:
        return GREETING_REPLY
    if q in _IDENTITY_ASKS:
        return IDENTITY_REPLY
    return None
# 原型映射：问句里的具体物先归到语料里常见的原型，再看原型和动作有没有共现。
# 没有原型也能用，只是共现证据更少。
PROTOTYPE_MAP = {
    "汽车": "车", "卡车": "车", "火车": "车", "自行车": "车", "公交车": "车", "飞机": "机器",
    "轮船": "船", "杯子": "容器", "瓶子": "容器", "箱子": "容器", "盒子": "容器",
    "水": "液体", "咖啡": "液体", "茶": "液体", "空气": "气体", "猫": "动物",
    "狗": "动物", "鸟": "动物", "鱼": "动物", "人": "人", "孩子": "人", "手机": "仪器",
    "电脑": "仪器", "钟表": "钟表", "钟": "钟表", "镜子": "镜", "灯泡": "灯",
}
# 和几乎任何实体都讲得通的动作：没有共现证据也不扣权
SAFE_UNIVERSAL_ACTIONS = {
    "观察", "测量", "看见", "想象", "记住", "吸引", "推动", "拉动", "旋转", "穿过",
    "出现", "变成", "发光", "发热", "振动", "碰撞", "聚集", "散开", "跳动", "停止",
}

# --- 抽象实体原型路由（P6.2）------------------------------------------------
# 金融/信息/规则类名词没有物理外形，语料里也不会有"股票旋转"这种共现。
# 只按词表归类，不猜语义；归不进去仍走物理兜底。
ABSTRACT_PROTOTYPE_RULES: Dict[str, Tuple[str, ...]] = {
    "abstract_finance": (
        "股票", "股价", "期权", "债券", "基金", "回购", "市值", "利率", "汇率", "通胀",
        "通货膨胀", "债务", "贷款", "房价", "工资", "价格", "利润", "成本", "投资", "分红",
        "估值", "现金流", "资产", "负债", "货币", "银行", "保险", "税收",
    ),
    "abstract_info": (
        "算法", "数据", "程序", "模型", "代码", "网络", "信息", "信号", "密码", "协议",
        "数据库", "软件", "系统", "文件", "语言", "记忆", "知识", "公式", "方程", "证明",
        "逻辑", "推理", "搜索", "排序", "加密", "互联网", "人工智能", "机器学习",
    ),
    "abstract_rule": (
        "法律", "规则", "制度", "合同", "政策", "税", "产权", "契约", "标准", "规定",
        "条例", "判决", "权利", "义务", "责任", "秩序", "道德", "惯例", "宪法", "条款",
    ),
}
# 只有有形状、能受力、能传热的东西才做得的动作。抽象实体碰上这些，直接压到近零。
PHYSICAL_ONLY_ACTIONS = {
    "推动", "拉动", "旋转", "碰撞", "加热", "冷却", "压缩", "拉伸", "弯曲", "熔化",
    "蒸发", "发光", "发热", "振动", "穿过", "吸引", "掉落", "滚动", "滑动", "摩擦",
    "悬浮", "流动", "膨胀", "收缩", "燃烧", "冻结", "沸腾", "破裂", "弹开", "弹起",
    "晃动", "振荡", "排斥", "聚集", "散开", "跳动",
}
# 对抽象实体仍然讲得通的动作：看、量、算、拆、比。字库里真有的才会被选进候选。
SAFE_COGNITIVE_ACTIONS = {"观察", "计算", "拆解", "测量", "看清", "理解", "比较", "检验", "记住", "看见"}


def resolve_entity_prototype(word: str,
                             overrides: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """返回 (原型族, 原型词)。物理物走 PROTOTYPE_MAP，抽象词走规则表，其余兜底。

    `overrides` 来自扩展库的 entity_prototypes（爬虫/人工标注），优先级最高；
    空字典时与旧行为逐字节一致。
    """
    if overrides and word in overrides:
        family = overrides[word]
        if family in ABSTRACT_PROTOTYPE_RULES:
            return family, family
        return family, PROTOTYPE_MAP.get(word, "")
    if word in PROTOTYPE_MAP:
        return "physical", PROTOTYPE_MAP[word]
    for family, words in ABSTRACT_PROTOTYPE_RULES.items():
        if word in words:
            return family, family
    return "default_physical", ""


EMPTY_EXT_HASH = "00000000"


def _seed(bank_id: str, style_deg: float, nonce: int, session: str = "",
          ext_hash: str = "") -> str:
    """无会话、无扩展库时与旧版逐字节一致；同会话同 nonce 仍确定。

    ext_hash 只在扩展库非空时插入种子链：学到新词后输出整体换快照，
    但删掉扩展库立刻回到原基线，不需要重算。
    """
    if ext_hash and ext_hash != EMPTY_EXT_HASH:
        tag = f"{bank_id}|{ext_hash}|{style_deg:.1f}|{int(nonce)}"
    else:
        tag = f"{bank_id}|{style_deg:.1f}|{int(nonce)}"
    if session:
        tag += f"|{session}"
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _style_band(style_deg: float) -> Tuple[str, str]:
    for lo, hi, name, desc in STYLE_BANDS:
        if lo <= style_deg <= hi:
            return name, desc
    return ("演讲" if style_deg > 80 else "讲义"), ""


def load_jargon_terms() -> List[str]:
    terms = list(DEFAULT_JARGON)
    path = KNOWLEDGE_DIR / "taste_discriminants.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for tag in rec.get("tags", []):
                if tag in {"术语", "行话"}:
                    for word in str(rec.get("contempt", "")).replace("“", " ").replace("”", " ").split():
                        w = word.strip("，。、；：（）()\"'")
                        if 2 <= len(w) <= 6:
                            terms.append(w)
    return sorted(set(terms))


class Vault:
    """已铸好的费曼字库。缺文件时抛 FileNotFoundError，由调用方转成 503。"""

    def __init__(self, path: Path = VAULT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"费曼字库不存在：{path}，先运行 plugins/feynman_vault.py")
        self.raw = json.loads(path.read_text(encoding="utf-8"))
        self.bank_id = "feynman-" + hashlib.sha256(
            json.dumps(self.raw, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()[:12]
        self.by_slot: Dict[str, List[dict]] = {
            s: list(self.raw.get(SLOT_KEYS[s], [])) for s in SLOTS
        }
        offsets = self.raw.get("slot_offsets") or {}
        self.slot_offsets = {
            s: float(offsets.get(s, SLOT_PHASE_OFFSET[s])) for s in SLOTS
        }
        self.skeletons = [
            sk for sk in self.raw.get("skeletons", [])
            if isinstance(sk, dict) and sk.get("template")
        ]
        self.cooc: Dict[str, int] = {
            str(k): int(v) for k, v in (self.raw.get("cooccurrence") or {}).items()
        }
        self.jargon = load_jargon_terms()
        self.entangled = self._load_entangled(self.raw.get("entangled_pairs", []))
        self._cond_tables = None

    @staticmethod
    def _load_entangled(default: list) -> list:
        """默认用字库自带的纠缠对；实验时可指向重算的槽位内配对表。

        QBOT_ENTANGLED_PAIRS=path：不改字库文件、不改 bank_id，只为消融实验。
        文件内容须为 [{"everyday": 实体, "physics": 后果, "count": n}, ...]。
        """
        override = os.getenv("QBOT_ENTANGLED_PAIRS")
        if not override:
            return default
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"纠缠对覆盖文件不存在：{path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"纠缠对覆盖文件须为列表：{path}")
        return data

    def pair_support(self, a: str, b: str) -> int:
        return self.cooc.get(f"{a}|{b}", 0) + self.cooc.get(f"{b}|{a}", 0)

    def cooc_count(self, a: str, b: str) -> int:
        """有向共现次数（与 pair_support 不同：级联打分要方向）。"""
        return self.cooc.get(f"{a}|{b}", 0)

    def _cond_stats(self) -> dict:
        """P8 条件表：P(动作|实体)、P(后果|动作) 的计数与边缘分布（Laplace 平滑用）。"""
        if self._cond_tables is not None:
            return self._cond_tables
        ent = {g["word"] for g in self.by_slot["entity"]}
        act = {g["word"] for g in self.by_slot["action"]}
        cons = {g["word"] for g in self.by_slot["consequence"]}
        ea: Dict[Tuple[str, str], int] = defaultdict(int)
        ac: Dict[Tuple[str, str], int] = defaultdict(int)
        e_tot: Dict[str, int] = defaultdict(int)
        a_tot: Dict[str, int] = defaultdict(int)   # 动作在 E→A 表中的总出现
        a_out: Dict[str, int] = defaultdict(int)   # 动作在 A→C 表中的总出现
        c_tot: Dict[str, int] = defaultdict(int)
        for key, n in self.cooc.items():
            a, _, b = key.partition("|")
            if a in ent and b in act:
                ea[(a, b)] += n; e_tot[a] += n; a_tot[b] += n
            elif a in act and b in ent:
                ea[(b, a)] += n; e_tot[b] += n; a_tot[a] += n
            if a in act and b in cons:
                ac[(a, b)] += n; a_out[a] += n; c_tot[b] += n
            elif a in cons and b in act:
                ac[(b, a)] += n; a_out[b] += n; c_tot[a] += n
        self._cond_tables = {
            "ent": ent, "act": act, "cons": cons, "ea": ea, "ac": ac,
            "e_tot": e_tot, "a_tot": a_tot, "a_out": a_out, "c_tot": c_tot,
            "seen_ea": sum(ea.values()), "seen_ac": sum(ac.values()),
        }
        return self._cond_tables

    def edge_lifts(self, entity: str, action: str, consequence: str) -> Tuple[float, float]:
        """束搜索用的两级提升比（未压平到 [0.6,1.6]）：P(动作|实体)/P(动作) 与 P(后果|动作)/P(后果)。

        按次数排序时，全局高频的"碰撞→粒子与反粒子的碰撞机会增加"会压过一切，
        不管问的是冰还是潮汐；按提升比排序才能让"这个动作/后果对这件事特别"的边站前面。
        """
        t = self._cond_stats()
        mea = mac = 1.0
        if t["seen_ea"] > 0 and t["e_tot"].get(entity, 0) > 0:
            n = t["ea"].get((entity, action), 0)
            p = (n + 1.0) / (t["e_tot"][entity] + len(t["act"]))
            q = (t["a_tot"].get(action, 0) + 1.0) / (t["seen_ea"] + len(t["act"]))
            mea = (p / q) ** 0.5
        if t["seen_ac"] > 0 and t["a_out"].get(action, 0) > 0:
            n = t["ac"].get((action, consequence), 0)
            p = (n + 1.0) / (t["a_out"][action] + len(t["cons"]))
            q = (t["c_tot"].get(consequence, 0) + 1.0) / (t["seen_ac"] + len(t["cons"]))
            mac = (p / q) ** 0.5
        return mea, mac

    def cond_multiplier(self, entity: str, action: str, consequence: str) -> float:
        """P8 级联乘子：动作相对实体、后果相对动作的共现提升，限幅 [0.5,2.0]。

        没有任何证据的一级恰为 1.0（旧字库没有 A→C 边时整项退回中性）。
        用对数提升比 + 平方根压平，避免低频边把乘子推到极端。
        """
        t = self._cond_stats()
        m = 1.0
        if t["seen_ea"] > 0 and t["e_tot"].get(entity, 0) > 0:
            n_ea = t["ea"].get((entity, action), 0)
            p = (n_ea + 1.0) / (t["e_tot"][entity] + len(t["act"]))
            q = (t["a_tot"].get(action, 0) + 1.0) / (t["seen_ea"] + len(t["act"]))
            m *= min(1.6, max(0.6, (p / q) ** 0.5))
        if t["seen_ac"] > 0 and t["a_out"].get(action, 0) > 0:
            n_ac = t["ac"].get((action, consequence), 0)
            p = (n_ac + 1.0) / (t["a_out"][action] + len(t["cons"]))
            q = (t["c_tot"].get(consequence, 0) + 1.0) / (t["seen_ac"] + len(t["cons"]))
            m *= min(1.6, max(0.6, (p / q) ** 0.5))
        return round(min(2.0, max(0.5, m)), 4)

    def stats(self) -> dict:
        return {
            "bank_id": self.bank_id,
            "corpus": self.raw.get("corpus"),
            "sections": self.raw.get("sections"),
            "sentences": self.raw.get("sentences"),
            "glyphs": {s: len(v) for s, v in self.by_slot.items()},
            "slot_offsets": {s: round(v, 1) for s, v in self.slot_offsets.items()},
            "skeletons": [
                {"pattern_type": sk.get("pattern_type"), "phase_deg": sk.get("phase_deg"),
                 "freq": sk.get("freq")}
                for sk in self.skeletons
            ],
            "cooccurrence_pairs": len(self.cooc),
            "entangled_pairs": len(self.entangled),
            "jargon_terms": len(self.jargon),
            "samples": {s: [g["word"] for g in v[:8]] for s, v in self.by_slot.items()},
        }


def _glyph_phase(g: dict) -> float:
    return float(g.get("phase_deg", 20.0))


def _jargon_hits(text: str, terms: List[str]) -> List[str]:
    return [t for t in terms if t in text]


def _coverage(topic: str, glyph: dict) -> bool:
    """话题里有没有这个字粒的影子：双字重叠即算。"""
    if not topic:
        return True
    t = topic.strip()
    w = glyph["word"]
    if w in t or t in w:
        return True
    return any(t[i:i + 2] in w for i in range(len(t) - 1))


def _circular_concentration(phases_deg: List[float], center_deg: float) -> float:
    """语体相干度 R∈[0,1]：四颗字粒的相位是否朝同一语体收拢。"""
    if not phases_deg:
        return 0.0
    xs = sum(math.cos(math.radians(p - center_deg)) for p in phases_deg) / len(phases_deg)
    ys = sum(math.sin(math.radians(p - center_deg)) for p in phases_deg) / len(phases_deg)
    return math.hypot(xs, ys)


def _is_abstract_word(word: str) -> bool:
    return any(word in words for words in ABSTRACT_PROTOTYPE_RULES.values())


def extract_entities(query: str, jargon: Optional[List[str]] = None, limit: int = 3,
                     prefer: Optional[set] = None) -> List[str]:
    """从用户问句里抽可注入的名词实体（最多 limit 个，按出现顺序去重）。

    只用词性 + 黑名单，不猜语义：抽错了最多排版句里出现一个生词，
    不会被当成事实；抽不出来就不注入，退回基线。jieba 缺失时按标点切。
    prefer 给一组"字库里真有的实体词"，命中的排在前面——问句里既有生词又有熟词时，
    让排版落在字库讲得通的那个上。
    """
    q = (query or "").strip()
    if not q:
        return []
    jargon = jargon or []
    out: List[str] = []
    pseg = _posseg()
    if pseg:
        pairs = [(w.strip(), f) for w, f in pseg.cut(q) if w.strip()]
    else:
        # jieba 不可用时词边界仍与右脑一致：共用 lang 规则，而不是按标点整段切
        # （整段切会把"熵到底是什么"当成一个词，实体永远抽不出来）。
        from plugins.word_segmentation import segment
        pairs = [(w, "n") for w in segment(q)]
    for word, flag in pairs:
        if len(word) == 1:
            if word not in SINGLE_CHAR_ENTITY:
                continue
        elif len(word) > 6 or word.isdigit():
            continue
        if word in USER_ENTITY_STOP or any(t in word for t in jargon):
            continue
        if _junk_fragment(word):
            continue
        # 抽象领域词（合同/股票/算法…）常被 jieba 标成 a/v，词表命中就直接收；
        # 物理过程名词（摩擦/振动…）常被标成 vn，同样按词表收
        if not (flag.startswith("n") or flag in ("eng", "nz", "j", "l")
                or word in VN_ENTITY_WHITELIST or _is_abstract_word(word)):
            continue
        if word not in out:
            out.append(word)
        if len(out) >= limit:
            break
    if prefer:
        known = [w for w in out if w in prefer]
        out = known + [w for w in out if w not in prefer]
    return out


def _user_glyph(word: str, style_deg: float, query: str, slot: str = "entity") -> dict:
    """用户名词的虚拟字粒：没有语料出处，相位取当前语体，出处标"用户提问"。"""
    return {
        "word": word, "slot": slot, "count": 0, "amp": 1.0, "domain": "user",
        "style": _style_band(style_deg)[0], "phase_deg": round(style_deg, 1),
        "sources": ["用户提问"], "example": query[:80], "source": "user",
    }



class PrintRequest(BaseModel):
    bank_id: Optional[str] = None
    style_deg: float = Field(60.0, ge=0.0, le=120.0)
    nonce: int = Field(0, ge=0, le=1_000_000)
    top_k: int = Field(8, ge=1, le=50)
    topic: str = Field("", max_length=200)
    operators: OperatorWeights = Field(default_factory=OperatorWeights)
    # 用户原话与会话标识：不传 = 基线行为，字节不变
    user_query: str = Field("", max_length=200)
    session_id: Optional[str] = Field(None, max_length=64)
    # 算子融合方式：linear = 现有的逐项连乘（默认，基线不变）；
    # boltzmann = 自由能 exp(-Σ w_k(1-s_k))，权重全零时恰好为 1.0
    fusion: str = Field("linear", pattern="^(linear|boltzmann)$")
    # 自由能温度：τ 越小越尖锐（E 相同则乘子更小）；linear 模式忽略此值
    fusion_tau: float = Field(1.0, gt=0.0, le=10.0)
    # 拓扑层：请求级开关，默认关；env FEYNMAN_ENABLE_TOPOLOGY=1 仍可全局打开
    use_topology: bool = False
    # P8 条件级联：实体→动作→后果逐级打分 + 顺序束搜索；默认关，关时与旧版逐字节一致
    conditioned_chain: bool = False
    # P10 解码期检索约束：后果候选只取实体在语料里的 E→C 边指向的从句，
    # 没有边的实体退回原池；默认关，关时与旧版逐字节一致
    retrieve_consequence: bool = False
    # P10b 检索池质量门禁：主语错配/方向矛盾/残句过滤；默认关
    retrieval_gates: bool = False
    # P10c 两跳检索：没有 E→C 边的实体改走 E→A→C；默认关
    retrieval_via_action: bool = False
    # P12 问句实体锁定：问句抽出实体时实体槽只留这些实体，不拿库内其它实体补位；
    # 抽不出实体时退回原池。默认关，关时与旧版逐字节一致
    topic_lock: bool = False
    # P4/P5 信息量偏好：空话重复的后果降权、引入新指称的后果加权；默认关
    informative_pref: bool = False
    # P14 检索排序偏好：有 E→C 边时 tier=2（专属从句）先于通用原子；
    # 只改排序不改候选集，默认关，关时与旧版逐字节一致
    specific_pref: bool = False
    # P21 同构空间：用户实体的"没有共现 ×0.35"离散惩罚换成连续相容乘子；
    # 默认关，关时不导入 plugins/isomorphism_p21，与旧版逐字节一致
    use_isomorphism: bool = False


def _select_per_slot(vault: Vault, style_deg: float, per_slot: int, topic: str,
                     user_entities: Optional[List[str]] = None,
                     user_query: str = "",
                     topic_lock: bool = False) -> Dict[str, List[dict]]:
    picked: Dict[str, List[dict]] = {}
    for slot in SLOTS:
        target = style_deg + vault.slot_offsets[slot]
        glyphs = [g for g in vault.by_slot[slot] if _coverage(topic, g)] if topic else list(vault.by_slot[slot])
        glyphs.sort(key=lambda g: (g.get("tier", 1), abs(_glyph_phase(g) - target), -g["count"]))
        picked[slot] = glyphs[:per_slot]
    if user_entities:
        user_glyphs = [_user_glyph(w, style_deg, user_query) for w in user_entities[:per_slot]]
        if topic_lock:
            # P12：只留问句实体；抽不出实体时走上面的原池，不清空
            picked["entity"] = user_glyphs
        else:
            fill = max(1, per_slot - len(user_glyphs))
            picked["entity"] = user_glyphs + picked["entity"][:fill]
    return picked


def _augment_cognitive_actions(vault: Vault, style_deg: float,
                               user_entities: List[str],
                               overrides: Optional[Dict[str, str]] = None) -> Optional[List[dict]]:
    """抽象实体问句里，候选动作补上认知动词。

    语料最近相位的三个动作很可能全是"旋转/推动"这类物理动作；光靠打分压制
    会让所有路径一起变低，印不出话。这里先把"观察/测量/记住"放进候选，
    原字粒与出处不动，只多一条可选路径。
    """
    if not any(resolve_entity_prototype(e, overrides)[0].startswith("abstract")
               for e in user_entities):
        return None
    target = style_deg + vault.slot_offsets["action"]
    pool = [g for g in vault.by_slot["action"] if g["word"] in SAFE_COGNITIVE_ACTIONS]
    if not pool:
        return None
    pool.sort(key=lambda g: (abs(_glyph_phase(g) - target), -g["count"]))
    return pool[:2]


def _operator_scores(path: Dict[str, dict], vault: Vault, entangled_keys: set,
                     jargon: Optional[List[str]] = None) -> Dict[str, float]:
    """每个算子给路径打一个 [0,1] 分；只衡量"像不像"，不判断物理对错。"""
    hook, ent, act, cons = (path[s] for s in SLOTS)
    ew, cw, hw, aw = ent["word"], cons["word"], hook["word"], act["word"]

    # 微观拆解：实体短而具体、动作短而可做。
    # 长度分档改成连续衰减——同实体同动作、只换后果的路径分数才会拉开，
    # 自由能/线性乘子才有东西可比较（分档相同会让融合退化成常数）。
    e_len = max(0.2, 1.0 - 0.25 * max(0, len(ew) - 3))
    a_len = max(0.45, 1.0 - 0.20 * max(0, len(aw) - 3))
    scores = {"micro": 0.55 * e_len + 0.45 * a_len}

    # 跨界类比：日常实体 × 物理后果，或语料里真的同句出现过
    cross = 0.1
    if ent.get("domain") == "everyday" and cons.get("domain") == "physics":
        cross = 1.0
    elif ent.get("domain") != cons.get("domain"):
        cross = 0.7
    if (ew, cw) in entangled_keys:
        cross = 1.0
    else:
        # 有共现支撑就往上抬一点，仍守住"跨域才是主证据"
        support = vault.pair_support(ew, cw)
        if support:
            cross = min(1.0, cross + 0.2 * min(1.0, support / 4.0))
    scores["cross"] = cross

    # 极端边界：引导语/后果里有没有"推到极限"的措辞（措辞探测器，不是物理判据）
    b_hits = sum(t in hw for t in BOUNDARY_TOKENS)
    c_hits = sum(t in cw or t in aw for t in BOUNDARY_TOKENS)
    scores["boundary"] = min(1.0, 0.7 * b_hits + 0.35 * c_hits)

    # 时间反演：后果可观测才适合"由果溯因"地讲；真正的效果在骨架选择上。
    # 长度连续：越短的后果越像一句能盯住的可观测事实。
    if cons.get("domain") == "physics":
        scores["reversal"] = 0.8 * min(1.0, max(0.4, len(cw) / 8.0))
    else:
        scores["reversal"] = 0.3 * min(1.0, max(0.4, len(cw) / 8.0))

    # 反直觉：共现次数低但有据最像费曼；从没共现≈0.75（新奇但可能胡扯），
    # 陈词滥调扣到 0.2 左右。连续衰减，避免整池同分。
    n = vault.pair_support(ew, cw) + vault.pair_support(ew, aw)
    if n == 0:
        scores["anti_intuition"] = 0.75
    else:
        scores["anti_intuition"] = 0.2 + 0.8 * math.exp(-(n - 1) / 4.0)

    # 剥掉术语：只报名字的措辞越少越像费曼；日常物 + 比喻最贴
    hits = _jargon_hits(" ".join((hw, ew, aw, cw)), jargon or [])
    if not hits:
        scores["jargon_strip"] = 1.0 if ent.get("domain") == "everyday" else 0.7
    elif len(hits) == 1:
        scores["jargon_strip"] = 0.25
    else:
        scores["jargon_strip"] = 0.0

    # 最小作用量：话要短，落点要守在一件事上（守恒/不变/最低/相等）。
    # 长度连续衰减，避免只有"≤6 字"一档导致整池同分。
    _invariants = ("守恒", "不变", "最低", "最小", "稳定", "相同", "相等", "为零", "平衡")
    inv = sum(1 for t in _invariants if t in cw)
    inv_term = min(1.0, 0.5 * inv)
    len_term = 1.0 if len(cw) <= 6 else max(0.0, 1.0 - 0.12 * (len(cw) - 6))
    act_term = 1.0 if len(aw) <= 3 else 0.55
    scores["least_action"] = min(1.0, 0.45 * inv_term + 0.35 * len_term + 0.20 * act_term)

    # 货物崇拜检验：结论必须能被观测验伪，而不是"看起来像"
    _checks = ("观测", "测量", "实验", "证明", "推翻", "不一致", "读数", "误差", "预测")
    chk = sum(1 for t in _checks if t in cw or t in aw)
    scores["cargo_cult"] = min(1.0, 0.45 * chk)
    return scores


def _operator_multiplier(scores: Dict[str, float], w: OperatorWeights,
                         fusion: str = "linear", tau: float = 1.0) -> float:
    """w=0 时恒为 1.0；线性融合总乘数限幅 [0.25,4.0]。

    fusion="boltzmann" 时把每个算子分数取补当能量项：E = Σ w_k·(1−s_k)，
    乘数 = exp(−E/τ)。权重全零 → E 恰好为 0 → 乘数恰好 1.0。
    注意 E≥0 恒成立，所以自由能乘数只能 ≤1；提案里的 4.0 上限不可达，
    这里按 1.0 封顶（要奖励只能靠线性融合或拓扑证据）。
    温度 τ 越小越尖锐：同一个 E 得到更小的乘子，近并列路径被拉开。
    Boltzmann 分支不设 0.25 下限：绝对能量只影响公共尺度、概率归一后消失，
    设下限反而会把 E 大的整池截成同一个常数，重新退化成均匀乘子。
    """
    weights = w.model_dump()
    if fusion == "boltzmann":
        energy = sum(max(0.0, float(v)) * (1.0 - scores[k])
                     for k, v in weights.items() if k in scores)
        if energy == 0.0:
            return 1.0
        safe_tau = max(0.05, float(tau))
        return min(1.0, max(1e-12, math.exp(-energy / safe_tau)))
    mult = 1.0
    for key, score in scores.items():
        mult *= max(0.3, min(1.6, 1.0 + weights[key] * (score - 0.5)))
    return max(0.25, min(4.0, mult))


BADGE_TEXT = {
    "micro": "🔬 拆到零件",
    "cross": "🌉 跨域搭桥",
    "boundary": "🧨 推到边界",
    "reversal": "⏪ 由果溯因",
    "anti_intuition": "🙃 反直觉组合",
    "jargon_strip": "🧽 剥掉术语",
    "least_action": "🎯 最小作用量",
    "cargo_cult": "🔍 可验伪",
}


def _badges(scores: Dict[str, float], w: OperatorWeights) -> List[str]:
    weights = w.model_dump()
    return [BADGE_TEXT[k] for k, s in scores.items() if weights[k] > 0 and s >= 0.75]


def _choose_skeleton(vault: Vault, style_deg: float,
                     w: OperatorWeights, metaphor_mode: bool = False,
                     abstract_mode: bool = False,
                     skeleton_scores: Optional[Dict[str, float]] = None) -> Optional[dict]:
    """骨架按语体距离选，算子投票改写偏好；无骨架时返回 None（退回逗号直拼）。"""
    if not vault.skeletons:
        return None
    learned = skeleton_scores or {}
    best = None
    best_score = -1e9
    for sk in vault.skeletons:
        kind = sk.get("pattern_type")
        # 语料里各句式的实测相位挤在 32°–55°（讲义腔占绝对多数），直接拿来当偏好会把
        # 比喻句推到所有中高相位。改为按语体弧给每类句式定目标相位。
        target = SKELETON_TARGET_PHASE.get(kind, float(sk.get("phase_deg", 45.0)))
        score = -abs(target - style_deg) / 40.0
        # 默认句式要稳，比喻句不出钱不选
        score += 0.35 if kind == "declarative" else 0.0
        score -= 0.6 if kind == "analogy" else 0.0
        score += 0.5 * w.boundary if kind == "conditional" else 0.0
        score += 0.6 * w.reversal if kind == "retro" else 0.0
        score += 0.4 * w.cross if kind == "analogy" else 0.0
        # 有用户实体时，比喻句正好把"你问的东西"放进"就像"的右槽
        score += 0.45 if metaphor_mode and kind == "analogy" else 0.0
        # 抽象概念（股票/算法/合同）做"就像"的喻体只会读出范畴错误，直接不选类比弧
        score -= 3.0 if abstract_mode and kind == "analogy" else 0.0
        score += 0.2 * w.anti_intuition if kind == "contrast" else 0.0
        score += 0.15 * w.micro if kind == "temporal" else 0.0
        score += 0.5 * w.least_action if kind in ("conditional", "causal") else 0.0
        score += 0.5 * w.cargo_cult if kind in ("retro", "contrast") else 0.0
        if w.jargon_strip >= 1.5 and kind == "analogy":
            score += 0.6
        score += 0.15 * min(float(sk.get("freq", 0)), 100.0) / 100.0
        # 反馈习得的骨架偏好：1.0 为中性，0.5–1.5 折算成 ±0.5 的偏置
        score += 0.5 * (float(learned.get(kind, 1.0)) - 1.0)
        if score > best_score:
            best, best_score = sk, score
    return best


def _render_line(skeleton: Optional[dict], glyphs: Dict[str, str],
                 vault: Optional[Vault] = None,
                 orders: Optional[Dict[str, str]] = None,
                 user_entity: str = "",
                 drop_hook: bool = False) -> str:
    """把四颗字粒填进骨架。整句是拼装句：字粒有出处，句式来自语料，整句不是原话。

    drop_hook 只做删减：段落的第二、三句前面已经有"打个比方，""说到底，"，
    再留起手式会读成"打个比方，我们看到，……"的双开头。
    """
    plain = "，".join(glyphs[s] for s in SLOTS)
    if not skeleton:
        return plain
    template = skeleton["template"]
    if drop_hook and template.startswith("{hook}"):
        template = template[len("{hook}"):].lstrip("，,：: ")
    bitmap = dict(glyphs)
    # 比喻句的双槽锁：左边是语料里的实体/动作，右边"就像…"放用户问的那个东西
    if (user_entity and skeleton.get("pattern_type") == "analogy"
            and glyphs["entity"] != user_entity):
        bitmap["consequence"] = user_entity
    # 及物动作放实体前（"观察电子"），不及物放实体后（"电子旋转"）
    bitmap["verb"] = (
        glyphs["action"] + glyphs["entity"]
        if (orders or {}).get("action") == "pre"
        else glyphs["entity"] + glyphs["action"]
    )
    # 骨架自带"假设"时别和起手式重复
    if "假设" in template and any(k in glyphs["hook"] for k in ("假设", "如果", "假如", "设想")):
        declarative = next(
            (sk["template"] for sk in (vault.skeletons if vault else [])
             if sk.get("pattern_type") == "declarative"),
            DEFAULT_TEMPLATE,
        )
        template = declarative
    try:
        return template.format(**bitmap).replace("，，", "，").replace("，。", "。")
    except (KeyError, IndexError):
        return plain


def _evaluate_path(path: Dict[str, dict], style_deg: float, jargon: List[str],
                   entangled_keys: set, vault: Optional[Vault] = None,
                   weights: Optional[OperatorWeights] = None,
                   learned_edges: Optional[Dict[str, float]] = None,
                   prototypes: Optional[Dict[str, str]] = None,
                   fusion: str = "linear", tau: float = 1.0,
                   topology=None, topology_analogy: bool = False,
                   conditioned: bool = False, p21=None) -> dict:
    words = [path[s]["word"] for s in SLOTS]
    line = "，".join(words)  # 表内用的直拼版；展示用渲染版在 run 里生成

    phases = [_glyph_phase(path[s]) for s in SLOTS]
    phase_sum = sum(phases) % 360.0
    offsets = vault.slot_offsets if vault else SLOT_PHASE_OFFSET
    # 每颗字粒与它该在的语体相位（旋钮 + 槽位偏移）的余弦对齐，再取平均。
    # 不能用相位和去比 4×旋钮：360° 回绕会让高相位组合在低旋钮上意外对齐。
    if p21 is not None:
        # 阶段 D：用字粒在语料里的圆周均值 + 可靠性做带宽容差；
        # 等价于给"首次相位"降噪，返回量纲仍是 [-1,1]
        alignments = [
            p21.engine.aligned(path[s]["word"], _glyph_phase(path[s]),
                               style_deg + offsets[s])
            for s in SLOTS
        ]
    else:
        alignments = [
            math.cos(math.radians(_glyph_phase(path[s]) - (style_deg + offsets[s])))
            for s in SLOTS
        ]
    projection = sum(alignments) / len(alignments)
    amp = 1.0
    for s in SLOTS:
        amp *= float(path[s].get("amp", SLOT_AMP[s]))
    # 语体相干度：相位弧是否跟旋钮对得上
    coherence = _circular_concentration(phases, style_deg)

    notes: List[str] = []
    weight = max(0.0, (projection + 1.0) / 2.0) ** 2 * (0.55 + 0.45 * coherence) * amp

    # 起手式是不是"费曼式开场"
    if any(m in path["hook"]["word"] for m in ("你", "我们", "想象", "假设", "注意", "看", "想", "比方")):
        weight *= 1.25
        notes.append("🗣️ 有对话感")

    # 跨域纠缠：日常实体 × 物理后果
    ent_domain = path["entity"].get("domain", "everyday")
    if ent_domain == "everyday" and path["consequence"].get("domain") == "physics":
        weight *= 1.35
        notes.append("⚡ 日常物 × 物理后果")
    elif ent_domain == "physics" and path["hook"].get("word") in {"你", "你看", "你想", "想象"}:
        weight *= 1.15
        notes.append("🔭 物理实体被拽进对话")
    key = (path["entity"]["word"], path["consequence"]["word"])
    if key in entangled_keys:
        weight *= 1.2
        notes.append("🔗 语料里真的同句出现过")

    # 心智算子：默认全 0，乘数恒 1.0，基线行为不变
    scores: Dict[str, float] = {}
    if vault is not None and weights is not None:
        scores = _operator_scores(path, vault, entangled_keys, jargon)
        mult = _operator_multiplier(scores, weights, fusion=fusion, tau=tau)
        if abs(mult - 1.0) > 1e-9:
            weight *= mult

    # 拓扑层（可选）：测地线能量 + 实心三角形/空洞；topology=None 时乘数恒 1.0
    topo_info: Optional[dict] = None
    if topology is not None:
        hw = path["hook"]["word"]
        ew = path["entity"]["word"]
        aw = path["action"]["word"]
        cw = path["consequence"]["word"]
        tm, is_simplex = topology.compute_geodesic_multiplier(hw, ew, aw, cw)
        notes_topo: List[str] = []
        if is_simplex:
            notes_topo.append("▲ 三槽在语料里同句出现过")
        elif tm < 1.0:
            notes_topo.append("△ 三槽没有同句证据")
        analogy_sim = None
        if topology_analogy:
            analogy_sim = topology.compute_analogy_alignment(ew, cw)
            if analogy_sim is not None:
                boost = min(2.34, max(0.43, math.exp(0.85 * analogy_sim)))
                tm *= boost
                if analogy_sim >= 0.25:
                    notes_topo.append(f"↔ 类比空间对齐 {analogy_sim:.2f}")
        tm = max(0.25, min(1.80, tm))
        covered = True
        if hasattr(topology, "covers"):
            covered = bool(topology.covers(hw, ew, aw, cw))
        if abs(tm - 1.0) > 1e-9:
            weight *= tm
            notes.extend(notes_topo)
        pair_ev = None
        if hasattr(topology, "pair_evidence"):
            pair_ev = topology.pair_evidence(ew, aw, cw)
            if pair_ev["tilt"] > 1.0:
                notes_topo.append(f"+ 成对共现 {pair_ev['supports']} ×{pair_ev['tilt']:.2f}")
        topo_info = {"multiplier": round(tm, 4), "is_simplex": bool(is_simplex),
                     "analogy_sim": None if analogy_sim is None else round(analogy_sim, 4),
                     "covered": covered,
                     "pair_score": pair_ev["score"] if pair_ev else 0.0,
                     "pair_tilt": pair_ev["tilt"] if pair_ev else 1.0}

    # P8 条件级联：动作看实体、后果看动作；无证据时恰好 1.0（基线不变）
    cond_info: Optional[float] = None
    if conditioned and vault is not None:
        cm = vault.cond_multiplier(path["entity"]["word"], path["action"]["word"],
                                   path["consequence"]["word"])
        cond_info = cm
        if abs(cm - 1.0) > 1e-9:
            weight *= cm
            notes.append(f"🔗 条件共现 ×{cm:.2f}")

    # 用户实体注入：路径落到用户问的东西上，按原型共现判断动作搭不搭
    p21_info: Optional[dict] = None
    if path["entity"].get("source") == "user":
        weight *= 1.6
        notes.append("🎯 落在你问的东西上")
        uw, aw = path["entity"]["word"], path["action"]["word"]
        cw = path["consequence"]["word"]
        family, proto = resolve_entity_prototype(uw, prototypes)
        support = 0
        if vault is not None:
            support = vault.pair_support(uw, aw) or (vault.pair_support(proto, aw) if proto else 0)
        if support:
            weight *= 1.15
            notes.append("🔗 原型与动作在语料里同句出现过")
        elif family.startswith("abstract"):
            # 抽象词只认"能看见/能数/能比"的动作；"推动股票"这类直接压死
            if aw in PHYSICAL_ONLY_ACTIONS:
                weight *= 0.02
                notes.append("⛔ 抽象概念不适用物理动作")
            elif aw in SAFE_COGNITIVE_ACTIONS:
                weight *= 1.35
                notes.append("🧠 抽象概念用认知动作来讲")
            elif p21 is not None and (m := p21.multiplier(uw, aw, cw)) and m[1].get("score") is not None:
                mult_, p21_info = m
                # 乘子改在池内按分位施加（P21Context.rerank_pool）；这里只记录，
                # 同一请求内的路径才可比，单条路径不再被固定中心压低。
                p21_info["pending"] = True
                notes.append(f"🧭 抽象词 × 动作相容 {p21_info.get('score')}（池内分位）")
            else:
                weight *= 0.35
                notes.append("⚠️ 动作与这个抽象概念没有共现")
        elif aw not in SAFE_UNIVERSAL_ACTIONS:
            if p21 is not None and (m := p21.multiplier(uw, aw, cw)) and m[1].get("score") is not None:
                mult_, p21_info = m
                p21_info["pending"] = True
                notes.append(f"🧭 语料外实体 × 动作相容 {p21_info.get('score')}（池内分位）")
            else:
                weight *= 0.35
                notes.append("⚠️ 动作与这个实体在语料里没有共现")
        # 进化库：只对用户实体学到的搭配生效，且只在已有点赞记录时乘一次
        if learned_edges:
            lm = learned_edges.get(f"{uw}|{aw}")
            if lm is not None and abs(float(lm) - 1.0) > 1e-9:
                weight *= float(lm)
                notes.append(f"🧪 反馈习得 ×{float(lm):.2f}")

    # 术语闸门：只报名字、不讲现象的措辞扣权；两颗以上直接否决
    pre_veto = weight
    hits: List[str] = []
    for s in SLOTS:
        hits.extend(_jargon_hits(path[s]["word"], jargon))
    if len(hits) >= 2:
        weight = 0.0
        notes.append("🚫 术语堆叠，否决")
    elif hits:
        weight *= 0.4
        notes.append(f"⚠️ 带术语：{hits[0]}")

    result = {
        "line": line,
        "glyphs": {s: path[s]["word"] for s in SLOTS},
        "orders": {s: path[s].get("order", "post") for s in SLOTS},
        "sources": {s: path[s].get("sources", [])[:2] for s in SLOTS},
        "styles": {s: path[s].get("style") for s in SLOTS},
        "phase_total_deg": round(phase_sum, 1),
        "phase_mean": round(sum(phases) / len(phases), 1),
        "projection": round(projection, 4),
        "coherence": round(coherence, 4),
        "weight": round(weight, 6),
        "pre_veto_weight": round(pre_veto, 6),
        "notes": notes,
        "jargon_hits": hits,
        "operator_scores": {k: round(v, 3) for k, v in scores.items()},
        "badges": _badges(scores, weights) if (scores and weights is not None) else [],
    }
    if topo_info is not None:
        result["topology"] = topo_info
    if cond_info is not None:
        result["conditioned_multiplier"] = cond_info
    if p21_info is not None:
        result["p21"] = p21_info
    return result


def _normalize_topology_pool(paths: List[dict]) -> None:
    """把拓扑乘子按候选池中位数归一到 1.0 附近。

    实测 p8b：真实候选几乎全部由测地曲率压在 0.3 左右，三角形/空洞因子
    从不触发——不归一时，拓扑层等于给全池乘一个常数，只在概率第 4 位小数
    上撬动近并列，是噪声不是证据。归一后只保留"比同池别的路径更弯/更直"
    的相对差异。注意：未命中三元组（covered=False）也带着测地曲率乘子，
    所以池内全体带 topology 的路径都参与归一并被改写；完全没有拓扑记录的
    路径不动。
    """
    pool = [p for p in paths if p.get("topology")]
    if len(pool) < 2:
        return
    vals = [float(p["topology"]["multiplier"]) for p in pool]
    med = statistics.median(vals)
    if med <= 1e-9:
        return
    for p in pool:
        info = p["topology"]
        raw = float(info["multiplier"])
        norm = max(0.5, min(2.0, raw / med))
        if raw > 1e-9 and abs(norm - raw) > 1e-9:
            p["weight"] = round(p["weight"] * (norm / raw), 6)
        info["multiplier_raw"] = round(raw, 4)
        info["multiplier"] = round(norm, 4)


def _arbitrate_double_penalty(paths: List[dict]) -> int:
    """条件级联与拓扑层同时惩罚时，只保留较轻的一条。

    实测 p8b：两条惩罚同时 <1 的路径约占 14%，乘积会把权重压到 0.4 以下；
    但两者看的是同一件事（这三颗字粒在语料里搭不搭），叠加没有新信息。
    规则：两条都 <1 时把较重的已乘因子"退还"——weight / min(m)；
    任一 ≥1（奖励）时不动，奖励照常叠加。
    """
    fixed = 0
    for p in paths:
        topo = p.get("topology") or {}
        cm = p.get("conditioned_multiplier")
        if cm is None or not topo:
            continue
        tm = float(topo.get("multiplier", 1.0))
        cm = float(cm)
        if tm < 1.0 and cm < 1.0 and min(tm, cm) > 1e-9:
            p["weight"] = round(p["weight"] / min(tm, cm), 6)
            topo["double_penalty_arbitrated"] = True
            if "🔗 双惩罚互斥" not in p.setdefault("notes", []):
                p["notes"].append("🔗 双惩罚互斥：只留较轻那条")
            fixed += 1
    return fixed


RETRIEVAL_TOP_K = 3

# P4/P5 信息量偏好系数：只看从句自身的结构信息，不猜语义。
# 语料层实测左邻多为虚词，按"左动作/左主语"匹配取不到证据，改用
# 从句是否自带主语、是否引入新指称来判断"通顺但空洞"。
INFORMATIVE_BOOST = 1.15          # 引入其它库内实体：有新指称
INFORMATIVE_SELF_PENALTY = 0.90   # 以本实体开头、没有其它实体：换个说法的重复


def _informativeness_factor(consequence: str, entity: str, vault: Vault) -> float:
    """按从句结构给后果乘一个系数；没有可用信号时中性 1.0。

    只判两件事：从句是不是把主语换个说法重复一遍；有没有引入新指称。
    主语错位（以另一个实体开头）不在这里罚——那由检索门禁管，而且
    检索要的正是引入新指称的从句，再罚会自相矛盾。
    """
    if not consequence or not entity:
        return 1.0
    ents = [g["word"] for g in vault.by_slot["entity"]]
    others = any(e != entity and e in consequence for e in ents)
    if consequence.startswith(entity):
        return INFORMATIVE_BOOST if others else INFORMATIVE_SELF_PENALTY
    return INFORMATIVE_BOOST if others else 1.0

# 方向矛盾词对：问句说"升高"时，后果里的"降低"是答反，不是回答
ANTONYM_PAIRS = (
    ("升高", "降低"), ("升高", "下降"), ("上升", "下降"),
    ("增大", "减小"), ("增加", "减少"), ("增加", "变小"),
    ("变亮", "变暗"), ("变热", "变冷"), ("加快", "减慢"), ("加速", "减速"),
)


def _retrieval_candidate_ok(word: str, entity: str, query: str = "") -> Tuple[bool, str]:
    """P10b 检索池质量门禁：返回 (是否保留, 拒绝原因)。

    三类确定性坏后果：
    - subject_mismatch：实体只是另一个名词的一部分（原子 ⊂ 原子弹），
      同段共现不等于同一主语；
    - contradiction：问句与后果方向相反（"温度升高以后" vs "温度降低"）；
    - malformed："是+裸谓词"的残句（"空间是弯曲"，缺"的"）。
    """
    tokens = [(w, p) for w, p in pseg.cut(word)]
    if any(w != entity and entity in w for w, _ in tokens):
        return False, "subject_mismatch"
    if query:
        for a, b in ANTONYM_PAIRS:
            if (a in query and b in word) or (b in query and a in word):
                return False, "contradiction"
    if len(tokens) >= 2 and tokens[-2][0] == "是" and tokens[-1][1] in ("a", "ad", "v", "vn"):
        return False, "malformed"
    return True, ""


def _retrieve_consequences(vault: Vault, entity: str, pool: List[dict],
                           target_deg: float, top_k: int = RETRIEVAL_TOP_K,
                           gates: bool = False, query: str = "",
                           via_action: bool = False,
                           specific_pref: bool = False) -> List[dict]:
    """P10 解码期检索：实体在语料里有 E→C 边时，后果候选只取这些边指向的从句。

    候选来自整个后果槽（可以是几百条库存），不是先按相位截好的 8 条；
    排序按共现支撑次数，同分再看相位距离。一条边都没有时退回原池——
    检索是约束真实语料证据，不是凭空发明约束。
    gates=True 时先过 _retrieval_candidate_ok；全被拦下则退回未过滤候选。
    via_action=True 时，没有 E→C 边的实体先用 E→A 找到动作、再经 A→C 两跳取后果。
    """
    if not entity:
        return pool

    def ok(g: dict) -> bool:
        return (not gates) or _retrieval_candidate_ok(g["word"], entity, query)[0]

    wired = [g for g in vault.by_slot["consequence"]
             if vault.cooc_count(entity, g["word"]) > 0]
    if wired:
        # P14：specific 优先只改变排序先后，不引入新的边
        # P15：人工策展的专属从句在 tier2 内优先于自动挖出的从句
        def _key(g: dict) -> tuple:
            tier_rank = 0 if (specific_pref and g.get("tier", 1) == 2) else 1
            curated_rank = 0 if (specific_pref and g.get("curated")) else 1
            return (tier_rank, curated_rank,
                    -vault.cooc_count(entity, g["word"]),
                    abs(_glyph_phase(g) - target_deg), g["word"])

        wired.sort(key=_key)
        kept = [g for g in wired if ok(g)]
        return (kept or wired)[:top_k]

    if via_action:
        acts = [a["word"] for a in vault.by_slot["action"]
                if vault.cooc_count(entity, a["word"]) > 0]
        if acts:
            hops = {}
            for g in vault.by_slot["consequence"]:
                s = sum(vault.cooc_count(a, g["word"]) for a in acts)
                if s > 0:
                    hops[g["word"]] = s
            if hops:
                items = [g for g in vault.by_slot["consequence"] if g["word"] in hops]
                items.sort(key=lambda g: (-hops[g["word"]],
                                          abs(_glyph_phase(g) - target_deg), g["word"]))
                kept = [g for g in items if ok(g)]
                return (kept or items)[:top_k]

    # 彻底没有证据：退回相位池，门禁仍拦残句（主语错配按实体查，方向矛盾按问句查）
    kept = [g for g in pool if ok(g)]
    return kept or pool


def _conditioned_beam(vault: Vault, picked: Dict[str, List[dict]], style_deg: float,
                      width_entity: int = 4, width_action: int = 3,
                      width_consequence: int = 2,
                      retrieval: bool = False, gates: bool = False,
                      via_action: bool = False,
                      specific_pref: bool = False,
                      query: str = "") -> List[Tuple[dict, dict, dict]]:
    """P8 顺序束搜索：实体 → 动作 → 后果，后一级只在前一级的共现边上展开。

    每级先按有向共现次数排序，没有证据时退回相位距离（与旧的按槽独立选同序）。
    宽度 4/3/2：比 8×8×8 全积小一个量级，且后槽只看得到前槽挑剩的候选。
    retrieval=True（P10）时每个实体的后果候选换成它的 E→C 检索池。
    """
    offs = vault.slot_offsets

    def gap(g: dict, slot: str) -> float:
        return abs(_glyph_phase(g) - (style_deg + offs[slot]))

    beam: List[Tuple[dict, dict, dict]] = []
    cons_target = style_deg + offs["consequence"]

    def wired(a: dict, pool: List[dict]) -> bool:
        """动作在 A→C 表里有出边：条件级联只该沿着有后果证据的动作走。"""
        return any(vault.cooc_count(a["word"], c["word"]) > 0 for c in pool)

    for e in picked["entity"][:width_entity]:
        # 检索模式下这个实体的后果池由 E→C 边决定；没有边时就是原池
        e_pool = (_retrieve_consequences(vault, e["word"], picked["consequence"],
                                         cons_target, gates=gates, query=query,
                                         via_action=via_action,
                                         specific_pref=specific_pref)
                  if retrieval else picked["consequence"])
        # 有证据时按 E→A 提升比排；没证据时先拿有 A→C 出边的动作（振动/碰撞），
        # 再按相位——只按相位会让未知实体永远落在"产生/开始/消失"这类空动作上。
        mea = vault._cond_stats()["e_tot"].get(e["word"], 0)
        a_out = vault._cond_stats()["a_out"]

        def _act_key(a: dict) -> tuple:
            if a.get("source") == "user":          # 问句自己的谓语：一定排最前
                return (0, 0.0, 0.0, gap(a, "action"))
            if mea:
                return (1, float(not wired(a, e_pool)),
                        -vault.edge_lifts(e["word"], a["word"], "")[0],
                        gap(a, "action"))
            return (1, float(not wired(a, e_pool)), -float(a_out.get(a["word"], 0)),
                    gap(a, "action"))

        acts = sorted(picked["action"], key=_act_key)
        if not mea and not any(a.get("source") == "user" for a in acts):
            # 未知实体：有带出边的动作就只用它们，别让相位最近的空动作混进束里
            causally_wired = [a for a in acts if wired(a, e_pool)]
            if causally_wired:
                acts = causally_wired
        for a in acts[:width_action]:
            # 后果同理：动作有 A→C 出边时只在"真有边"的后果里按相位挑——
            # 按提升比排会让每个动作永远印同一条最强后果（"折射率会变得非常大"）
            a_out = vault._cond_stats()["a_out"].get(a["word"], 0)
            if a_out:
                wired_cons = [c for c in e_pool
                              if vault.cooc_count(a["word"], c["word"]) > 0]
                cons = sorted(wired_cons or e_pool, key=lambda c: gap(c, "consequence"))
            else:
                # 用户动词没有 A→C 出边：先看实体自己的 E→C 边（冰→冰会浮…），
                # 再退回相位；只在近相位的几条里挑，避免所有动作都印同一条后果
                e_cons = [c for c in e_pool
                          if vault.cooc_count(e["word"], c["word"]) > 0]
                if e_cons:
                    cons = sorted(e_cons, key=lambda c: gap(c, "consequence"))
                else:
                    near = sorted(e_pool, key=lambda c: gap(c, "consequence"))[:6]
                    if near:
                        k = sum(map(ord, a["word"])) % len(near)
                        near = near[k:] + near[:k]
                    cons = near
            for c in cons[:width_consequence]:
                beam.append((e, a, c))
    return beam


def run_feynman_print(vault: Vault, req: PrintRequest, ext_hash: str = "",
                      learned_edges: Optional[Dict[str, float]] = None,
                      skeleton_scores: Optional[Dict[str, float]] = None,
                      prototypes: Optional[Dict[str, str]] = None,
                      topology=None) -> dict:
    if req.bank_id and req.bank_id != vault.bank_id:
        raise KeyError(req.bank_id)

    seed = _seed(vault.bank_id, req.style_deg, req.nonce, req.session_id or "", ext_hash)
    conditioned = bool(req.conditioned_chain)
    # 纯招呼/自报家门：不进语料排版，走模板回复（标记 greeting，前端据此不显示"凑不齐三句"）
    talk = detect_smalltalk(req.user_query)
    if talk:
        band, desc = _style_band(req.style_deg)
        return {
            "collapsed": True,
            "printed_collapse": talk,
            "top_resonant_line": talk,
            "top_probability": None,
            "top_badges": [],
            "style_band": band,
            "style_desc": desc,
            "paths": [], "paths_total": 0, "paths_alive": 0,
            "seed": seed, "purity": 1.0, "killed": [],
            "skeleton": None, "operators": req.operators.model_dump(),
            "user_entities": [], "user_verbs": [], "ext_hash": ext_hash or EMPTY_EXT_HASH,
            "conditioned": conditioned, "greeting": True,
            "message": "招呼语走模板回复，不调用语料排版。",
        }
    # 默认 3^4=81 条路径；条件级联时放宽候选池，由束搜索控制组合数
    per_slot = 8 if conditioned else 3
    user_entities = extract_entities(req.user_query, vault.jargon, limit=per_slot,
                                     prefer={g["word"] for g in vault.by_slot["entity"]})
    # 问句谓语注入动作槽：字库里没有"发光/吸/浮"时拿问句动词直接当动作字。
    # "发光"这类词既可能被抽成实体（vn 白名单）又可能是谓语：取掉它以后
    # 问句还有主语，就当动作；没有主语了（"摩擦为什么会生热"），它仍是实体。
    # 动词本身已在字库动作表里也照样注入——问句问的就是它，不能被别的动作挤掉。
    user_verbs = extract_query_verbs(req.user_query)
    if user_verbs:
        # 既当实体又当动词的词（"摩擦"被 jieba 切出 vn）：问句还有别的谓语时，
        # 它留给主语槽（"摩擦|生热"），不能整句只剩一个词还退化成实体。
        if len(user_verbs) > 1:
            user_verbs = [v for v in user_verbs if v not in user_entities] or user_verbs[:1]
        remaining = [e for e in user_entities if e not in user_verbs]
        if remaining:
            user_entities = remaining
        else:
            user_verbs = []
    user_entity = user_entities[0] if user_entities else ""
    picked = _select_per_slot(vault, req.style_deg, per_slot, req.topic,
                              user_entities=user_entities, user_query=req.user_query,
                              topic_lock=bool(req.topic_lock))
    # 把用户谓语做成动作字粒，排在动作池最前
    verb_glyphs: List[dict] = []
    for v in user_verbs:
        g = _user_glyph(v, req.style_deg, req.user_query, slot="action")
        # 动词在问句里出现在实体之前（"吸铁"）→ 及物语序，放实体前面
        g["order"] = ("pre" if user_entity and req.user_query.find(v) >= 0
                      and req.user_query.find(v) < req.user_query.find(user_entity) else "post")
        verb_glyphs.append(g)
    if verb_glyphs:
        seen_verbs = {g["word"] for g in verb_glyphs}
        picked["action"] = verb_glyphs + [g for g in picked["action"]
                                          if g["word"] not in seen_verbs]
    # 抽象实体问句：候选动作里补两条认知动词，避免整池都是物理动作而无路可印
    bridge = _augment_cognitive_actions(vault, req.style_deg, user_entities, prototypes)
    if bridge:
        seen = {g["word"] for g in bridge}
        picked["action"] = bridge + [g for g in picked["action"] if g["word"] not in seen]
    if any(not picked[s] for s in SLOTS):
        missing = [SLOT_LABELS[s] for s in SLOTS if not picked[s]]
        return {
            "collapsed": False,
            "printed_collapse": BLANK_GLYPH,
            "message": f"语料里没有可用的{'、'.join(missing)}字粒，印不出可核对的句子。",
            "blank": True,
            "style_band": _style_band(req.style_deg)[0],
            "paths": [], "paths_total": 0, "seed": seed,
            "purity": 0.0, "killed": [],
            "user_entities": user_entities,
            "user_verbs": user_verbs,
            "ext_hash": ext_hash or EMPTY_EXT_HASH,
        }

    abstract_mode = any(
        resolve_entity_prototype(e, prototypes)[0].startswith("abstract") for e in user_entities
    )
    # P21：只在开关打开且离线基底存在时构建；文件缺失就退回基线离散惩罚，
    # 不让排版因缺少产物而失败。
    p21_ctx = None
    if req.use_isomorphism:
        from plugins.isomorphism_p21 import P21Context, load_engine
        _p21_engine = load_engine()
        if _p21_engine is not None:
            p21_ctx = P21Context(_p21_engine, user_entities, user_verbs,
                                 learned_edges, prototypes)
    skeleton = _choose_skeleton(vault, req.style_deg, req.operators,
                                metaphor_mode=bool(user_entities),
                                abstract_mode=abstract_mode,
                                skeleton_scores=skeleton_scores)
    # 类比空间只在这两类情形下启用：骨架选了比喻句，或旋钮落在 92° 类比弧
    topo_analogy = bool(
        topology is not None and (
            (skeleton and skeleton.get("pattern_type") == "analogy")
            or 82.0 <= req.style_deg <= 102.0))
    # 比喻句：左槽讲语料里的东西，右槽"就像…"放用户问的那个词。
    # 用户实体不能同时占两槽，否则会出现"就像咖啡杯"式的自指。
    entity_picks = picked["entity"]
    if skeleton and skeleton.get("pattern_type") == "analogy" and user_entity:
        corpus_only = [g for g in entity_picks if g.get("source") != "user"]
        entity_picks = corpus_only or entity_picks

    entangled_keys = {(p["everyday"], p["physics"]) for p in vault.entangled if "everyday" in p}
    paths = []
    if conditioned:
        # 动作池补边：相位选出的 8 条里未必有带 A→C 出边的动作，
        # 每个实体再补一条 E→A 计数最高的动作，否则"有实体没因果动作"会退相位凑数。
        seen_act = {g["word"] for g in picked["action"]}
        extra_acts = []
        for e in entity_picks[:4]:
            for g in sorted(vault.by_slot["action"],
                            key=lambda x: (-vault.cooc_count(e["word"], x["word"]), x["word"])):
                if g["word"] in seen_act:
                    continue
                if vault.cooc_count(e["word"], g["word"]) > 0:
                    extra_acts.append(g)
                    seen_act.add(g["word"])
                    break
        act_pool = picked["action"] + extra_acts
        # 后果池按 A→C 边补全：相位选出的 8 条之外，把动作池有出边的后果也放进来，
        # 否则"有动作没后果"的链会被相位无关的后果凑满，因果证据用不上。
        seen_cons = {g["word"] for g in picked["consequence"]}
        extra_cons = [
            g for g in vault.by_slot["consequence"]
            if g["word"] not in seen_cons
            and any(vault.cooc_count(a["word"], g["word"]) > 0 for a in act_pool)
        ]
        cond_picked = {**picked, "entity": entity_picks, "action": act_pool,
                       "consequence": picked["consequence"] + extra_cons}
        # 问句里有实体时，条件束只在用户实体上展开：加语料实体进来，因果证据
        # 更足的"电子/量子理论"会把问句的东西整片挤掉，印出来答非所问。
        user_picks = [g for g in entity_picks if g.get("source") == "user"]
        if user_picks:
            # 只拿第一个用户实体当主语：第二个往常常是同一句里切出来的谓词碎片
            # （"吸铁""发光"），当主语印出来是病句
            cond_picked = {**cond_picked, "entity": user_picks[:1]}
        # 顺序束：实体→动作→后果先缩到几十组，再与起手式组合，后槽只看前槽的边
        for g2, g3, g4 in _conditioned_beam(vault, cond_picked, req.style_deg,
                                            retrieval=bool(req.retrieve_consequence),
                                            gates=bool(req.retrieval_gates),
                                            via_action=bool(req.retrieval_via_action),
                                            specific_pref=bool(req.specific_pref),
                                            query=req.user_query):
            for g1 in picked["hook"][:4]:
                paths.append(_evaluate_path(
                    {"hook": g1, "entity": g2, "action": g3, "consequence": g4},
                    req.style_deg, vault.jargon, entangled_keys,
                    vault=vault, weights=req.operators,
                    learned_edges=learned_edges, prototypes=prototypes,
                    fusion=req.fusion, tau=req.fusion_tau, topology=topology,
                    topology_analogy=topo_analogy, conditioned=True, p21=p21_ctx))
    else:
        cons_target = req.style_deg + vault.slot_offsets["consequence"]
        for g1 in picked["hook"]:
            for g2 in entity_picks:
                # P10：每个实体的后果候选由它的 E→C 边决定；没边时是原池
                cons_for_e = (_retrieve_consequences(vault, g2["word"],
                                                     picked["consequence"], cons_target,
                                                     gates=bool(req.retrieval_gates),
                                                     query=req.user_query,
                                                     via_action=bool(req.retrieval_via_action),
                                                     specific_pref=bool(req.specific_pref))
                              if req.retrieve_consequence else picked["consequence"])
                for g3 in picked["action"]:
                    for g4 in cons_for_e:
                        paths.append(_evaluate_path(
                            {"hook": g1, "entity": g2, "action": g3, "consequence": g4},
                            req.style_deg, vault.jargon, entangled_keys,
                            vault=vault, weights=req.operators,
                            learned_edges=learned_edges, prototypes=prototypes,
                            fusion=req.fusion, tau=req.fusion_tau, topology=topology,
                            topology_analogy=topo_analogy, p21=p21_ctx))

    # P21 池内重排（阶段 C2）：全局机制项 + 相容分位乘子。
    # 在信息量偏好与拓扑归一之前施加，后续的池内归一都只看相对差异。
    p21_pool = None
    if p21_ctx is not None:
        p21_pool = p21_ctx.rerank_pool(paths, user_entities[0] if user_entities else "")

    # P4/P5：信息量偏好只调整权重，不改候选集，也不影响默认路径
    if req.informative_pref:
        for p in paths:
            g = p["glyphs"]
            p["weight"] *= _informativeness_factor(
                g["consequence"], g["entity"], vault)

    # 拓扑乘子池内归一：约掉测地曲率带来的全池常数，只保留相对差异
    if topology is not None:
        _normalize_topology_pool(paths)
        if conditioned:
            _arbitrate_double_penalty(paths)

    def _decorate(p: dict) -> dict:
        p["rendered"] = _render_line(skeleton, p["glyphs"], vault, p.get("orders"),
                                     user_entity=user_entity)
        return p

    alive = [p for p in paths if p["weight"] > 0]
    killed = sorted(
        [p for p in paths if p["weight"] == 0 and p["jargon_hits"]],
        key=lambda p: (-p["pre_veto_weight"], p["line"]),
    )[:3]
    total_mass = sum(p["weight"] for p in paths)
    tainted_mass = sum(p["weight"] for p in paths if p["jargon_hits"])
    purity = round(1.0 - tainted_mass / total_mass, 4) if total_mass > 0 else 0.0

    if not alive:
        return {
            "collapsed": False,
            "printed_collapse": BLANK_GLYPH,
            "message": "81 条路径全被术语闸门或相位投影否决；缩小话题或降低旋钮再试。",
            "blank": True,
            "style_band": _style_band(req.style_deg)[0],
            "paths": [_decorate(p) for p in sorted(paths, key=lambda p: p["line"])[: req.top_k]],
            "paths_total": len(paths),
            "seed": seed,
            "purity": purity,
            "killed": [_decorate(p) for p in killed],
            "skeleton": skeleton,
            "operators": req.operators.model_dump(),
            "user_entities": user_entities,
            "user_verbs": user_verbs,
            "ext_hash": ext_hash or EMPTY_EXT_HASH,
            "retrieval": bool(req.retrieve_consequence),
            "topic_lock": bool(req.topic_lock),
            "informative_pref": bool(req.informative_pref),
            "specific_pref": bool(req.specific_pref),
        }

    total = sum(p["weight"] for p in alive)
    for p in alive:
        p["probability"] = round(p["weight"] / total, 6)
    alive.sort(key=lambda p: (-p["probability"], p["line"]))
    for p in alive:
        _decorate(p)
    for p in killed:
        _decorate(p)

    # P21 诚实诊断（阶段 F）：只记录，不参与任何权重或选择。
    # 正交残差 = 后果方向超出"实体+动作"张成空间的比例；
    # 熵 = 载荷是否散在很多维上。阈值等真实问句分布回来再定。
    p21_diag = None
    if p21_ctx is not None:
        p21_diag = []
        for p in alive[:8]:
            g = p["glyphs"]
            audit = p21_ctx.engine.distrust(g["entity"], g["action"], g["consequence"])
            p21_diag.append({"line": p["line"], **audit})

    rng = random.Random(int(seed, 16))
    # 问句带了谓语：单句坍缩也要优先落在用户问的那件事上（"电子发光"不能印成"电子停止"）。
    # 但"理解/看见"这类是提问方式，不是被问的物理动作：优先它们会印出"磁场看见"病句。
    verb_set = {g["word"] for g in verb_glyphs} - PERCEPTION_VERBS
    if conditioned:
        # P8 条件级联的候选是几十条概率相近的路径（最高约 0.02），按原概率抽样
        # 等于随机印一句、常常丢掉问句里的实体。这里只在"落在最高档实体上"的
        # 路径里抽，权重按概率平方：问句的东西一定在，同一 seed 仍可复现。
        best_entity = alive[0]["glyphs"]["entity"]
        peak = alive[0]["probability"] or 1e-9
        pool = [p for p in alive
                if p["glyphs"]["entity"] == best_entity and p["probability"] >= peak * 0.5]
        # 问句带了谓语：宁可这次印得差一点，也要印用户问的那件事（"电子发光"）
        if verb_set:
            verb_pool = [p for p in pool if p["glyphs"]["action"] in verb_set]
            pool = verb_pool or pool
        collapse = rng.choices(pool, weights=[(p["probability"] / peak) ** 2 for p in pool], k=1)[0]
    else:
        collapse_pool = alive
        if verb_set:
            verb_pool = [p for p in alive if p["glyphs"]["action"] in verb_set]
            collapse_pool = verb_pool or alive
        collapse = rng.choices(collapse_pool,
                               weights=[p["probability"] for p in collapse_pool], k=1)[0]

    band, desc = _style_band(req.style_deg)
    result = {
        "collapsed": True,
        "printed_collapse": collapse["rendered"],
        "top_resonant_line": alive[0]["rendered"],
        "top_probability": alive[0]["probability"],
        "top_badges": alive[0]["badges"],
        **({"p21_diagnostics": p21_diag} if p21_diag is not None else {}),
        **({"p21_pool": p21_pool} if p21_pool is not None else {}),
        "style_band": band,
        "style_desc": desc,
        "paths": alive[: req.top_k],
        "paths_total": len(paths),
        "paths_alive": len(alive),
        "seed": seed,
        "bank_id": vault.bank_id,
        "purity": purity,
        "killed": killed,
        "skeleton": skeleton,
        "operators": req.operators.model_dump(),
        "user_entities": user_entities,
        "user_verbs": user_verbs,
        "ext_hash": ext_hash or EMPTY_EXT_HASH,
        "conditioned": conditioned,
        "retrieval": bool(req.retrieve_consequence),
        "topic_lock": bool(req.topic_lock),
        "specific_pref": bool(req.specific_pref),
    }
    if topology is not None:
        result["topology_hash"] = topology.topo_hash
    return result
