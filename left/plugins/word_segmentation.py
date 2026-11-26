"""左脑的词级分词：和右脑共用同一套规则。

右脑检索用 ``omega_mind.lang`` 的规则把问句切成词；左脑排版、挖矿、字库
构建也要切词。两边各切各的，同一个句子就会得到两套边界，桥接层对不上。
这里不重新实现分词，只把右脑的语言核心包一层给左脑用：

    from plugins.word_segmentation import segment
    segment("熵到底是什么")        # → ["熵", "到底", "是", "什么"]

规则表在 ``right/omega_mind/lang/<language>/`` 里，按语言和 profile 选。
jieba 仍然负责词性标注（``extract_entities`` 需要 n/v 标签），但它只做"给
词打标签"，不再决定词的边界；jieba 不可用时词边界照样是同一套。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "right") not in sys.path:
    sys.path.insert(0, str(ROOT / "right"))

from omega_mind import lang  # noqa: E402


def rules(language: str = "zh", profile: Optional[str] = None):
    """取一份语言规则；语言或 profile 不存在时直接抛错，不悄悄退回默认。"""
    return lang.load(language, profile)


def segment(text: str, language: str = "zh", profile: Optional[str] = None) -> List[str]:
    """把一段文本切成词，边界与右脑检索完全一致。"""
    return rules(language, profile).segment(text)


def content_words(text: str, language: str = "zh", profile: Optional[str] = None) -> List[str]:
    """只留实词：切词后去掉单字虚词和规则里的停用大词。"""
    core = rules(language, profile)
    words = core.segment(text)
    if language != "zh":
        return words
    stop = core._unigram_stop
    return [w for w in words if not (len(w) == 1 and w in stop)]
