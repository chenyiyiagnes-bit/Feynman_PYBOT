"""量子活字印刷引擎：三槽位字粒 → 并行路径振幅 → 观测者心境投影 → 坍缩印句。

数学约定（与主系统的相位语义保持一致）：
- 每条路径 = 主语字粒 × 动作字粒 × 归宿字粒，振幅取乘积、相位相加；
- 观测者心境 m 是三相相干基矢，投影 = cos(Σθ − 3m)，只有同向分量能落墨；
- 套话字粒取负振幅，路径整体反相，任何心境下都被 max(0, ·) 拦掉——不靠额外系数硬压；
- 相邻字粒相位差进入相干因子 (1+cosΔθ)/2，反相相邻词自然阻尼；
- 采样用 (bank_id, 心境, nonce) 派生的种子，拖回同一角度必然复现同一句。

庄严声明：这是可解释的相位启发式，不是对真实量子态的模拟。干涉只发生在
「路径自身相位 vs 观测基矢」之间；不同候选句之间不叠加（测量结果是正交的）。
"""

import hashlib
import json
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from plugins.quantum_tokenizer import (
    DEFAULT_MAX_PER_SLOT,
    FontBank,
    QGlyph,
    quantize_corpus_to_glyphs,
)

STORE_PATH = Path(os.getenv("QBOT_FONT_BANK_PATH", "data/font_banks.json"))
STORE_LIMIT = 32
MOOD_RANGE = (0.0, 180.0)


class QuantizeRequest(BaseModel):
    corpus_text: str = Field(..., min_length=1, max_length=20000)
    max_per_slot: int = Field(DEFAULT_MAX_PER_SLOT, ge=2, le=8)
    force_refresh: bool = False


class PrintRequest(BaseModel):
    bank_id: str
    observer_mood_deg: float = Field(90.0, ge=0.0, le=180.0)
    nonce: int = Field(0, ge=0, le=1_000_000)
    top_k: int = Field(10, ge=1, le=50)


class FontBankStore:
    """字库缓存：内存 + 落盘，按 bank_id 取用，最多 32 盘。"""

    def __init__(self, path: Path = STORE_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._banks: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.is_file():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._banks = raw
        except Exception:
            self._banks = {}

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._banks, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    def get(self, bank_id: str) -> Optional[FontBank]:
        with self._lock:
            entry = self._banks.get(bank_id)
            return FontBank.model_validate(entry["bank"]) if entry else None

    def put(self, bank: FontBank) -> None:
        with self._lock:
            self._banks[bank.bank_id] = {
                "stored_at": time.time(),
                "bank": bank.model_dump(),
            }
            if len(self._banks) > STORE_LIMIT:
                oldest = sorted(self._banks, key=lambda k: self._banks[k]["stored_at"])
                for stale in oldest[: len(self._banks) - STORE_LIMIT]:
                    self._banks.pop(stale, None)
            self._flush()

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._banks)

    def clear(self) -> int:
        with self._lock:
            n = len(self._banks)
            self._banks = {}
            self._flush()
            return n


bank_store = FontBankStore()


def _seed(bank_id: str, mood_deg: float, nonce: int) -> int:
    digest = hashlib.sha256(f"{bank_id}|{mood_deg:.1f}|{nonce}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _coherence(g1: QGlyph, g2: QGlyph) -> float:
    delta = math.radians(g2.phase_deg - g1.phase_deg)
    return 0.5 + 0.5 * (1.0 + math.cos(delta)) / 2.0   # 0.5 ~ 1.0


def _evaluate_path(g1: QGlyph, g2: QGlyph, g3: QGlyph, mood_deg: float) -> dict:
    k = 3
    mood_rad = math.radians(mood_deg)
    amp_product = g1.amp * g2.amp * g3.amp
    coherence = _coherence(g1, g2) * _coherence(g2, g3)
    phase_total_deg = (g1.phase_deg + g2.phase_deg + g3.phase_deg) % 360.0
    delta = math.radians(phase_total_deg) - k * mood_rad

    magnitude = amp_product * coherence
    projection = magnitude * math.cos(delta)
    imaginary = magnitude * math.sin(delta)
    cliche = any(g.cliche or g.amp < 0 for g in (g1, g2, g3))

    weight = 0.0 if cliche else max(0.0, projection) ** 2

    tags = {g1.tag, g2.tag, g3.tag}
    if cliche:
        note = "🚫 套话字粒反相振幅：任何心境下均被否决"
    elif g1.tag == g3.tag:
        note = f"✨ 首尾纠缠共振 (#{g1.tag})"
    elif len(tags & {"quantum", "logic"}) == 3:
        note = "⚛️ 三相逻辑同频"
    elif g2.tag in (g1.tag, g3.tag):
        note = f"🔗 局部相干衔接 (#{g2.tag})"
    else:
        note = "⚡ 跨界意象张力"

    return {
        "sentence": f"{g1.word}{g2.word}{g3.word}",
        "components": [g1.word, g2.word, g3.word],
        "tags": [g1.tag, g2.tag, g3.tag],
        "phase_deg": round(phase_total_deg, 1),
        "magnitude": round(magnitude, 4),
        "real": round(projection, 4),
        "imag": round(imaginary, 4),
        "prob_weight": weight,
        "cliche": cliche,
        "note": note,
    }


def _print_sync(bank: FontBank, mood_deg: float, nonce: int, top_k: int) -> dict:
    mood_deg = min(MOOD_RANGE[1], max(MOOD_RANGE[0], float(mood_deg)))
    paths = [
        _evaluate_path(g1, g2, g3, mood_deg)
        for g1 in bank.subjects
        for g2 in bank.verbs
        for g3 in bank.objects
    ]

    total = sum(p["prob_weight"] for p in paths)
    for p in paths:
        p["probability"] = round(p["prob_weight"] / total * 100, 2) if total > 0 else 0.0
    paths.sort(key=lambda p: (-p["probability"], p["sentence"]))

    printed = ""
    if total > 0:
        rng = random.Random(_seed(bank.bank_id, mood_deg, nonce))
        printed = rng.choices(
            [p["sentence"] for p in paths], weights=[p["prob_weight"] for p in paths], k=1
        )[0]

    return {
        "bank_id": bank.bank_id,
        "observer_mood_deg": mood_deg,
        "seed": _seed(bank.bank_id, mood_deg, nonce),
        "collapsed": bool(printed),
        "printed_collapse": printed,
        "top_resonant_line": paths[0]["sentence"] if paths else "",
        "message": "" if printed else "当前心境基矢与所有字粒正交，波函数完全相消——请微调心境旋钮或补充更丰富的语料。",
        "paths_total": len(paths),
        "paths": paths[:top_k],
    }


def run_quantum_print(req: PrintRequest) -> dict:
    bank = bank_store.get(req.bank_id)
    if bank is None:
        raise KeyError(f"字库 {req.bank_id} 不存在或已过期，请先重新铸字（/movable_type/quantize）")
    return _print_sync(bank, req.observer_mood_deg, req.nonce, req.top_k)


async def run_quantize(req: QuantizeRequest) -> dict:
    text = req.corpus_text.strip()
    if not text:
        raise ValueError("语料为空")
    # 先按离线确定性 digest 探测缓存（在线/离线同语料不同 bank_id，不会串味）
    from plugins.quantum_tokenizer import corpus_digest
    probe = corpus_digest(text, req.max_per_slot, "online")
    probe_offline = corpus_digest(text, req.max_per_slot, "offline")
    if not req.force_refresh:
        for pid in (probe, probe_offline):
            cached = bank_store.get(pid)
            if cached is not None:
                return {"cache_hit": True, "font_bank": cached.model_dump(),
                        "bank_id": cached.bank_id, "source_summary": cached.source_summary}
    bank = await quantize_corpus_to_glyphs(text, max_per_slot=req.max_per_slot)
    bank_store.put(bank)
    return {"cache_hit": False, "font_bank": bank.model_dump(),
            "bank_id": bank.bank_id, "source_summary": bank.source_summary}


def sweep_moods(bank_id: str, moods=(0, 45, 90, 135, 170), nonce: int = 0) -> dict:
    """调试/评估辅助：同一盘活字在不同心境下各自印出什么。"""
    bank = bank_store.get(bank_id)
    if bank is None:
        raise KeyError(bank_id)
    return {int(m): _print_sync(bank, float(m), nonce, 3)["printed_collapse"] for m in moods}
