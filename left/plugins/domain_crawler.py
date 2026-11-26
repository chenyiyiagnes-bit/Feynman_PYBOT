"""旁路领域扩展器：只从维基百科摘要里摘 (实体, 动作) 边，默认关闭。

边界（比 PYBOT3 草图收紧的地方）：
- 默认 `FEYNMAN_ENABLE_EXTENDER=0`，必须显式开启 + 手动触发，不挂定时任务。
- 5 秒超时、单实体缓存、一次最多 5 个实体；失败只报错，不推测内容。
- 摘要里出现讣告/追悼类字样整条丢弃（与字库的 DEATH_MARKERS 同一套判据）；
  元话语名词、术语、泛词一律不进库。
- 只写 `data/domain_extensions.json`，从不碰 `feynman_style_vault.json`。
"""

import os
import threading
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

from plugins.feynman_vault import DEATH_MARKERS
from plugins.feynman_typesetter import (
    SAFE_COGNITIVE_ACTIONS, resolve_entity_prototype,
)
from plugins.domain_expander import update_extensions

ENABLED_ENV = "FEYNMAN_ENABLE_EXTENDER"
WIKI_SUMMARY_URL = "https://zh.wikipedia.org/api/rest_v1/page/summary/{title}"
TIMEOUT_SECONDS = 5.0
MAX_ENTITIES_PER_CALL = 5
MAX_EDGES_PER_ENTITY = 3
SUMMARY_MAX_CHARS = 3000
CACHE_TTL_SECONDS = 24 * 3600
USER_AGENT = "FeynmanQBot/2.0 (local typesetter; contact: local user)"

_cache: Dict[str, Tuple[float, Optional[str]]] = {}
_cache_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv(ENABLED_ENV, "0").strip() == "1"


def _has_death_marker(text: str) -> bool:
    return any(m in text for m in DEATH_MARKERS)


def fetch_summary(entity: str, client: Optional[httpx.Client] = None) -> Optional[str]:
    """取中文维基摘要；命中缓存直接返回。任何失败返回 None，由调用方如实报告。"""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(entity)
        if hit and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]
    url = WIKI_SUMMARY_URL.format(title=quote(entity, safe=""))
    try:
        if client is not None:
            resp = client.get(url, headers={"User-Agent": USER_AGENT},
                              timeout=TIMEOUT_SECONDS, follow_redirects=True)
        else:
            with httpx.Client() as c:
                resp = c.get(url, headers={"User-Agent": USER_AGENT},
                             timeout=TIMEOUT_SECONDS, follow_redirects=True)
        if resp.status_code != 200:
            extract = None
        else:
            data = resp.json()
            # 消歧义页/列表页没有可用正文
            if data.get("type") in ("disambiguation",):
                extract = None
            else:
                extract = (data.get("extract") or "").strip()[:SUMMARY_MAX_CHARS] or None
    except Exception:
        extract = None
    with _cache_lock:
        _cache[entity] = (now, extract)
    return extract


def harvest_edges(text: str, allowed_actions: set) -> List[str]:
    """摘要里真实出现的动作字粒；最多 MAX_EDGES_PER_ENTITY 个，宁缺毋滥。"""
    found = [w for w in allowed_actions if len(w) >= 1 and w in text]
    # 按摘要里的出现顺序取前三个，长词同位置优先（"观察"先于"看"）
    found.sort(key=lambda w: (text.find(w), -len(w)))
    picked: List[str] = []
    for w in found:
        if any(w in p for p in picked):
            continue
        picked.append(w)
        if len(picked) >= MAX_EDGES_PER_ENTITY:
            break
    return picked


def crawl_entity(entity: str, allowed_actions: set,
                 client: Optional[httpx.Client] = None) -> dict:
    """抓一个实体 → 只写扩展库。返回 {entity, status, edges, prototype, summary_chars}。"""
    if not enabled():
        return {"entity": entity, "status": "disabled", "edges": [], "prototype": None}
    text = fetch_summary(entity, client=client)
    if not text:
        return {"entity": entity, "status": "no_summary", "edges": [], "prototype": None}
    if _has_death_marker(text) or _has_death_marker(entity):
        return {"entity": entity, "status": "blocked_death_marker", "edges": [],
                "prototype": None}

    edges = harvest_edges(text, allowed_actions)
    family, _ = resolve_entity_prototype(entity)
    prototype = family if family != "default_physical" else None
    # 摘要里反复出现认知动词、而实体本身没标注时，补一个 abstract_info 标签让排版别配物理动作
    if prototype is None and any(a in SAFE_COGNITIVE_ACTIONS for a in edges):
        prototype = "abstract_info"

    if not edges and prototype is None:
        return {"entity": entity, "status": "no_usable_edges", "edges": [],
                "prototype": None, "summary_chars": len(text)}

    def mutate(ext: dict) -> None:
        if prototype:
            ext.setdefault("entity_prototypes", {})[entity] = prototype
        cooc = {tuple(e) for e in ext.get("cooccurrence_edges") or []}
        cooc.update((entity, a) for a in edges)
        ext["cooccurrence_edges"] = [list(e) for e in sorted(cooc)]
        log = ext.setdefault("crawl_log", {})
        log[entity] = {"at": int(time.time()), "source": "zh.wikipedia",
                       "edges": edges, "prototype": prototype}
        # crawl_log 不参与 ext_hash（见 domain_expander._HASHED_FIELDS）

    update_extensions(mutate)
    return {"entity": entity, "status": "ok", "edges": edges,
            "prototype": prototype, "summary_chars": len(text)}


def crawl_many(entities: List[str], allowed_actions: set,
               client: Optional[httpx.Client] = None) -> List[dict]:
    return [crawl_entity(e, allowed_actions, client=client)
            for e in entities[:MAX_ENTITIES_PER_CALL]]


__all__ = ["enabled", "fetch_summary", "harvest_edges",
           "crawl_entity", "crawl_many", "MAX_ENTITIES_PER_CALL"]
