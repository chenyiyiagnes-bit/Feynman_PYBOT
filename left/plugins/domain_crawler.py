"""旁路领域扩展器：只从维基百科摘要里摘 (实体, 动作) 边，默认关闭。

边界（比 PYBOT3 草图收紧的地方）：
- 默认 `FEYNMAN_ENABLE_EXTENDER=0`，必须显式开启 + 手动触发，不挂定时任务。
- 单实体超时、一次最多 5 个实体；失败只报错，不推测内容。
- 只访问允许名单内的维基域名；重定向落点同样校验。
- 摘要按小句过滤讣告/追悼字样（与字库的 DEATH_MARKERS 同一套判据），
  只删命中的小句；全部删光才整条 blocked_death_marker。
- 抽取按小句进行：否定/条件/假设小句整句跳过，名词化搭配（看法、观察者、
  计算器…）不算动作，尽量要求实体在该小句里出现。
- 请求之间限速，429/5xx 只重试一次并尊重 Retry-After（有上限）。
- 缓存落盘 `data/crawl_cache.json`，按状态区分 TTL（ok 24h、not_found 30d、
  error 10min）；缓存键含主机名。
- 每次抓取留 run_id 与 `edge_provenance`，可按 run 回滚；这两项和 crawl_log
  都不参与 ext_hash。
- 只写 `data/domain_extensions.json`，从不碰 `feynman_style_vault.json`。
"""

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

from plugins.feynman_vault import DEATH_MARKERS
from plugins.feynman_typesetter import (
    SAFE_COGNITIVE_ACTIONS, resolve_entity_prototype,
)
from plugins.domain_expander import (
    load_domain_extensions_with_hash, update_extensions,
)

ENABLED_ENV = "FEYNMAN_ENABLE_EXTENDER"

WIKI_HOST = os.getenv("QBOT_WIKI_HOST", "zh.wikipedia.org").strip() or "zh.wikipedia.org"
WIKI_FALLBACK_HOST = (
    os.getenv("QBOT_WIKI_FALLBACK_HOST", "en.wikipedia.org").strip()
)
_EXTRA_HOSTS = tuple(
    h.strip() for h in os.getenv("QBOT_WIKI_ALLOWED_HOSTS", "").split(",") if h.strip()
)
ALLOWED_HOSTS = tuple(dict.fromkeys(
    (WIKI_HOST, WIKI_FALLBACK_HOST) + _EXTRA_HOSTS))

SUMMARY_PATH = "/api/rest_v1/page/summary/{title}"
SEARCH_PATH = "/w/rest.php/v1/search/page?q={query}&limit=1"
LANGLINK_PATH = "/w/api.php?action=query&format=json&prop=langlinks&lllang=zh&titles={title}"

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


TIMEOUT_SECONDS = _env_float("QBOT_CRAWL_TIMEOUT", 5.0)
MAX_ENTITIES_PER_CALL = max(1, _env_int("QBOT_CRAWL_MAX_ENTITIES", 5))
MAX_EDGES_PER_ENTITY = 3
SUMMARY_MAX_CHARS = 3000
MIN_INTERVAL_SECONDS = max(0.0, _env_float("QBOT_CRAWL_INTERVAL", 0.25))
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRY_TIMES = 1
RETRY_AFTER_CAP = 5.0
USER_AGENT = "FeynmanQBot/2.0 (local typesetter; contact: local user)"

CACHE_PATH = Path(os.getenv("QBOT_FEYNMAN_CRAWL_CACHE", "data/crawl_cache.json"))
CACHE_TTL_OK = 24 * 3600
CACHE_TTL_MISS = 30 * 24 * 3600
CACHE_TTL_TRANSIENT = 10 * 60
MAX_RUNS = 200

_cache: Dict[str, dict] = {}
_cache_loaded = False
_cache_lock = threading.Lock()
_last_status: Dict[str, dict] = {}

_rate_lock = threading.Lock()
_last_request_at = 0.0

_CJK = re.compile(r"[㐀-鿿]")
_CLAUSE_END = re.compile(r"(?<=[。！？!?；;，,])|\n+")

# 否定、条件、假设：这些句子的动作不是事实，整句丢掉
_NEGATION_MARKERS = (
    "不应", "不会", "不能", "不是", "不可", "不为", "没有", "并未", "尚无",
    "无法", "未必", "从未", "绝不", "尚未", "并非", "假如", "如果", "若是",
    "假设", "所谓", "除非", "无论", "不管", "即使", "尽管", "以免", "防止",
)
# 名词化搭配不是动作
_NOUN_COMPOUNDS = {
    "看法", "想法", "做法", "说法", "用法", "吃法", "写法", "读法",
    "学者", "读者", "作者", "编者", "观察者", "观察法", "观察力",
    "计算器", "计算法", "记忆力", "想象力", "推动力", "理解力",
    "感光度", "分辨率", "作用力", "支持力", "吸引力", "排斥力",
}
_COMPOUND_SUFFIXES = "法性者器力度感观"


def enabled() -> bool:
    return os.getenv(ENABLED_ENV, "0").strip() == "1"


def _has_death_marker(text: str) -> bool:
    return any(m in text for m in DEATH_MARKERS)


# ---------------------------------------------------------------- 缓存


def _ttl(status: Optional[str]) -> int:
    if status == "ok":
        return CACHE_TTL_OK
    if status == "not_found":
        return CACHE_TTL_MISS
    return CACHE_TTL_TRANSIENT


def _load_cache() -> None:
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    for key, entry in data.items():
        if isinstance(entry, dict) and "at" in entry:
            _cache[str(key)] = entry


def _save_cache() -> None:
    with _cache_lock:
        snapshot = dict(_cache)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(CACHE_PATH.parent), prefix=".crawl_tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_path, CACHE_PATH)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception:
        # 缓存失败不影响抓取；下次重新联网
        pass


def clear_cache() -> None:
    """清空内存与磁盘缓存（测试和手工重置用）。"""
    global _cache_loaded
    with _cache_lock:
        _cache.clear()
    _last_status.clear()
    _cache_loaded = True
    try:
        if CACHE_PATH.exists():
            CACHE_PATH.unlink()
    except Exception:
        pass


def _cache_key(entity: str) -> str:
    return f"{WIKI_HOST}|{entity}"


# ---------------------------------------------------------------- HTTP


def allowed_host(host: Optional[str]) -> bool:
    return bool(host) and host in ALLOWED_HOSTS


def _url(host: str, path: str) -> str:
    if not allowed_host(host):
        raise ValueError(f"host not in allowlist: {host}")
    return f"https://{host}{path}"


def _throttle() -> None:
    global _last_request_at
    if MIN_INTERVAL_SECONDS <= 0:
        return
    with _rate_lock:
        wait = _last_request_at + MIN_INTERVAL_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _http_get(client, url: str):
    if client is not None:
        return client.get(
            url, headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS, follow_redirects=True)
    import httpx  # 延迟导入：没装 httpx 时模块仍可导入，抓取如实报错
    with httpx.Client() as c:
        return c.get(
            url, headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_SECONDS, follow_redirects=True)


def _retry_after(resp) -> Optional[float]:
    headers = getattr(resp, "headers", None) or {}
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    try:
        return max(0.0, min(float(raw), RETRY_AFTER_CAP))
    except (TypeError, ValueError):
        return None


def _check_final_host(resp) -> None:
    final = getattr(resp, "url", None)
    if final is None:
        return
    host = urlsplit(str(final)).hostname
    if host and not allowed_host(host):
        raise ValueError(f"redirect outside allowlist: {host}")


def _get_json(client, host: str, path: str) -> Tuple[int, Optional[dict]]:
    url = _url(host, path)
    _throttle()
    resp = _http_get(client, url)
    for _ in range(RETRY_TIMES):
        if resp.status_code not in RETRY_STATUSES:
            break
        delay = _retry_after(resp)
        time.sleep(delay if delay is not None else 0.5)
        _throttle()
        resp = _http_get(client, url)
    try:
        _check_final_host(resp)
    except ValueError:
        return 0, None
    data = None
    if resp.status_code == 200:
        try:
            data = resp.json()
        except Exception:
            data = None
    return resp.status_code, data


def _summary_text(title: str, client, host: str) -> Tuple[Optional[str], str]:
    code, data = _get_json(
        client, host, SUMMARY_PATH.format(title=quote(title, safe="")))
    if code == 404:
        return None, "not_found"
    if code != 200 or not isinstance(data, dict):
        return None, "error"
    if data.get("type") in ("disambiguation", "list"):
        return None, "disambiguation"
    extract = (data.get("extract") or "").strip()[:SUMMARY_MAX_CHARS]
    if not extract:
        return None, "not_found"
    return extract, "ok"


def _search_title(query: str, client, host: str) -> Optional[str]:
    code, data = _get_json(
        client, host, SEARCH_PATH.format(query=quote(query, safe="")))
    if code != 200 or not isinstance(data, dict):
        return None
    pages = data.get("pages") or []
    if pages and isinstance(pages[0], dict):
        title = (pages[0].get("title") or "").strip()
        return title or None
    return None


def _langlink_zh(title: str, client) -> Optional[str]:
    if not WIKI_FALLBACK_HOST:
        return None
    code, data = _get_json(
        client, WIKI_FALLBACK_HOST,
        LANGLINK_PATH.format(title=quote(title, safe="")))
    if code != 200 or not isinstance(data, dict):
        return None
    pages = (data.get("query") or {}).get("pages") or {}
    if not isinstance(pages, dict):
        return None
    for page in pages.values():
        links = (page or {}).get("langlinks") or []
        if links and isinstance(links[0], dict):
            target = (links[0].get("*") or links[0].get("title") or "").strip()
            if target:
                return target
    return None


def _entity_variants(entity: str) -> List[str]:
    out = [entity]
    if _CJK.search(entity) and not any(ch in entity for ch in "（("):
        out.append(entity + "（物理学）")
    return out


def _lookup(entity: str, client) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
    saw_error = False
    for variant in _entity_variants(entity):
        text, status = _summary_text(variant, client, WIKI_HOST)
        if status == "ok" and text:
            host = WIKI_HOST
            return text, "ok", host, _url(host, SUMMARY_PATH.format(title=quote(variant, safe="")))
        if status == "disambiguation":
            title = _search_title(variant, client, WIKI_HOST)
            if title and title != variant:
                text2, st2 = _summary_text(title, client, WIKI_HOST)
                if st2 == "ok" and text2:
                    return (text2, "ok", WIKI_HOST,
                            _url(WIKI_HOST, SUMMARY_PATH.format(title=quote(title, safe=""))))
        saw_error = saw_error or status == "error"

    if not _CJK.search(entity) and WIKI_FALLBACK_HOST:
        text, status = _summary_text(entity, client, WIKI_FALLBACK_HOST)
        if status == "ok" and text:
            zh_title = _langlink_zh(entity, client)
            if zh_title:
                text2, st2 = _summary_text(zh_title, client, WIKI_HOST)
                if st2 == "ok" and text2:
                    return (text2, "ok", WIKI_HOST,
                            _url(WIKI_HOST, SUMMARY_PATH.format(title=quote(zh_title, safe=""))))
            return (text, "ok", WIKI_FALLBACK_HOST,
                    _url(WIKI_FALLBACK_HOST,
                         SUMMARY_PATH.format(title=quote(entity, safe=""))))
        saw_error = saw_error or status == "error"

    return None, ("error" if saw_error else "not_found"), None, None


def fetch_summary_with_status(entity: str, client=None):
    """取摘要并带回状态/来源。状态：ok / not_found / error。"""
    entity = (entity or "").strip()
    if not entity:
        return None, "not_found", None, None
    _load_cache()
    key = _cache_key(entity)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - float(hit.get("at") or 0) < _ttl(hit.get("status")):
            info = {"status": hit.get("status"), "host": hit.get("host"),
                    "url": hit.get("url")}
            _last_status[entity] = info
            return hit.get("text"), hit.get("status"), hit.get("host"), hit.get("url")
    try:
        text, status, host, url = _lookup(entity, client)
    except Exception:
        text, status, host, url = None, "error", None, None
    entry = {"at": now, "text": text, "status": status, "host": host, "url": url}
    with _cache_lock:
        _cache[key] = entry
    _save_cache()
    _last_status[entity] = {"status": status, "host": host, "url": url}
    return text, status, host, url


def fetch_summary(entity: str, client=None) -> Optional[str]:
    """取中文维基摘要；命中缓存直接返回。任何失败返回 None，由调用方如实报告。"""
    text, _status, _host, _url = fetch_summary_with_status(entity, client=client)
    return text


# ---------------------------------------------------------------- 抽取


def _clauses(text: str) -> List[str]:
    return [c.strip() for c in _CLAUSE_END.split(text or "") if c and c.strip()]


def clean_summary(text: str, entity: Optional[str] = None) -> Tuple[str, int]:
    """按小句删掉讣告/追悼内容；返回 (剩余文本, 删掉的小句数)。"""
    kept: List[str] = []
    dropped = 0
    for clause in _clauses(text):
        if _has_death_marker(clause):
            dropped += 1
            continue
        kept.append(clause)
    if entity and kept:
        mentioned = [c for c in kept if entity in c]
        if mentioned:
            kept = mentioned
    return "".join(kept).strip(), dropped


def _accept_occurrence(clause: str, word: str, idx: int) -> bool:
    if word in _NOUN_COMPOUNDS:
        return False
    after = idx + len(word)
    if after < len(clause) and len(word) <= 2 and clause[after] in _COMPOUND_SUFFIXES:
        return False
    return True


def _first_ok(clause: str, word: str) -> int:
    start = 0
    while True:
        idx = clause.find(word, start)
        if idx < 0:
            return -1
        if _accept_occurrence(clause, word, idx):
            return idx
        start = idx + 1


def harvest_edges(text: str, allowed_actions: set,
                  entity: Optional[str] = None) -> List[str]:
    """小句级抽取动作字粒；否定/条件句跳过，名词化搭配不算，最多三个。"""
    words = [w for w in allowed_actions if w and w != entity]
    clauses = _clauses(text)
    if entity:
        mentioned = [c for c in clauses if entity in c]
        if mentioned:
            clauses = mentioned
    found: List[Tuple[int, int, int, str]] = []
    for ci, clause in enumerate(clauses):
        if any(m in clause for m in _NEGATION_MARKERS):
            continue
        for word in words:
            idx = _first_ok(clause, word)
            if idx >= 0:
                found.append((ci, idx, -len(word), word))
    found.sort(key=lambda item: (item[0], item[1], item[2]))
    picked: List[str] = []
    for _ci, _idx, _neg_len, word in found:
        if any(word in p for p in picked):
            continue
        picked.append(word)
        if len(picked) >= MAX_EDGES_PER_ENTITY:
            break
    return picked


# ---------------------------------------------------------------- 抓取与回滚


def _new_run_id(entity: str) -> str:
    raw = f"{entity}:{time.time_ns()}:{os.getpid()}".encode("utf-8")
    return "run-" + hashlib.sha1(raw).hexdigest()[:12]


def _run_record(entity: str, status: str, *, edges: List[str],
                prototype: Optional[str], info: dict, text: Optional[str] = None,
                dropped: int = 0) -> dict:
    record = {
        "run_id": _new_run_id(entity),
        "at": int(time.time()),
        "entity": entity,
        "status": status,
        "source": info.get("host") or "zh.wikipedia",
        "edges": list(edges),
        "prototype": prototype,
        "dropped_sentences": dropped,
    }
    if info.get("url"):
        record["source_url"] = info["url"]
    if text is not None:
        record["summary_chars"] = len(text)
        record["summary_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return record


def _persist_run(record: dict) -> None:
    def mutate(ext: dict) -> None:
        ext.setdefault("crawl_runs", []).append(record)
        runs = ext["crawl_runs"]
        if len(runs) > MAX_RUNS:
            del runs[:-MAX_RUNS]
        ext.setdefault("crawl_log", {})[record["entity"]] = {
            "at": record["at"], "run_id": record["run_id"],
            "status": record["status"], "source": record["source"],
            "source_url": record.get("source_url"),
            "edges": list(record.get("edges") or []),
            "prototype": record.get("prototype"),
            "dropped_sentences": record.get("dropped_sentences", 0),
        }

    update_extensions(mutate)


def crawl_entity(entity: str, allowed_actions: set,
                 client=None) -> dict:
    """抓一个实体 → 只写扩展库。返回 {entity, status, edges, prototype, run_id}。"""
    if not enabled():
        return {"entity": entity, "status": "disabled", "edges": [], "prototype": None}
    if _has_death_marker(entity):
        return {"entity": entity, "status": "blocked_death_marker", "edges": [],
                "prototype": None}

    text = fetch_summary(entity, client=client)
    info = _last_status.get(entity) or {}
    if not text:
        record = _run_record(entity, "no_summary", edges=[], prototype=None, info=info)
        _persist_run(record)
        return {"entity": entity, "status": "no_summary", "edges": [],
                "prototype": None, "run_id": record["run_id"]}

    cleaned, dropped = clean_summary(text, entity=entity)
    if not cleaned:
        record = _run_record(entity, "blocked_death_marker", edges=[],
                             prototype=None, info=info, dropped=dropped)
        _persist_run(record)
        return {"entity": entity, "status": "blocked_death_marker", "edges": [],
                "prototype": None, "run_id": record["run_id"],
                "dropped_sentences": dropped}

    edges = harvest_edges(cleaned, allowed_actions, entity=entity)
    family, _ = resolve_entity_prototype(entity)
    prototype = family if family != "default_physical" else None
    # 摘要里反复出现认知动词、而实体本身没标注时，补一个 abstract_info 标签让排版别配物理动作
    if prototype is None and any(a in SAFE_COGNITIVE_ACTIONS for a in edges):
        prototype = "abstract_info"

    if not edges and prototype is None:
        record = _run_record(entity, "no_usable_edges", edges=[], prototype=None,
                             info=info, text=cleaned, dropped=dropped)
        _persist_run(record)
        return {"entity": entity, "status": "no_usable_edges", "edges": [],
                "prototype": None, "summary_chars": len(text),
                "run_id": record["run_id"], "dropped_sentences": dropped}

    record = _run_record(entity, "ok", edges=edges, prototype=prototype,
                         info=info, text=cleaned, dropped=dropped)

    def mutate(ext: dict) -> None:
        if prototype:
            ext.setdefault("entity_prototypes", {})[entity] = prototype
        cooc = {tuple(e) for e in ext.get("cooccurrence_edges") or []}
        prov = ext.setdefault("edge_provenance", {})
        baseline = ext.setdefault("edge_baseline", {})
        for action in edges:
            key = f"{entity}|{action}"
            if (entity, action) in cooc and key not in prov:
                baseline[key] = True
            ids = prov.setdefault(key, [])
            if record["run_id"] not in ids:
                ids.append(record["run_id"])
            cooc.add((entity, action))
        ext["cooccurrence_edges"] = [list(e) for e in sorted(cooc)]
        ext.setdefault("crawl_runs", []).append(record)
        if len(ext["crawl_runs"]) > MAX_RUNS:
            del ext["crawl_runs"][:-MAX_RUNS]
        ext.setdefault("crawl_log", {})[entity] = {
            "at": record["at"], "run_id": record["run_id"],
            "status": "ok", "source": record["source"],
            "source_url": record.get("source_url"),
            "edges": list(edges), "prototype": prototype,
            "dropped_sentences": dropped,
        }
        # crawl_log / crawl_runs / edge_provenance 均不参与 ext_hash

    update_extensions(mutate)
    return {"entity": entity, "status": "ok", "edges": edges,
            "prototype": prototype, "summary_chars": len(cleaned),
            "run_id": record["run_id"], "dropped_sentences": dropped}


def crawl_many(entities: List[str], allowed_actions: set,
               client=None) -> List[dict]:
    """去重后按顺序抓取，一次最多 MAX_ENTITIES_PER_CALL 个。"""
    ordered: List[str] = []
    seen = set()
    for raw in entities or ():
        entity = (raw or "").strip()
        if not entity or entity in seen:
            continue
        seen.add(entity)
        ordered.append(entity)
        if len(ordered) >= MAX_ENTITIES_PER_CALL:
            break
    return [crawl_entity(e, allowed_actions, client=client) for e in ordered]


def rollback_run(run_id: str) -> dict:
    """撤销一次抓取：去掉该 run 的边；别处（另一 run 或基线）也贡献的边保留。"""
    result = {"run_id": run_id, "ok": False, "removed_edges": [],
              "kept_edges": [], "reason": ""}

    def mutate(ext: dict) -> None:
        runs = ext.get("crawl_runs") or []
        hit = [r for r in runs if r.get("run_id") == run_id]
        if not hit:
            result["reason"] = "no_such_run"
            return
        ext["crawl_runs"] = [r for r in runs if r.get("run_id") != run_id]
        prov = ext.get("edge_provenance") or {}
        baseline = ext.get("edge_baseline") or {}
        cooc = {tuple(e) for e in ext.get("cooccurrence_edges") or []}
        touched_entities = {r.get("entity") for r in hit}
        for record in hit:
            entity = record.get("entity")
            for action in record.get("edges") or []:
                key = f"{entity}|{action}"
                ids = [x for x in prov.get(key, []) if x != run_id]
                if ids:
                    prov[key] = ids
                    result["kept_edges"].append([entity, action])
                    continue
                prov.pop(key, None)
                if baseline.get(key):
                    result["kept_edges"].append([entity, action])
                    continue
                if (entity, action) in cooc:
                    cooc.discard((entity, action))
                    result["removed_edges"].append([entity, action])
                else:
                    result["kept_edges"].append([entity, action])
        ext["cooccurrence_edges"] = [list(e) for e in sorted(cooc)]
        log = ext.get("crawl_log") or {}
        for entity in touched_entities:
            if (log.get(entity) or {}).get("run_id") != run_id:
                continue
            previous = next(
                (r for r in reversed(ext["crawl_runs"]) if r.get("entity") == entity),
                None)
            if previous:
                log[entity] = {
                    "at": previous.get("at"), "run_id": previous.get("run_id"),
                    "status": previous.get("status"),
                    "source": previous.get("source"),
                    "source_url": previous.get("source_url"),
                    "edges": list(previous.get("edges") or []),
                    "prototype": previous.get("prototype"),
                    "dropped_sentences": previous.get("dropped_sentences", 0),
                }
            else:
                log.pop(entity, None)
        result["ok"] = True

    update_extensions(mutate)
    return result


def crawl_stats(limit: int = 20) -> dict:
    """读扩展库里的抓取记录：状态计数、实体数、边数、最近若干次。"""
    data, _ = load_domain_extensions_with_hash(enable_extensions=True)
    runs = data.get("crawl_runs") or []
    log = data.get("crawl_log") or {}
    limit = max(1, min(int(limit or 20), 100))
    return {
        "total_runs": len(runs),
        "status_counts": dict(Counter(r.get("status") or "unknown" for r in runs)),
        "entities": len(log),
        "entity_status_counts": dict(Counter(
            (v or {}).get("status") or "unknown" for v in log.values()
            if isinstance(v, dict))),
        "edges": len(data.get("cooccurrence_edges") or []),
        "recent": list(reversed(runs))[:limit],
    }


__all__ = ["enabled", "fetch_summary", "fetch_summary_with_status",
           "harvest_edges", "clean_summary", "crawl_entity", "crawl_many",
           "rollback_run", "crawl_stats", "clear_cache", "allowed_host",
           "MAX_ENTITIES_PER_CALL", "MAX_EDGES_PER_ENTITY"]
