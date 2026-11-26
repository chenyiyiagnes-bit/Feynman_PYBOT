import os
from typing import Any

import httpx

from shared.llm_ledger import ledger

# DeepSeek 官方 OpenAI 兼容接口（无需额外 SDK，复用项目已有的 httpx）
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")


def _get_api_key() -> str:
    # 优先 DEEPSEEK_API_KEY，兼容回退 GEMINI_API_KEY 占位
    return os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()


def client_available() -> bool:
    return bool(_get_api_key())


def chat_completion(
    model: str,
    prompt: str,
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    response_format: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> str:
    """同步调用 DeepSeek chat/completions，返回首个 choice 的文本内容。

    抛出的异常由调用方捕获并回退到离线仿真内核。
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY")

    # 预算闸门在真正发请求之前判定：超限直接抛 BudgetExhausted，调用方回退离线内核
    ledger.grant(model)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    resp = httpx.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
