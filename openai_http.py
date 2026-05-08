"""OpenAI 兼容 SDK 共用的 httpx 同步客户端（系统代理 / SSL / 显式代理）。"""

from __future__ import annotations

import httpx

from config import Settings


def openai_sync_http_client(settings: Settings) -> httpx.Client | None:
    """不需要自定义时返回 None，由 LangChain 使用默认行为。"""
    proxy = (settings.openai_proxy or "").strip()
    custom = (
        not settings.openai_http_verify_ssl
        or bool(proxy)
        or not settings.openai_http_trust_env
    )
    if not custom:
        return None

    kw: dict = {
        "timeout": httpx.Timeout(120.0, connect=45.0),
        "trust_env": settings.openai_http_trust_env,
    }
    if proxy:
        kw["proxy"] = proxy
    if not settings.openai_http_verify_ssl:
        kw["verify"] = False
    return httpx.Client(**kw)
