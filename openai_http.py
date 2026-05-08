"""OpenAI 兼容 SDK 共用的 httpx 同步客户端（系统代理 / SSL / 显式代理）。"""

from __future__ import annotations

import os
from contextlib import contextmanager

import httpx

from config import Settings


@contextmanager
def without_empty_openai_env_keys():
    """
    若环境里有 OPENAI_API_KEY=""（空字符串），部分 OpenAI SDK / LangChain 仍会优先采用该变量，
    覆盖构造函数里传入的 api_key，导致向网关发出无效 Authorization（嵌入常见 401）。
    构建客户端前后应暂时移除此类空变量。
    """
    popped: dict[str, str] = {}
    for key in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        v = os.environ.get(key)
        if v is not None and not str(v).strip():
            popped[key] = v
            del os.environ[key]
    try:
        yield
    finally:
        os.environ.update(popped)


@contextmanager
def openai_env_for_embedding_client(settings: Settings):
    """
    构建 OpenAIEmbeddings 时：部分 OpenAI SDK 会优先读环境里的 OPENAI_API_KEY，
    忽略构造函数里的 api_key=EMBEDDING_API_KEY。云端同时配 DeepSeek 对话 + 云雾嵌入时，
    会把 DeepSeek 密钥发给 yunwu → 401；本地若未设 OPENAI_API_KEY 则不会出现。

    已单独配置 EMBEDDING_API_KEY 时，在构造嵌入客户端期间暂时移出 OPENAI_API_KEY / DEEPSEEK_API_KEY。
    """
    popped: dict[str, str] = {}
    explicit_embed = bool((settings.embedding_api_key or "").strip())
    keys = ("OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    if explicit_embed:
        for key in keys:
            if key in os.environ:
                popped[key] = os.environ.pop(key)
    try:
        with without_empty_openai_env_keys():
            yield
    finally:
        os.environ.update(popped)


def normalize_openai_compat_base(url: str | None) -> str | None:
    """去掉首尾空白与末尾 `/`，减少部分网关对 `.../v1` vs `.../v1/` 处理不一致的问题。"""
    if not url:
        return url
    return str(url).strip().rstrip("/")


def openai_sync_http_client(settings: Settings) -> httpx.Client:
    """
    始终返回显式 httpx 客户端，便于统一 timeout / SSL / 代理行为。

    本地 ingest 正常而云端 401 时，可在 Secrets 设 `OPENAI_HTTP_TRUST_ENV=false`，
    避免托管环境里的 HTTP(S)_PROXY 干扰 Authorization（详见 deploy_streamlit_cloud.txt）。
    """
    kw: dict = {
        "timeout": httpx.Timeout(120.0, connect=45.0),
        "trust_env": settings.openai_http_trust_env,
    }
    proxy = (settings.openai_proxy or "").strip()
    if proxy:
        kw["proxy"] = proxy
    if not settings.openai_http_verify_ssl:
        kw["verify"] = False
    return httpx.Client(**kw)
