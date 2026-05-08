"""嵌入模型构建：OpenAI 兼容 API 或本地 HuggingFace（对话用 DeepSeek 时常用后者）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.embeddings import Embeddings

from config import get_settings

if TYPE_CHECKING:
    from config import Settings


class _IsolateChatKeysEmbeddings(Embeddings):
    """
    已配置 EMBEDDING_API_KEY 时，每次嵌入请求仍可能被底层 SDK 读 os.environ['OPENAI_API_KEY']
    （对话用 DeepSeek），导致请求发到云雾却带上错误密钥 → 401。在每次 embed 调用期间暂时移出对话用环境变量。
    """

    def __init__(self, inner: Embeddings, settings: Settings) -> None:
        self._inner = inner
        self._settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from openai_http import openai_env_for_embedding_client

        with openai_env_for_embedding_client(self._settings):
            return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        from openai_http import openai_env_for_embedding_client

        with openai_env_for_embedding_client(self._settings):
            return self._inner.embed_query(text)


def _is_deepseek_base(url: str | None) -> bool:
    if not url:
        return False
    return "deepseek" in url.lower()


def build_embeddings() -> Embeddings:
    settings = get_settings()
    provider = (settings.embedding_provider or "openai").strip().lower()

    if provider == "huggingface":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "本地嵌入需要安装: pip install sentence-transformers\n"
                "或在 .env 设置 EMBEDDING_PROVIDER=openai 并配置 EMBEDDING_API_BASE（含 embeddings 的服务）。"
            ) from e
        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)

    if provider != "openai":
        raise ValueError(f"未知的 EMBEDDING_PROVIDER={provider!r}，请使用 openai 或 huggingface。")

    # 对话用 DeepSeek 时 OPENAI_API_KEY 是 DeepSeek 的，不能拿去打 OpenAI 官方嵌入 → 401
    has_embed_key = bool((settings.embedding_api_key or "").strip())
    if _is_deepseek_base(settings.openai_api_base) and not has_embed_key:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "对话为 DeepSeek 且未单独配置 EMBEDDING_API_KEY 时，嵌入会改用 HuggingFace，"
                "请安装: pip install sentence-transformers"
            ) from e
        return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)

    ek, eb = settings.embedding_llm_params()

    # 实际用于嵌入请求的 base（embedding_llm_params 会回落到 OPENAI_API_BASE）
    if _is_deepseek_base(eb):
        raise ValueError(
            "嵌入请求的 base_url 仍指向 DeepSeek（通常无 OpenAI 兼容的 embeddings，会得到 404）。\n\n"
            "请检查：\n"
            "• 项目根目录 .env 里 `EMBEDDING_API_BASE=https://api.openai.com/v1` 行首不要有 #，并已保存；"
            "文件编码建议 UTF-8（Windows 记事本另存为 UTF-8）。\n"
            "• 若系统环境变量里有空的 EMBEDDING_API_BASE=，请删掉该变量（本项目的 Settings 已启用 env_ignore_empty）。\n\n"
            "或改用本地向量：EMBEDDING_PROVIDER=huggingface ，并 pip install sentence-transformers。\n"
        )

    if not (ek or "").strip():
        raise ValueError(
            "嵌入模式为 openai，但未配置可用 API 密钥。"
            "请在 .env 或 Streamlit Secrets 中设置 EMBEDDING_API_KEY（或能与嵌入 base 共用的 OPENAI_API_KEY），"
            "或设置 EMBEDDING_PROVIDER=huggingface 使用本地向量模型。"
        )

    from langchain_openai import OpenAIEmbeddings

    from openai_http import (
        normalize_openai_compat_base,
        openai_env_for_embedding_client,
        openai_sync_http_client,
    )

    http_client = openai_sync_http_client(settings)
    kw: dict = {
        "model": settings.embedding_model,
        "api_key": ek,
        "base_url": normalize_openai_compat_base(eb),
        "http_client": http_client,
    }
    with openai_env_for_embedding_client(settings):
        inner = OpenAIEmbeddings(**kw)
    if (settings.embedding_api_key or "").strip():
        return _IsolateChatKeysEmbeddings(inner, settings)
    return inner
