"""全局配置：通过环境变量与 .env 注入，供摄入、检索、RAG 链共用。"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 固定为「与本文件同目录」的 .env，避免从其它 cwd 运行 ingest 时读不到配置（此时 EMBEDDING_API_BASE 会失效）。
_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 系统里若存在空的 EMBEDDING_API_BASE=，否则会盖住 .env 里的正确值并回落到对话用 OPENAI_API_BASE
        env_ignore_empty=True,
    )

    # 对话与嵌入均走 OpenAI 兼容协议（默认官方 https://api.openai.com/v1 等；亦兼容其它网关）。
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    )
    openai_api_base: str | None = Field(default=None, validation_alias="OPENAI_API_BASE")

    chat_model: str = Field(default="gpt-4o-mini", validation_alias="CHAT_MODEL")
    embedding_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    # 嵌入单独走其他兼容网关时使用（不设则与对话共用 OPENAI_API_KEY / OPENAI_API_BASE）
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_api_base: str | None = Field(default=None, validation_alias="EMBEDDING_API_BASE")
    # openai：OpenAI 兼容 embeddings API；huggingface：本地 sentence-transformers
    embedding_provider: str = Field(default="openai", validation_alias="EMBEDDING_PROVIDER")
    huggingface_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="HUGGINGFACE_EMBEDDING_MODEL",
    )

    # 路径（规划中的 chroma_db 目录在本仓库用 FAISS 落盘，便于 Windows 开箱安装）
    project_root: Path = Field(default_factory=lambda: _PROJECT_ROOT)
    data_docs_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "docs",
        validation_alias="DATA_DOCS_DIR",
    )
    vector_store_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent / "data" / "vector_db",
        validation_alias="VECTOR_STORE_DIR",
    )

    # 分块与检索
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, validation_alias="CHUNK_OVERLAP")
    retriever_k: int = Field(default=4, validation_alias="RETRIEVER_K")

    # Agent / 业务工具（增强版）
    analytics_api_base: str | None = Field(
        default=None,
        validation_alias="ANALYTICS_API_BASE",
    )
    analytics_timeout_s: float = Field(default=15.0, validation_alias="ANALYTICS_TIMEOUT_S")

    # HTTPS 客户端（嵌入 / 对话）：企业代理 MITM、SSL 解密失败时可调低校验或关闭 trust_env
    openai_http_verify_ssl: bool = Field(default=True, validation_alias="OPENAI_HTTP_VERIFY_SSL")
    openai_http_trust_env: bool = Field(
        default=True,
        validation_alias="OPENAI_HTTP_TRUST_ENV",
    )
    openai_proxy: str | None = Field(default=None, validation_alias="OPENAI_PROXY")

    def chat_llm_params(self) -> tuple[str | None, str | None]:
        """(api_key, base_url)，供 ChatOpenAI 等对话模型使用。"""
        key = (self.openai_api_key or "").strip() or None
        base = (self.openai_api_base or "").strip() or None
        return key, base

    def embedding_llm_params(self) -> tuple[str | None, str | None]:
        """(api_key, base_url)，供 OpenAIEmbeddings；优先 EMBEDDING_API_*，否则回落对话侧。"""
        key = (self.embedding_api_key or self.openai_api_key or "").strip() or None
        eb = (self.embedding_api_base or "").strip() or None
        ob = (self.openai_api_base or "").strip() or None
        base = eb or ob
        # 对话走 DeepSeek、且单独配置了嵌入密钥但未配 base（或被空 env 盖掉）时，默认走 OpenAI 官方 embeddings
        embed_key_only = (self.embedding_api_key or "").strip() and not eb
        if embed_key_only and ob and "deepseek" in ob.lower():
            base = "https://api.openai.com/v1"
        elif not base:
            base = None
        return key, base


def get_settings() -> Settings:
    return Settings()
