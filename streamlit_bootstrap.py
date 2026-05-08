"""Streamlit Cloud：把 Secrets 写入 os.environ，并在无向量索引时尝试 ingest。"""

from __future__ import annotations

import os
from pathlib import Path


def inject_streamlit_secrets() -> None:
    """把 Streamlit Secrets（TOML）递归展开并写入 os.environ，键名为大写路径（如 OPENAI_API_KEY）。"""
    try:
        import streamlit as st  # noqa: PLC0415
        from streamlit.errors import StreamlitSecretNotFoundError  # noqa: PLC0415
    except ImportError:
        return
    try:
        sec = st.secrets
        # 本地无 .streamlit/secrets.toml 时，__len__/__bool__ 会解析并抛 StreamlitSecretNotFoundError
        if not sec:
            return
    except RuntimeError:
        return
    except StreamlitSecretNotFoundError:
        return

    def _set_env(key: str, val: object) -> None:
        # 云端从网页粘贴 Secrets 时常见首尾空格、UTF-8 BOM，会导致与本地 .env 行为不一致（嵌入 401 等）
        if isinstance(val, str):
            os.environ[key] = val.lstrip("\ufeff").strip()
        elif isinstance(val, (int, float, bool)):
            os.environ[key] = str(val)

    try:
        raw: dict = dict(sec)
    except Exception:
        raw = {k: sec[k] for k in sec}

    def _walk(prefix: str, obj: object) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                pk = f"{prefix}_{k}" if prefix else str(k)
                _walk(pk, v)
        elif prefix:
            _set_env(prefix.upper(), obj)

    _walk("", raw)


def apply_embedding_fallback_if_no_api_key() -> str:
    """
    嵌入走 OpenAI 兼容接口时必须提供 api_key。
    若未配置任何嵌入密钥，则设置 EMBEDDING_PROVIDER=huggingface，避免 OpenAIEmbeddings 初始化崩溃。
    返回非空字符串时表示已向用户展示提示文案。
    """
    from config import get_settings

    s = get_settings()
    prov = (s.embedding_provider or "openai").strip().lower()
    if prov == "huggingface":
        return ""

    ek, _ = s.embedding_llm_params()
    if (ek or "").strip():
        return ""

    os.environ["EMBEDDING_PROVIDER"] = "huggingface"
    return (
        "未检测到可用于「文本嵌入」的 API 密钥，已自动改用 **HuggingFace 本地向量模型** 构建索引"
        "（首次需下载模型，约数分钟）。若你希望继续用 OpenAI 嵌入，请在 Secrets 中配置 "
        "`EMBEDDING_API_KEY` 或可用的 `OPENAI_API_KEY`（与嵌入网关一致）。"
    )


def vector_index_missing(vector_store_dir: Path) -> bool:
    return not (vector_store_dir / "index.faiss").is_file()


def run_ingest_if_needed(*, show_streamlit_ui: bool) -> tuple[bool, str]:
    """返回 (是否成功, 说明文本)。"""
    from config import get_settings

    settings = get_settings()
    persist = Path(settings.vector_store_dir)
    if not vector_index_missing(persist):
        return True, ""

    try:
        from ingest import ingest
    except ImportError as e:
        return False, f"无法导入 ingest：{e}"

    if show_streamlit_ui:
        import streamlit as st  # noqa: PLC0415

        try:
            with st.spinner("首次部署：正在根据 data/docs 构建向量索引（可能需要几分钟）…"):
                n = ingest(force_rebuild=False)
        except Exception as e:
            # 若不捕获，Streamlit Cloud 会把异常正文涂掉，用户看不到 ingest 里的友好说明
            return False, str(e)
        if n <= 0:
            return False, "ingest 未生成任何分块，请检查仓库内 data/docs 是否有 .md/.txt/.pdf。"
        return True, ""
    try:
        n = ingest(force_rebuild=False)
    except Exception as e:
        return False, str(e)
    if n <= 0:
        return False, "ingest 未生成任何分块。"
    return True, ""
