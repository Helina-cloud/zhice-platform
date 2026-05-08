"""Streamlit Cloud：把 Secrets 写入 os.environ，并在无向量索引时尝试 ingest。"""

from __future__ import annotations

import os
from pathlib import Path


def inject_streamlit_secrets() -> None:
    """在导入依赖 OPENAI_* / EMBEDDING_* 的配置之前调用。"""
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError:
        return
    try:
        sec = st.secrets
    except RuntimeError:
        return
    if not sec:
        return

    def _set_env(key: str, val: object) -> None:
        if isinstance(val, (str, int, float, bool)):
            os.environ[key] = str(val)

    for key in sec:
        val = sec[key]
        if isinstance(val, dict):
            for subk, subv in val.items():
                _set_env(f"{key}_{subk}".upper(), subv)
        else:
            _set_env(str(key), val)


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

        with st.spinner("首次部署：正在根据 data/docs 构建向量索引（可能需要几分钟）…"):
            n = ingest(force_rebuild=False)
        if n <= 0:
            return False, "ingest 未生成任何分块，请检查仓库内 data/docs 是否有 .md/.txt/.pdf。"
        return True, ""
    n = ingest(force_rebuild=False)
    if n <= 0:
        return False, "ingest 未生成任何分块。"
    return True, ""
