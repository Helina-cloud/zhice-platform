"""
混合检索：稠密向量（FAISS）+ 稀疏词面（BM25），合并去重后截断为 top-k。
BM25 语料来自与向量库相同的分块文档，保证与 ingest 结果一致。
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import get_settings
from embeddings import build_embeddings


class HybridRetriever(BaseRetriever):
    """并行调用向量检索与 BM25，按向量优先顺序合并去重。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_retriever: BaseRetriever
    bm25_retriever: BaseRetriever
    k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        _ = run_manager
        v_docs: list[Document] = self.vector_retriever.invoke(query)
        b_docs: list[Document] = self.bm25_retriever.invoke(query)

        seen: set[str] = set()
        merged: list[Document] = []

        def _key(d: Document) -> str:
            src = (d.metadata or {}).get("source", "")
            si = (d.metadata or {}).get("start_index")
            return f"{src}::{si}::{d.page_content[:240]}"

        for d in v_docs + b_docs:
            k = _key(d)
            if k in seen:
                continue
            seen.add(k)
            merged.append(d)
            if len(merged) >= self.k:
                break
        return merged


def load_faiss() -> FAISS:
    settings = get_settings()
    persist = Path(settings.vector_store_dir)
    if not (persist / "index.faiss").is_file():
        raise FileNotFoundError(
            f"未找到 FAISS 索引。请先运行: python ingest.py\n期望路径: {persist}"
        )
    return FAISS.load_local(
        str(persist),
        build_embeddings(),
        allow_dangerous_deserialization=True,
    )


def documents_from_faiss(vs: FAISS) -> list[Document]:
    """从 FAISS 附带的 docstore 取出全部 Document，供 BM25 索引。"""
    store = vs.docstore
    mapping = getattr(vs, "index_to_docstore_id", None)
    if mapping:
        ids = list(mapping.values())
        if hasattr(store, "mget"):
            got = store.mget(ids)
            return [d for d in got if d is not None]
    if hasattr(store, "_dict"):
        return list(store._dict.values())
    return []


def build_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    vs = load_faiss()
    bm25_source = documents_from_faiss(vs)
    if not bm25_source:
        raise RuntimeError(
            "向量库已加载但无法读取文档条目，请使用 python ingest.py --force 重新构建索引。"
        )

    k_vec = max(2, settings.retriever_k)
    k_bm25 = max(2, settings.retriever_k)

    vector_retriever = vs.as_retriever(search_kwargs={"k": k_vec})
    bm25_retriever = BM25Retriever.from_documents(bm25_source)
    bm25_retriever.k = k_bm25

    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        k=settings.retriever_k,
    )
