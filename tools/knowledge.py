"""知识库检索工具：封装混合检索器，供 Agent 在 ReAct 循环中显式调用。"""

from __future__ import annotations

from langchain_core.tools import tool

from rag_chain import format_docs
from retriever import build_hybrid_retriever

_retriever = None


def _retriever_singleton():
    global _retriever
    if _retriever is None:
        _retriever = build_hybrid_retriever()
    return _retriever


@tool
def search_company_knowledge(query: str) -> str:
    """检索企业内部知识库（PDF/Markdown 等已索引文档）。用于制度、流程、产品说明、内部术语等事实性问题。"""
    docs = _retriever_singleton().invoke(query)
    if not docs:
        return "（知识库未返回相关片段）"
    return format_docs(docs)
