"""
RAG 问答链（LCEL）：Runnable 组合检索、提示词与聊天模型。
对话模型走 OpenAI 兼容客户端（可在 .env 配置 DeepSeek：OPENAI_API_BASE + CHAT_MODEL）。
用法：在项目根目录  python rag_chain.py "你的问题"
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import get_settings
from openai_http import (
    normalize_openai_compat_base,
    openai_sync_http_client,
    without_empty_openai_env_keys,
)
from retriever import build_hybrid_retriever


def format_docs(docs: list) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs, start=1):
        src = (d.metadata or {}).get("source", "unknown")
        parts.append(f"[片段 {i} | 来源: {src}]\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain():
    settings = get_settings()
    retriever = build_hybrid_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是「智策」企业知识助手。请仅根据用户提供的上下文片段回答问题；"
                "若上下文不足以回答，请明确说明「根据现有知识库无法回答」，不要编造事实。"
                "回答使用简体中文，条理清晰。",
            ),
            (
                "human",
                "上下文：\n{context}\n\n问题：{question}",
            ),
        ]
    )

    ck, cb = settings.chat_llm_params()
    http_client = openai_sync_http_client(settings)
    llm_kw: dict = {
        "model": settings.chat_model,
        "temperature": 0.2,
        "api_key": ck,
        "base_url": normalize_openai_compat_base(cb),
        "http_client": http_client,
    }
    with without_empty_openai_env_keys():
        llm = ChatOpenAI(**llm_kw)

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(description="ZhiCe RAG 问答")
    p.add_argument("question", nargs="?", default="", help="要问的问题")
    args = p.parse_args()
    q = args.question.strip() or (sys.argv[1] if len(sys.argv) > 1 else "").strip()
    if not q:
        q = input("请输入问题: ").strip()
    if not q:
        raise SystemExit("未提供问题。")

    ans = build_rag_chain().invoke({"question": q})
    print(ans)
