"""公开网络搜索：优先使用 `ddgs`（duckduckgo-search 继任包），避免 LangChain 封装里的过时 backend。"""

from __future__ import annotations

from langchain_core.tools import tool


def _ddgs_search(query: str, max_results: int = 8) -> list[str]:
    try:
        from ddgs import DDGS  # pip install ddgs
    except ImportError:
        from duckduckgo_search import DDGS  # 兼容仅安装 duckduckgo-search 的环境

    chunks: list[str] = []
    with DDGS() as ddgs:
        # 不传 backend='api'，由库默认 auto，减少弃用告警
        it = ddgs.text(query, max_results=max_results)
        for r in it:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            href = (r.get("href") or "").strip()
            if not (title or body):
                continue
            line = "\n".join(x for x in (title, body, href) if x)
            chunks.append(line)
            if len(chunks) >= max_results:
                break
    return chunks


@tool
def web_search(query: str) -> str:
    """在互联网上检索公开信息，用于新闻、竞品、术语补充；企业内部制度应优先用 search_company_knowledge。"""
    q = (query or "").strip()
    if not q:
        return "（空查询）"
    try:
        snippets = _ddgs_search(q)
        if not snippets:
            return (
                "未检索到有效结果（可能被频率限制、地区网络限制或 DuckDuckGo 不可用）。"
                "可稍后重试或使用浏览器查询。"
            )
        return "\n\n---\n\n".join(snippets)
    except Exception as e:  # noqa: BLE001
        return (
            "网络搜索失败。请确认：`pip install ddgs`，且本机能访问外网。"
            f" 详情: {e}"
        )
