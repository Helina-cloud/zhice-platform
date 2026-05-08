"""智策 Agent 工具集：知识检索、业务 API、安全计算、公开搜索。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.tools import BaseTool

from tools.analytics import query_business_metric
from tools.calculator import safe_calculator
from tools.knowledge import search_company_knowledge
from tools.search_web import web_search
from tools.weather_openmeteo import weather_forecast


def build_zhice_tools() -> Sequence[BaseTool]:
    """组装默认工具列表，供 ReAct Agent 绑定。"""
    return [
        search_company_knowledge,
        safe_calculator,
        query_business_metric,
        weather_forecast,
        web_search,
    ]
