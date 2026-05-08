"""业务数据查询工具：通过 HTTP GET 调用外部指标/报表类 API（示例化实现）。"""

from __future__ import annotations

import json
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

from config import get_settings


@tool
def query_business_metric(metric_name: str) -> str:
    """查询业务系统暴露的指标接口（JSON）。metric_name 为指标标识，如 revenue_daily、active_users。"""
    settings = get_settings()
    base = (settings.analytics_api_base or "").strip().rstrip("/")
    if not base:
        return (
            "未配置 ANALYTICS_API_BASE，无法调用业务 API。"
            "在 .env 中设置后，将向 `ANALYTICS_API_BASE/metrics/<指标名>` 发起 GET。"
        )

    url = f"{base}/metrics/{quote(metric_name, safe='')}"
    try:
        with httpx.Client(timeout=settings.analytics_timeout_s) as client:
            r = client.get(url)
            r.raise_for_status()
            if "application/json" in r.headers.get("content-type", ""):
                return json.dumps(r.json(), ensure_ascii=False)[:8000]
            return r.text[:8000]
    except Exception as e:  # noqa: BLE001
        return f"请求失败: {e}"
