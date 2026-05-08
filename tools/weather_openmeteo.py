"""天气预报（Open-Meteo）：免费 HTTPS，无需 Key；国内网络通常比 DuckDuckGo 稳定。"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_ZH: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "雪粒",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨性阵雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷雨",
    96: "雷雨伴冰雹",
    99: "强雷雨伴冰雹",
}


def _weather_zh(code: float | int | None) -> str:
    if code is None:
        return "未知"
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "未知"
    return _WMO_ZH.get(c, f"天气代码{c}")


def _format_location_head(hit: dict, fallback: str) -> str:
    disp = hit.get("name", fallback)
    admin1 = hit.get("admin1") or ""
    country = hit.get("country") or ""
    head = f"「{disp}」"
    if admin1:
        head += f"（{admin1}"
        if country:
            head += f"，{country}"
        head += "）"
    elif country:
        head += f"（{country}）"
    return head


def _geocode_first(client: httpx.Client, loc: str) -> dict | None:
    gr = client.get(
        GEOCODE_URL,
        params={"name": loc, "count": 3, "language": "zh"},
    )
    gr.raise_for_status()
    results = gr.json().get("results") or []
    return results[0] if results else None


def _forecast_daily(client: httpx.Client, lat: float, lon: float, fdays: int) -> dict:
    fr = client.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": (
                "weathercode,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max"
            ),
            "forecast_days": fdays,
            "timezone": "Asia/Shanghai",
        },
    )
    fr.raise_for_status()
    return fr.json().get("daily") or {}


def _day_line(
    day: str,
    code: float | int | None,
    hi: float | None,
    lo: float | None,
    p: float | None,
) -> str:
    seg = f"- {day}：{_weather_zh(code)}"
    if hi is not None and lo is not None:
        seg += f"，气温约 {lo:.0f}°C～{hi:.0f}°C"
    if p is not None:
        seg += f"，降水概率约 {p:.0f}%"
    return seg


def _format_daily_lines(daily: dict) -> list[str]:
    dates = daily.get("time") or []
    codes = daily.get("weathercode") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    pop = daily.get("precipitation_probability_max") or []
    return [
        _day_line(
            day,
            codes[i] if i < len(codes) else None,
            tmax[i] if i < len(tmax) else None,
            tmin[i] if i < len(tmin) else None,
            pop[i] if i < len(pop) else None,
        )
        for i, day in enumerate(dates)
    ]


@tool
def weather_forecast(location: str, forecast_days: int = 3) -> str:
    """查询某地未来数日天气预报（最高/最低气温、降水概率、天气概况）。location 如：珠海、北京、深圳市南山区。"""
    loc = (location or "").strip()
    if not loc:
        return "（地点为空）"
    fdays = max(1, min(int(forecast_days), 16))

    try:
        with httpx.Client(timeout=25.0) as client:
            hit = _geocode_first(client, loc)
            if hit is None:
                return f"未在地理库中找到「{loc}」，请换标准地名或英文拼写。"
            lat, lon = hit["latitude"], hit["longitude"]
            daily = _forecast_daily(client, lat, lon, fdays)
            dates = daily.get("time") or []

            head = _format_location_head(hit, loc)
            lines = [head + f" 未来 {len(dates)} 日预报（Asia/Shanghai）："]
            lines.extend(_format_daily_lines(daily))
            lines.append("（数据来源 Open-Meteo，仅供参考）")
            return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"天气接口请求失败（网络或远端异常）：{e}"
