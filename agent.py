"""
智策增强版 — ReAct Agent（LangGraph）与会话记忆。

- 使用 LangGraph `create_react_agent`：模型在「思考 → 选工具 → 观察」循环中调度工具。
- `MemorySaver` + `thread_id`：同一线程内多轮对话保留状态（面试中可对比生产用的 Postgres / Redis Checkpointer）。

用法（项目根目录）：
  python agent.py "单个问题"
  python agent.py --repl                  # 多轮对话（同一窗口内连续输入；exit 退出）
  python agent.py --thread demo --repl     # 指定会话 ID，与其它终端会话隔离
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings

# 须在导入 langgraph / langchain checkpoint 链之前注册。
try:
    _dep = importlib.import_module("langchain_core._api.deprecation")
    _lc_pending = getattr(_dep, "LangChainPendingDeprecationWarning", None)
    if _lc_pending is not None:
        warnings.filterwarnings("ignore", category=_lc_pending)
except ImportError:
    pass
# 匹配整句警告文案（部分版本上仅 category 或仅前缀匹配不生效）
warnings.filterwarnings("ignore", message=r".*[Dd]efault value of `allowed_objects`.*")
warnings.filterwarnings("ignore", category=Warning, message=r"allowed_objects")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# 该条 PendingDeprecation 在 langgraph 首次 import checkpoint serde 时发出；catch_warnings 最稳。
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

from config import get_settings
from openai_http import (
    normalize_openai_compat_base,
    openai_sync_http_client,
    without_empty_openai_env_keys,
)
from tools import build_zhice_tools
from tools.weather_openmeteo import weather_forecast

SYSTEM_PROMPT = """你是「智策」企业级智能知识助手（增强版）。

工作方式（ReAct）：
1. 先判断用户问题需要知识库、业务数据、精确计算还是公开信息。
2. 需要时依次调用工具；不要凭空编造数字、政策条文或接口返回值。
3. 知识库工具返回的是检索片段，请据此归纳；若片段不足，明确说明不确定。
4. 涉及多步任务时，先拆解子目标，再逐步调用工具验证，最后给出简洁结论。
5. 回答使用简体中文。
6. 询问「某地天气 / 气温 / 降雨」等：优先调用 weather_forecast(location=城市名)；其它实时新闻、资讯再调用 web_search。禁止未调用工具就让用户自己去天气 App。若工具失败再简要说明原因。
7. 若用户消息里已附带「【系统已通过 weather_forecast…」段落，说明天气接口结果已在上下文中，请直接据此回答，勿再说无法查询。"""


def _clean_weather_location(loc: str) -> str:
    """去掉误粘在地名上的「明天/今天」等时间词（避免 珠海明天天气 → 珠海明天）。"""
    s = loc.strip()
    tail = re.compile(
        r"(?:明天|后天|今天|今日|这周末|周末|天气|气温|怎么样|如何|的)+$"
    )
    head = re.compile(r"^(?:的|在|查|查询|请问|帮忙|我想)+")
    for _ in range(4):
        t = tail.sub("", s).strip()
        h = head.sub("", t).strip()
        if h == s:
            break
        s = h
    return s


def _guess_weather_location(text: str) -> str | None:
    """从中文口语里抠地名；失败则返回 None（不强行猜）。"""
    if not any(k in text for k in ("天气", "气温", "下雨", "降雨", "下雪", "降温", "台风")):
        return None
    # 地名组用非贪婪，避免「珠海」+「明天」+「天气」被整块吃进 group1
    patterns = [
        r"(?:今天|今日|明天|后天|这周末)([\u4e00-\u9fa5]{2,10}?)(?:市|区|县)?的?(?:天气|气温)",
        r"(?:今天|今日|明天|后天|这周末)(?:的)?([\u4e00-\u9fa5]{2,10}?)(?:市|区|县)?的?(?:天气|气温)",
        r"([\u4e00-\u9fa5]{2,10}?)(?:市|区|县)?(?:今天|今日|明天|后天)?的?(?:天气|气温)",
        r"([\u4e00-\u9fa5]{2,10}?)(?:市|区|县)?.{0,4}?天气",
    ]
    skip = {"怎么样", "如何", "请问", "帮忙", "查询", "我想", "能不能"}
    for p in patterns:
        m = re.search(p, text.strip())
        if not m:
            continue
        loc = _clean_weather_location(m.group(1))
        if len(loc) >= 2 and loc not in skip:
            return loc
    return None


def _augment_with_weather_if_needed(user_text: str, verbose: bool) -> str:
    """部分模型/中转不按 OpenAI tool_calls 规范返回，导致工具从未执行；天气类问题在此强制拉一次 Open-Meteo。"""
    loc = _guess_weather_location(user_text)
    if not loc:
        return user_text
    try:
        raw = weather_forecast.invoke({"location": loc, "forecast_days": 4})
    except Exception as e:  # noqa: BLE001
        raw = f"（weather_forecast 调用异常：{e}）"
    if verbose:
        print(f"[zhice-agent] 已直连 weather_forecast(Open-Meteo), location={loc!r}", file=sys.stderr)
    return (
        f"{user_text}\n\n"
        f"【以下已由系统调用 weather_forecast（Open-Meteo）取得，请据此用中文简洁回答用户，勿编造】\n{raw}"
    )


def _dump_message_trace(messages: list, verbose: bool) -> None:
    if not verbose:
        return
    print("[zhice-agent] ---------- message trace ----------", file=sys.stderr)
    for i, m in enumerate(messages):
        name = type(m).__name__
        extra = ""
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            extra = f" tool_calls={m.tool_calls!r}"
        if isinstance(m, ToolMessage):
            extra = f" name={getattr(m, 'name', '')!r}"
        body = (m.content if isinstance(m.content, str) else str(m.content))[:300]
        print(f"  [{i}] {name}{extra}\n      {body!r}", file=sys.stderr)
    print("[zhice-agent] -------------------------------------", file=sys.stderr)


def build_agent_app():
    settings = get_settings()
    ck, cb = settings.chat_llm_params()
    if not ck:
        raise ValueError(
            "未配置 API 密钥：请在 .env 中设置 OPENAI_API_KEY（或 DEEPSEEK_API_KEY）后再运行 Agent。"
        )
    http_client = openai_sync_http_client(settings)
    llm_kw: dict = {
        "model": settings.chat_model,
        "temperature": 0.1,
        "api_key": ck,
        "base_url": normalize_openai_compat_base(cb),
        "http_client": http_client,
    }
    with without_empty_openai_env_keys():
        llm = ChatOpenAI(**llm_kw)
    tools = list(build_zhice_tools())
    checkpointer = MemorySaver()
    return create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def run_chat_turn(
    graph,
    thread_id: str,
    user_text: str,
    *,
    verbose: bool = False,
) -> str:
    """在已有 graph（含 MemorySaver）上追加一轮用户消息。"""
    config = {"configurable": {"thread_id": thread_id}}
    payload = _augment_with_weather_if_needed(user_text, verbose)
    result = graph.invoke(
        {"messages": [HumanMessage(content=payload)]},
        config,
    )
    messages = result.get("messages") or []
    _dump_message_trace(messages, verbose)
    for m in reversed(messages):
        if isinstance(m, AIMessage) and (m.content or "").strip():
            if isinstance(m.content, str):
                return m.content.strip()
            parts = [p.get("text", "") for p in m.content if isinstance(p, dict)]
            return "\n".join(parts).strip()
    return "(未产生文本回复)"


def run_chat(thread_id: str, user_text: str, *, verbose: bool = False) -> str:
    """单进程单次问答（每次调用新建 graph，无跨调用记忆）。"""
    return run_chat_turn(build_agent_app(), thread_id, user_text, verbose=verbose)


def run_repl(thread_id: str, *, verbose: bool = False) -> None:
    """同一进程内多轮对话：MemorySaver 持续生效；关闭窗口或退出后即清空。"""
    graph = build_agent_app()
    print(
        "已进入多轮对话（会话 thread_id=%s）。输入 exit / quit / q 或 Ctrl+Z+Enter 结束。\n"
        % (thread_id,),
        file=sys.stderr,
    )
    while True:
        try:
            line = input("你: ")
        except EOFError:
            print(file=sys.stderr)
            break
        text = line.strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit", "q", "bye"):
            break
        reply = run_chat_turn(graph, thread_id, text, verbose=verbose)
        print("智策:", reply)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZhiCe ReAct Agent")
    parser.add_argument("--thread", default="default", help="会话线程 ID，用于多轮记忆隔离")
    parser.add_argument(
        "-i",
        "--repl",
        action="store_true",
        help="多轮对话模式：不退出进程，连续输入；记忆仅在本窗口本次运行有效",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="在 stderr 打印是否预取天气、以及 LangGraph 消息链（便于确认工具/API 是否被调用）",
    )
    parser.add_argument("message", nargs="?", default="", help="用户消息（与 --repl 互斥时忽略）")
    args = parser.parse_args()

    if args.repl:
        run_repl(args.thread, verbose=args.verbose)
        raise SystemExit(0)

    text = (args.message or "").strip()
    if not text:
        text = input("请输入: ").strip()
    if not text:
        raise SystemExit("消息为空。若要多轮对话请使用: python agent.py --repl")
    print(run_chat(args.thread, text, verbose=args.verbose))
