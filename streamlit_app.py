"""
智策 Web 界面（Streamlit）。

本地运行：
  streamlit run streamlit_app.py

部署说明：
- Streamlit：推荐 Streamlit Community Cloud、自有服务器 Docker、Railway/Render 等（绑定仓库 + 环境变量即可）。
- Vercel：主要面向 Node/静态站点与 Serverless Functions，不直接托管 Streamlit 长连接应用。
  若必须上 Vercel：需拆成「FastAPI 后端 + 前端」或仅用 Vercel 托管静态页再调外部 API。
"""

from __future__ import annotations

import uuid

import streamlit as st

from streamlit_bootstrap import (
    apply_embedding_fallback_if_no_api_key,
    inject_streamlit_secrets,
    run_ingest_if_needed,
)


def _init_state() -> None:
    if "messages_rag" not in st.session_state:
        st.session_state.messages_rag = []
    if "messages_agent" not in st.session_state:
        st.session_state.messages_agent = []
    if "agent_graph" not in st.session_state:
        st.session_state.agent_graph = None
    if "agent_thread" not in st.session_state:
        st.session_state.agent_thread = f"ui-{uuid.uuid4().hex[:12]}"


def _get_rag_chain():
    if st.session_state.get("_rag_chain") is None:
        from rag_chain import build_rag_chain

        st.session_state._rag_chain = build_rag_chain()
    return st.session_state._rag_chain


def _get_agent_graph():
    if st.session_state.agent_graph is None:
        from agent import build_agent_app

        st.session_state.agent_graph = build_agent_app()
    return st.session_state.agent_graph


def main() -> None:
    # 必须在首次使用其它 st.* 之前调用；勿放在模块顶层，否则 bare python / IDE 导入会报 ScriptRunContext。
    st.set_page_config(page_title="智策 ZhiCe", page_icon="📘", layout="wide")

    # Streamlit Cloud：仪表盘里配置的 Secrets → 环境变量，供 pydantic-settings 读取（与本地 .env 等价）。
    inject_streamlit_secrets()

    emb_hint = apply_embedding_fallback_if_no_api_key()
    if emb_hint.strip():
        st.info(emb_hint)

    ok_idx, idx_msg = run_ingest_if_needed(show_streamlit_ui=True)
    if not ok_idx:
        st.error(
            "向量索引不可用。\n\n"
            + idx_msg
            + "\n\n请在 Streamlit Cloud「Secrets」中配置嵌入所需变量（如 "
            "`EMBEDDING_PROVIDER=huggingface` 或 OpenAI 嵌入的 KEY/BASE），然后刷新。"
        )
        st.stop()

    _init_state()

    st.title("智策 · 企业知识助手")
    st.caption("RAG 检索问答 · Agent 工具增强（天气 / 搜索 / 计算等）")

    with st.sidebar:
        st.header("模式")
        mode = st.radio(
            "选择交互方式",
            ("RAG 问答", "Agent 多轮"),
            help="RAG：每轮独立检索知识库，无工具。Agent：多轮记忆 + 工具调用。",
        )
        st.divider()
        st.markdown(
            "**部署**：`streamlit run streamlit_app.py` 或 [Streamlit Cloud](https://streamlit.io/cloud)。"
            " Vercel 不适合直接托管 Streamlit，可改用 FastAPI + 前端。"
        )
        if st.button("清空当前模式对话"):
            if mode == "RAG 问答":
                st.session_state.messages_rag = []
            else:
                st.session_state.messages_agent = []
                st.session_state.agent_graph = None
                st.session_state.agent_thread = f"ui-{uuid.uuid4().hex[:12]}"
            st.rerun()

    if mode == "RAG 问答":
        st.subheader("知识库问答（混合检索 + LCEL）")
        for msg in st.session_state.messages_rag:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("输入问题…", key="rag_input"):
            st.session_state.messages_rag.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("检索并生成中…"):
                    try:
                        chain = _get_rag_chain()
                        answer = chain.invoke({"question": prompt})
                    except Exception as e:  # noqa: BLE001
                        err = str(e)
                        hint503 = ""
                        if "503" in err or "service_unavailable" in err.lower() or "too busy" in err.lower():
                            hint503 = (
                                "\n\n提示：503 / Service unavailable 多为上游 LLM 或中转繁忙，"
                                "请稍后重试或临时更换模型/服务商。"
                            )
                        answer = (
                            f"调用失败：{e}\n\n"
                            "本地请确认 `.env` 与 `python ingest.py`；"
                            "云端请在 Secrets 中配置密钥并重试。"
                            + hint503
                        )
                st.markdown(answer)
            st.session_state.messages_rag.append({"role": "assistant", "content": answer})
    else:
        st.subheader("Agent（ReAct + 本页内多轮记忆）")
        st.caption(f"本会话 thread_id：`{st.session_state.agent_thread}`")

        from agent import run_chat_turn

        for msg in st.session_state.messages_agent:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("输入消息…", key="agent_input"):
            st.session_state.messages_agent.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("思考与工具调用中…"):
                    try:
                        graph = _get_agent_graph()
                        answer = run_chat_turn(
                            graph,
                            st.session_state.agent_thread,
                            prompt,
                            verbose=False,
                        )
                    except Exception as e:  # noqa: BLE001
                        answer = f"Agent 调用失败：{e}"
                st.markdown(answer)
            st.session_state.messages_agent.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
