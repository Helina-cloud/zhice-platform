"""
文档摄入与向量索引构建：扫描 data/docs，支持 Markdown / 纯文本 / PDF，分块后写入 FAISS。
运行：在项目根目录执行  python ingest.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings
from embeddings import build_embeddings


def _load_documents(docs_dir: Path) -> list[Document]:
    """多源加载：按扩展名选用 TextLoader / PyPDFLoader。"""
    patterns: list[tuple[str, type, dict]] = [
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.pdf", PyPDFLoader, {}),
    ]
    out: list[Document] = []
    for glob, cls, kwargs in patterns:
        loader = DirectoryLoader(
            str(docs_dir),
            glob=glob,
            loader_cls=cls,
            loader_kwargs=kwargs,
            silent_errors=True,
            show_progress=True,
        )
        out.extend(loader.load())
    return out


def ingest(force_rebuild: bool = False) -> int:
    settings = get_settings()
    docs_dir = Path(settings.data_docs_dir)
    persist = Path(settings.vector_store_dir)

    if not docs_dir.is_dir():
        docs_dir.mkdir(parents=True, exist_ok=True)
        print(f"已创建文档目录（请放入待索引文件）: {docs_dir}")
        return 0

    raw_docs = _load_documents(docs_dir)
    if not raw_docs:
        print(f"未在 {docs_dir} 下找到可加载的 .md / .txt / .pdf 文档。")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    splits = splitter.split_documents(raw_docs)
    if not splits:
        return 0

    if force_rebuild and persist.exists():
        shutil.rmtree(persist)

    embeddings = build_embeddings()

    try:
        db = FAISS.from_documents(splits, embeddings)
    except Exception as e:
        err = str(e).lower()
        if "404" in err or "notfound" in err.replace(" ", ""):
            raise RuntimeError(
                "嵌入接口返回 404：多见于对话用了 DeepSeek，却把 embeddings 请求发到同一 base URL，"
                "或 EMBEDDING_MODEL 在当前网关不存在。\n"
                "处理：在 .env 设置 EMBEDDING_API_BASE + EMBEDDING_API_KEY 指向提供 embeddings 的服务，"
                "或设置 EMBEDDING_PROVIDER=huggingface 并 pip install sentence-transformers。"
            ) from e
        if (
            "429" in err
            or "insufficient_quota" in err
            or "rate_limit" in err
            or "rate limit" in err
        ):
            raise RuntimeError(
                "OpenAI 嵌入接口返回 429 / 配额不足（insufficient_quota）：账户可用额度或计费未开通，"
                "与智策代码无关。\n\n"
                "可选处理：\n"
                "1) 登录 https://platform.openai.com 检查 Billing / Usage / 限额；\n"
                "2) 不消费 OpenAI 嵌入：在 .env 设 EMBEDDING_PROVIDER=huggingface，执行 "
                "`pip install sentence-transformers`，再 `python ingest.py --force`。\n"
            ) from e
        if (
            "401" in err
            or "invalid_api_key" in err
            or "incorrect api key" in err
            or "authentication" in err
        ):
            s = get_settings()
            ek_explicit = bool((s.embedding_api_key or "").strip())
            eb_explicit = bool((s.embedding_api_base or "").strip())
            _k, eff_base = s.embedding_llm_params()
            host = urlparse(eff_base or "").netloc or "（未能解析 base）"
            raise RuntimeError(
                "嵌入接口返回 401：当前密钥未被该网关接受（Key、Base、路径须同属一家服务）。\n\n"
                "排查：\n"
                "• Secrets / .env 里是否真有顶层键 `EMBEDDING_API_KEY`、`EMBEDDING_API_BASE`（嵌套节须展开成这类名字，见 deploy_streamlit_cloud.txt）。\n"
                "• Base 一般为 OpenAI 兼容服务的根，例如 `https://api.openai.com/v1`（末尾 `/v1` 常需要）。\n"
                "• 密钥是否对应「嵌入」权限；不要把 DeepSeek 聊天密钥填进 OpenAI 嵌入。\n"
                "• 本地 ingest 正常而仅 Streamlit Cloud 报 401：在 Secrets 增加 "
                "`OPENAI_HTTP_TRUST_ENV=false`，排除托管环境里 HTTP(S)_PROXY 干扰鉴权头；保存后 **Redeploy**。\n"
                "• 部分中转 API 会按**出口 IP**限制：家用网络能通过，数据中心出口可能被拒（有时也表现为 401）。"
                "可向服务商确认是否允许云端调用，或 Secrets 设 `EMBEDDING_PROVIDER=huggingface` 走本地向量。\n"
                "• **勿**写 `OPENAI_API_KEY=\"\"` 占位：空字符串仍写入环境变量，少数 SDK 会与嵌入密钥混淆；不需要请删掉该行。\n\n"
                f"【诊断·不含密钥】已单独配置 EMBEDDING_API_KEY：{'是' if ek_explicit else '否'}；"
                f"已单独配置 EMBEDDING_API_BASE：{'是' if eb_explicit else '否'}；"
                f"实际请求主机：{host}；OPENAI_HTTP_TRUST_ENV={s.openai_http_trust_env}\n"
                "若「已单独配置」为否，说明应用未读到你的嵌入变量，请检查 Secrets 键名并重新部署。"
            ) from e
        if (
            "ssl" in err
            or "connection error" in err
            or "connecterror" in err.replace(" ", "")
            or "unexpected_eof" in err.replace(" ", "")
            or "eof occurred in violation" in err
            or "apiconnectionerror" in err.replace(" ", "")
            or "proxy" in err
        ):
            raise RuntimeError(
                "连接嵌入接口失败（TLS/代理/网络）：常见于公司 HTTPS 代理、SSL 解密或中转线路不稳定。\n\n"
                "可依次尝试：\n"
                "1) 当前终端暂时关闭系统代理再 ingest：`set HTTPS_PROXY=` `set HTTP_PROXY=`（PowerShell 用 "
                "`Remove-Item Env:HTTPS_PROXY`）；或在 .env 设 OPENAI_HTTP_TRUST_ENV=false "
                "（禁止 httpx 读取环境变量里的代理，直连 API）。\n"
                "2) 必须走代理时：在 .env 设置 OPENAI_PROXY=http://用户:密码@主机:端口，并确保代理信任链正确。\n"
                "3) 仅限排查时可设 OPENAI_HTTP_VERIFY_SSL=false（降低安全性）。\n"
                "4) 完全不走云端嵌入：EMBEDDING_PROVIDER=huggingface + pip install sentence-transformers。\n"
            ) from e
        raise
    persist.mkdir(parents=True, exist_ok=True)
    db.save_local(str(persist))

    print(f"索引完成：{len(splits)} 条分块写入 {persist}（FAISS index.faiss + index.pkl）")
    return len(splits)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="ZhiCe 文档摄入")
    p.add_argument(
        "--force",
        action="store_true",
        help="清空已有 vector_db 后全量重建",
    )
    args = p.parse_args()
    ingest(force_rebuild=args.force)
