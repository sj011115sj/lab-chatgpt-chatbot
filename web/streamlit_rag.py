"""Streamlit 도서관 챗봇: MCP 검색/대출/운영시간 조회와 문서 RAG.

도서 검색, 대출 현황, 운영시간은 library_server.py를 MCP로 조회한다.
RAG 탭은 업로드한 문서만 근거로 답한다. (.md / .txt / .pdf 지원)

실행:
    streamlit run web/streamlit_rag.py
    # 컨테이너에서 8501 포트가 자동 포워딩된다.
"""
import asyncio
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

load_dotenv()

st.set_page_config(page_title="도서관 챗봇", page_icon="📚")
st.title("도서관 챗봇 📚")


async def _request_mcp(kind: str, name: str, arguments: dict | None = None) -> str:
    """MCP stdio 서버에서 도구를 호출하거나 리소스를 읽는다."""
    server = Path(__file__).resolve().parents[1] / "mcp_demo" / "library_server.py"
    params = StdioServerParameters(command=sys.executable, args=[str(server)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            if kind == "resource":
                result = await session.read_resource(AnyUrl(name))
                items = result.contents
            else:
                available_tools = await session.list_tools()
                if name not in {item.name for item in available_tools.tools}:
                    raise RuntimeError(f"MCP 서버가 {name} 도구를 제공하지 않습니다.")

                result = await session.call_tool(name, arguments or {})
                if result.isError:
                    raise RuntimeError(f"MCP 서버의 {name} 호출에 실패했습니다.")
                items = result.content

            parts = [getattr(item, "text", "") for item in (items or [])]
            parts = [part for part in parts if part]
            return "\n".join(parts) if parts else str(result)


def request_mcp(kind: str, name: str, arguments: dict | None = None) -> str:
    """Streamlit 동기 코드에서 MCP 비동기 클라이언트를 호출한다."""
    return asyncio.run(_request_mcp(kind, name, arguments))

PROMPT = ChatPromptTemplate.from_template(
    """아래 '문맥'만 근거로 한국어로 답하세요.
문맥에 답이 없으면 반드시 "문서에서 찾을 수 없습니다"라고만 답하세요.

[문맥]
{context}

[질문]
{question}
"""
)


@st.cache_resource(show_spinner="문서 인덱싱 중...")
def build_chain(file_bytes: bytes, suffix: str):
    """업로드 파일 → 청크 → FAISS → RAG 체인. (캐시: 같은 파일은 1회만 인덱싱)"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)

    vs = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)

    def fmt(ds):
        return "\n\n".join(d.page_content for d in ds)

    return (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )


book_tab, loan_tab, hours_tab, rag_tab = st.tabs(
    ["📚 도서 검색", "🪪 대출 현황", "🕒 운영시간", "📄 문서 RAG"]
)

with book_tab:
    st.caption("MCP 도서관 서버에 제목이나 저자를 검색합니다.")
    with st.form("book_search_form"):
        query = st.text_input(
            "도서 제목 또는 저자",
            placeholder="예: 해리포터, 반지의 제왕",
        )
        submitted = st.form_submit_button("도서 검색")

    if submitted:
        if not query.strip():
            st.warning("검색할 도서 제목이나 저자를 입력해 주세요.")
        else:
            try:
                with st.spinner("도서관에서 검색 중..."):
                    result = request_mcp(
                        "tool", "search_book", {"query": query.strip()}
                    )
                st.write(result)
            except Exception as exc:
                st.error(f"MCP 도서 검색에 실패했습니다: {exc}")

with loan_tab:
    st.caption("회원 ID로 현재 대출 권수와 연체 권수를 조회합니다.")
    with st.form("loan_status_form"):
        member_id = st.text_input("회원 ID", placeholder="예: M13")
        loan_submitted = st.form_submit_button("대출 현황 조회")

    if loan_submitted:
        if not member_id.strip():
            st.warning("회원 ID를 입력해 주세요.")
        else:
            try:
                with st.spinner("대출 현황을 조회 중..."):
                    result = request_mcp(
                        "tool",
                        "loan_status",
                        {"member_id": member_id.strip()},
                    )
                st.write(result)
            except Exception as exc:
                st.error(f"대출 현황 조회에 실패했습니다: {exc}")

with hours_tab:
    st.caption("MCP 도서관 리소스에서 운영시간을 읽어옵니다.")
    if st.button("운영시간 조회", key="library_hours_button"):
        try:
            with st.spinner("운영시간을 조회 중..."):
                result = request_mcp("resource", "config://library")
            st.write(result)
        except Exception as exc:
            st.error(f"운영시간 조회에 실패했습니다: {exc}")

with rag_tab:
    st.caption("업로드한 문서의 내용을 근거로 답변합니다.")
    uploaded = st.file_uploader(
        "문서 업로드 (.md / .txt / .pdf)",
        type=["md", "txt", "pdf"],
        key="rag_document",
    )

    if not uploaded:
        st.info(
            "문서를 업로드하면 그 내용으로만 답하는 챗봇이 생성됩니다. "
            "샘플: `data/sample_docs/hr_policy.md`"
        )
    else:
        chain = build_chain(uploaded.getvalue(), Path(uploaded.name).suffix.lower())
        st.success(f"'{uploaded.name}' 인덱싱 완료 — 질문해 보세요.")

        question = st.chat_input("문서 내용을 물어보세요")
        if question:
            st.chat_message("user").write(question)
            with st.chat_message("assistant"):
                st.write(chain.invoke(question))
