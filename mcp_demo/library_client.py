"""실습 6 Task 3 — MCP 도서관 챗봇 (브라우저 UI).

Gradio 화면에서 질문하면 LangChain 에이전트가 MCP 서버의 도구와 리소스를
필요에 따라 호출한다. library_server.py 는 stdio 로 자동 실행된다.

실행:
    python mcp_demo/library_client.py
    # 브라우저에서 http://127.0.0.1:7861 열기
"""
import sys
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import AnyUrl

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# 같은 폴더의 서버를 현재 파이썬으로 실행 (CWD 와 무관하게 동작)
SERVER = Path(__file__).resolve().parent / "library_server.py"
SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=[str(SERVER)])

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)


def _text(result) -> str:
    """MCP 결과에서 사람이 읽을 텍스트를 가져온다."""
    parts = []
    for item in getattr(result, "content", []) or []:
        parts.append(getattr(item, "text", str(item)))
    return "\n".join(parts) if parts else str(result)


def _message_text(message) -> str:
    """LangChain 메시지의 텍스트를 반환한다."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


async def respond(message: str, history: list[dict]) -> str:
    """대화 기록을 에이전트에 전달하고, MCP 서버를 통해 필요한 정보를 조회한다."""
    try:
        # 이 요청의 도구 호출 동안 MCP 연결을 유지한다.
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                available_tools = await session.list_tools()
                tool_names = {item.name for item in available_tools.tools}

                @tool
                async def search_book(query: str) -> str:
                    """도서 제목이나 저자로 도서관 자료를 검색한다."""
                    if "search_book" not in tool_names:
                        return "MCP 서버가 search_book 도구를 제공하지 않습니다."
                    result = await session.call_tool("search_book", {"query": query})
                    return _text(result)

                @tool
                async def loan_status(member_id: str) -> str:
                    """회원 ID로 현재 대출 및 연체 현황을 조회한다."""
                    if "loan_status" not in tool_names:
                        return "MCP 서버가 loan_status 도구를 제공하지 않습니다."
                    result = await session.call_tool(
                        "loan_status", {"member_id": member_id}
                    )
                    return _text(result)

                @tool
                async def library_hours() -> str:
                    """도서관 운영시간을 조회한다."""
                    result = await session.read_resource(AnyUrl("config://library"))
                    return "\n".join(
                        getattr(item, "text", str(item))
                        for item in result.contents
                    )

                agent = create_agent(
                    llm,
                    [search_book, loan_status, library_hours],
                    system_prompt=(
                        "너는 친절한 도서관 도우미다. 도서 검색, 회원 대출 현황, "
                        "운영시간 질문에는 제공된 도구나 리소스를 사용해 확인한다. "
                        "조회 결과에 없는 정보는 지어내지 말고 찾을 수 없다고 말한다."
                    ),
                )

                messages = [
                    {"role": item["role"], "content": item["content"]}
                    for item in history
                    if item.get("role") in {"user", "assistant"}
                ]
                messages.append({"role": "user", "content": message})
                result = await agent.ainvoke({"messages": messages})
                return _message_text(result["messages"][-1])
    except Exception as exc:
        print(f"MCP 챗봇 오류: {exc}", file=sys.stderr)
        return "도서관 챗봇에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요."


demo = gr.ChatInterface(
    respond,
    type="messages",
    title="도서관 MCP 챗봇 📚",
    description="도서 검색, 회원 대출 현황, 도서관 운영시간을 물어보세요.",
)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
