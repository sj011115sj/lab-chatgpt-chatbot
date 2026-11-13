"""실습 5 Task 1 — Streamlit 챗봇.

웹 챗봇 골격을 이 저장소 안에서 직접 구현한다.
사이드바: 페르소나 선택 / temperature 슬라이더, 본문: 스트리밍 응답 + 운영 가드.

실행:
    streamlit run web/streamlit_app.py
    # 컨테이너에서 8501 포트가 자동 포워딩된다.
"""
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

PERSONAS = {
    "친절한 도우미": "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요.",
    "엄격한 사서": "당신은 매우 격식 있는 사서입니다. 짧고 단정하게 답합니다.",
    "친근한 친구": "당신은 친근한 동네 친구입니다. 반말로 편하게 답합니다.",
    "현실적인 조언을 하는 친구" : "당신은 현실적으로 조언하는 동네 친구입니다. 가장 오래된 친구처럼 반말로 현실적으로 조언해줍니다",
    "공감을 잘해주는 친구" : "당신은 공감을 매우 잘해주는 동네 친구입니다. 가장 오래된 친구처럼 반말로 친구의 기분이 상하지않게 늘 공감해줍니다"
}
MAX_CHARS = 500  # 운영 가드 — 입력 길이 제한

st.set_page_config(page_title="나만의 ChatGPT", page_icon="💬")
st.title("나만의 ChatGPT 💬 (Streamlit)")

# ── 사이드바: 페르소나 / temperature ──────────────────────────────
with st.sidebar:
    persona = st.selectbox("페르소나", list(PERSONAS.keys()))
    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.1)
    if st.button("대화 초기화"):
        st.session_state.clear()
        st.rerun()
    st.caption("FAQ:")
    # 버튼은 누른 그 회차에만 True — 선택이 남는 radio 와 달리 rerun 때 재전송되지 않는다
    # 리스트로 먼저 모두 그린다 (next() 로 단락 평가하면 누른 버튼 뒤의 버튼이 사라진다)
    clicked = [q for q in ["연차 며칠?", "재택 규정?", "API 키 관리법?"] if st.button(q)]
    faq = clicked[0] if clicked else None

llm = ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=temperature)

# ── 대화 상태 ─────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{"role", "content"}]
    st.session_state.total_tokens = 0

# 환영 메시지 (처음 1회)
if not st.session_state.messages:
    st.chat_message("assistant").write("안녕하세요! 무엇이든 물어보세요 😊")

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

# FAQ 버튼 또는 입력창
user_input = st.chat_input(f"메시지 ({MAX_CHARS}자 이내)") or faq

if user_input:
    # 운영 가드 1 — 입력 길이
    if len(user_input) > MAX_CHARS:
        st.warning(f"입력은 {MAX_CHARS}자 이내로 부탁드립니다.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # LangChain 메시지로 변환 (system + 누적 history)
    msgs = [SystemMessage(content=PERSONAS[persona])]
    for m in st.session_state.messages:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        msgs.append(cls(content=m["content"]))

    with st.chat_message("assistant"):
        try:
            # 운영 가드 2 — 스트리밍 응답
            def gen():
                for chunk in llm.stream(msgs):
                    yield chunk.content or ""
            answer = st.write_stream(gen())
        except Exception:
            # 운영 가드 3 — Stack trace 노출 금지
            answer = "죄송합니다. 일시적인 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # 운영 가드 4 — 누적 비용(근사) 표시
    st.session_state.total_tokens += len(user_input) + len(answer)
    approx_cost = st.session_state.total_tokens / 4 * (0.3 / 1_000_000)
    st.sidebar.metric("누적 비용(근사)", f"${approx_cost:.5f}")
