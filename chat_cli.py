#!/usr/bin/env python3
"""
멀티턴 채팅 CLI — LangChain 기반 대화 메모리 챗봇
Usage: python chat_cli.py
명령어: /reset (초기화), /stats (토큰 통계), /quit (종료)
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# notebooks/05_web_chatbot.ipynb의 LangChain 예제와 같은 설정
load_dotenv()
MODEL = "gpt-6-luna"  # reasoning_effort="none" 을 함께 지정한다 (교재 5.3)
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '확인 필요'라고만 답하세요."
DEFAULT_TEMPERATURE = 0.3
MAX_HISTORY_TURNS = 2

llm = ChatOpenAI(
    model=MODEL,
    reasoning_effort="none",
    temperature=DEFAULT_TEMPERATURE,
)

# 노트북/Streamlit 예제처럼 대화 기록은 role/content dict로 보관하고,
# 모델을 호출할 때 LangChain 메시지 클래스로 변환한다.
history = []
usage_log = []


def to_langchain_messages(messages):
    """대화 기록(dict)을 system + LangChain 메시지 목록으로 변환한다."""
    langchain_messages = [SystemMessage(content=SYSTEM)]
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"지원하지 않는 메시지 role: {role}")
    return langchain_messages


def _total_tokens(response):
    """LangChain 응답의 사용량 메타데이터에서 전체 토큰 수를 읽는다."""
    usage = getattr(response, "usage_metadata", None) or {}
    total = usage.get("total_tokens")
    if total is not None:
        return total

    # 구버전 langchain-openai는 OpenAI 사용량을 response_metadata에 담는다.
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage", {})
    return usage.get("total_tokens")


def reset():
    """대화 히스토리와 통계 초기화."""
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def chat(message: str, temperature: float = DEFAULT_TEMPERATURE) -> str:
    """사용자 메시지에 대한 챗봇 응답 생성."""
    # 새 turn을 시작하기 전에 가장 오래된 user/assistant 대화 한 쌍을 제거한다.
    if len(history) >= MAX_HISTORY_TURNS * 2:
        del history[:2]

    history.append({"role": "user", "content": message})
    try:
        # invoke() 및 메시지 변환은 노트북 05의 LangChain 흐름과 동일하다.
        model = llm if temperature == DEFAULT_TEMPERATURE else llm.bind(temperature=temperature)
        response = model.invoke(to_langchain_messages(history))
    except Exception:
        # 호출에 실패하면 미완성 user 메시지를 기록에 남기지 않는다.
        history.pop()
        raise

    answer = response.content
    history.append({"role": "assistant", "content": answer})
    usage_log.append(_total_tokens(response))
    return answer


def show_stats():
    """토큰 및 비용 통계 출력."""
    if not usage_log:
        print("(아직 메시지 없음)")
        return

    known_usage = [tokens for tokens in usage_log if tokens is not None]
    print(f"  턴 수: {len(usage_log)}")
    if not known_usage:
        print("  토큰 사용량: LangChain 응답에서 제공되지 않음")
        return

    PRICE = 0.30 / 1_000_000  # 입출력 혼합 평균
    total = sum(known_usage)
    per_turn = total / len(known_usage)

    if len(known_usage) != len(usage_log):
        print(f"  사용량 확인된 턴: {len(known_usage)}/{len(usage_log)}")
    print(f"  누적 토큰: {total}")
    print(f"  누적 비용: ${total * PRICE:.5f}")
    print(f"  평균 턴당: {per_turn:.1f} tokens")
    print(f"  100턴 추정: ${per_turn * 100 * PRICE:.4f}")


def main():
    """메인 CLI 루프."""
    print("=" * 60)
    print("멀티턴 AI 챗봇 CLI (LangChain)")
    print("=" * 60)
    print("명령어:")
    print("  /reset   - 대화 초기화")
    print("  /stats   - 토큰 및 비용 통계")
    print("  /persona - 페르소나 변경 (구현 예정)")
    print("  /quit    - 종료")
    print("=" * 60)
    print()

    reset()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == "/reset":
                reset()
                continue

            if user_input == "/stats":
                show_stats()
                continue

            if user_input == "/quit":
                print("(대화 종료)")
                break

            print("Bot: ", end="", flush=True)
            response = chat(user_input)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n(Ctrl+C로 종료)")
            break
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    main()
