#!/usr/bin/env python3
"""
멀티턴 채팅 CLI — OpenAI 기반 대화 메모리 챗봇
Usage: python chat_cli.py
명령어: /reset (초기화), /quit (종료)
"""

from dotenv import load_dotenv
from openai import OpenAI

# 초기화
load_dotenv()
client = OpenAI()
MODEL = "gpt-6-luna"   # 호출마다 reasoning_effort="none" 을 함께 넣는다 (교재 5.3)
MAX_HISTORY_TURNS = 2

# 상태
history = []
usage_log = []
SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 모르면 '이학윤 돼지'라고만 답하세요."


def reset():
    """대화 히스토리 초기화."""
    global history, usage_log
    history.clear()
    usage_log.clear()
    print("(대화 초기화)")


def chat(message: str, temperature: float = 0.3) -> str:
    """사용자 메시지에 대한 챗봇 응답 생성."""
    global history, usage_log, SYSTEM

    # 새 turn을 시작하기 전에 가장 오래된 user/assistant 대화 한 쌍을 제거한다.
    if len(history) >= MAX_HISTORY_TURNS * 2:
        del history[:2]
    
    history.append({"role": "user", "content": message})
    
    resp = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="none",
        temperature=temperature,
        messages=[{"role": "system", "content": SYSTEM}, *history],
    )
    
    answer = resp.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    usage_log.append(resp.usage.total_tokens)
    
    return answer


def show_stats():
    """토큰 및 비용 통계 출력."""
    if not usage_log:
        print("(아직 메시지 없음)")
        return
    
    PRICE = 0.30 / 1_000_000  # 입출력 혼합 평균
    total = sum(usage_log)
    per_turn = total / len(usage_log)
    
    print(f"  턴 수: {len(usage_log)}")
    print(f"  누적 토큰: {total}")
    print(f"  누적 비용: ${total * PRICE:.5f}")
    print(f"  평균 턴당: {per_turn:.1f} tokens")
    print(f"  100턴 추정: ${per_turn * 100 * PRICE:.4f}")


def main():
    """메인 CLI 루프."""
    print("=" * 60)
    print("멀티턴 AI 챗봇 CLI")
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
