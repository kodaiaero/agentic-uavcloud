from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent.state import GraphState
from agent.llm.client import call_gemini_chat
from agent.llm.prompts import build_system_prompt


def recommend_agent(state: GraphState) -> dict:
    """Gemini で初回の診断・提案を生成し、チャット履歴を初期化する。"""
    messages = [
        SystemMessage(content=build_system_prompt(state.inventory, state.capability)),
        HumanMessage(content="現在のファイル構成を診断して、最適な処理ルートを提案してください。"),
    ]

    response_text = call_gemini_chat(messages)
    messages.append(AIMessage(content=response_text))

    print(f"\n========== AI アドバイザー ==========\n{response_text}")
    print("====================================")

    return {
        "recommendation": response_text,
        "chat_history": messages,
    }
