from __future__ import annotations

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import GraphState, ROUTE_LABELS
from agent.llm.client import call_gemini_chat


def chat_loop(state: GraphState) -> dict:
    """ユーザー入力を受け取り、次のアクションを判定する。"""
    available = _get_available_routes(state.capability)

    print("\n========== 対話モード ==========")
    if available:
        print(f"  ルート実行: {', '.join(available)} を入力")
    print("  質問:       自由にテキストを入力")
    print("  再スキャン: rescan と入力")
    print("  終了:       quit と入力")
    print("================================")

    user_input = input("\n> ").strip()
    upper = user_input.upper()

    if upper in ("QUIT", "Q", "EXIT"):
        return {"user_input": user_input, "next_action": "quit"}

    if upper in ("RESCAN", "RS", "再スキャン"):
        return {"user_input": user_input, "next_action": "rescan"}

    if upper in available:
        print(f"\n→ ルート{upper}（{ROUTE_LABELS[upper]}）を選択しました。")
        return {"user_input": user_input, "next_action": "execute", "selected_route": upper}

    return {"user_input": user_input, "next_action": "chat"}


def chat_respond(state: GraphState) -> dict:
    """ユーザーの自由質問に Gemini で回答する。"""
    history = list(state.chat_history)
    history.append(HumanMessage(content=state.user_input))

    response_text = call_gemini_chat(history)
    history.append(AIMessage(content=response_text))

    print(f"\n========== AI アドバイザー ==========\n{response_text}")
    print("====================================")

    return {"chat_history": history}


def rescan_notify(state: GraphState) -> dict:
    """再スキャン前にユーザーに通知し、チャット履歴をリセットする。"""
    print("\n🔄 ディレクトリを再スキャンします...")
    return {"chat_history": []}


def _get_available_routes(cap) -> list[str]:
    return [r for r, attr in [("A", "route_a"), ("B", "route_b"), ("C", "route_c"), ("D", "route_d")] if getattr(cap, attr)]
