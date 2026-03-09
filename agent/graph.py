from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent.state import GraphState
from agent.nodes.scan import scan_files, analyze_capability
from agent.nodes.recommend import recommend_agent
from agent.nodes.chat import chat_loop, chat_respond, rescan_notify
from agent.nodes.execute import execute_mock


def route_after_chat(state: GraphState) -> str:
    """chat_loop の next_action に基づいてグラフの分岐先を決定する。"""
    match state.next_action:
        case "execute":
            return "execute_mock"
        case "rescan":
            return "rescan_notify"
        case "quit":
            return END
        case _:  # "chat"
            return "chat_respond"


def build_graph() -> StateGraph:
    """LangGraphのワークフローを構築する。

    START → scan_files → analyze_capability → recommend_agent → chat_loop
              ↑                                                 ↓ (conditional)
              │                                    ┌───────────┼───────────┐
              │                                    ↓           ↓           ↓
              └─── rescan_notify ←──        chat_respond   execute_mock   END
                                                 │
                                                 └─→ chat_loop
    """
    builder = StateGraph(GraphState)

    builder.add_node("scan_files", scan_files)
    builder.add_node("analyze_capability", analyze_capability)
    builder.add_node("recommend_agent", recommend_agent)
    builder.add_node("chat_loop", chat_loop)
    builder.add_node("chat_respond", chat_respond)
    builder.add_node("rescan_notify", rescan_notify)
    builder.add_node("execute_mock", execute_mock)

    builder.add_edge(START, "scan_files")
    builder.add_edge("scan_files", "analyze_capability")
    builder.add_edge("analyze_capability", "recommend_agent")
    builder.add_edge("recommend_agent", "chat_loop")

    builder.add_conditional_edges(
        "chat_loop",
        route_after_chat,
        {
            "chat_respond": "chat_respond",
            "rescan_notify": "rescan_notify",
            "execute_mock": "execute_mock",
            END: END,
        },
    )

    builder.add_edge("chat_respond", "chat_loop")
    builder.add_edge("rescan_notify", "scan_files")
    builder.add_edge("execute_mock", END)

    return builder
