from __future__ import annotations

import os

from langchain_core.messages import AIMessage, ToolMessage

from agent.tools import ALL_TOOLS


def call_gemini_chat(messages: list, enable_tools: bool = False) -> str:
    """Vertex AI 経由で Gemini をチャット形式で呼び出す。

    enable_tools=True の場合、ツール呼び出しに対応する ReAct ループを実行する。
    """

    project = os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION")
    model = os.environ.get("VERTEX_MODEL")

    if not project or not location or not model:
        print("\n[WARNING] .env に VERTEX_PROJECT / VERTEX_LOCATION / VERTEX_MODEL が設定されていません。")
        print("  → .env.example を参考に .env を作成してください。")
        return _fallback_response()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            project=project,
            location=location,
        )

        if enable_tools:
            return _react_loop(llm, messages)
        else:
            response = llm.invoke(messages)
            return response.content

    except Exception as e:
        print(f"\n[WARNING] Vertex AI 呼び出しに失敗しました: {e}")
        print("  → 以下を確認してください:")
        print(f"    1. gcloud auth application-default login を実行済みか")
        print(f"    2. プロジェクト '{project}' で Vertex AI API が有効か")
        print(f"    3. リージョン '{location}' が正しいか")
        return _fallback_response()


def _react_loop(llm, messages: list, max_iterations: int = 5) -> str:
    """Gemini がツール呼び出しを要求する限りループし、最終テキスト回答を返す。"""

    llm_with_tools = llm.bind_tools(ALL_TOOLS, tool_choice="auto")
    tool_map = {t.name: t for t in ALL_TOOLS}

    working_messages = list(messages)

    for i in range(max_iterations):
        response: AIMessage = llm_with_tools.invoke(working_messages)
        working_messages.append(response)

        if not response.tool_calls:
            return response.content

        # ツール呼び出しを実行
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  🔧 ツール実行: {tool_name}({tool_args})")

            tool_fn = tool_map.get(tool_name)
            if tool_fn:
                result = tool_fn.invoke(tool_args)
            else:
                result = {"error": f"未知のツール: {tool_name}"}

            working_messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # max_iterations に達した場合、最後のレスポンスを返す
    final = llm_with_tools.invoke(working_messages)
    return final.content


def _fallback_response() -> str:
    return (
        "（Gemini API 未接続のためルールベースで回答）\n"
        "ファイルスキャン結果とルート判定を参照し、\n"
        "実行可能な最上位ルートの利用を推奨します。"
    )
