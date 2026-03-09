from __future__ import annotations

import os


def call_gemini_chat(messages: list) -> str:
    """Vertex AI 経由で Gemini をチャット形式で呼び出す。"""

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
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"\n[WARNING] Vertex AI 呼び出しに失敗しました: {e}")
        print("  → 以下を確認してください:")
        print(f"    1. gcloud auth application-default login を実行済みか")
        print(f"    2. プロジェクト '{project}' で Vertex AI API が有効か")
        print(f"    3. リージョン '{location}' が正しいか")
        return _fallback_response()


def _fallback_response() -> str:
    return (
        "（Gemini API 未接続のためルールベースで回答）\n"
        "ファイルスキャン結果とルート判定を参照し、\n"
        "実行可能な最上位ルートの利用を推奨します。"
    )
