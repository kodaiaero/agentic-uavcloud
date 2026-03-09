"""
Agentic UAV Cloud — ドローン測量データ診断PoC
LangGraph + Gemini (Vertex AI) による対話型ファイル診断・処理ルート提案CLI
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from agent import build_graph
from agent.state import GraphState


def main():
    if len(sys.argv) < 2:
        print("使い方: uv run main.py <ディレクトリパス>")
        print("例:     uv run main.py /path/to/drone_data")
        sys.exit(1)

    target_dir = sys.argv[1]
    print(f"\n🛩️  Agentic UAV Cloud — ドローンデータ診断（対話モード）")
    print(f"対象ディレクトリ: {target_dir}")

    app = build_graph().compile()
    app.invoke(GraphState(target_dir=target_dir))


if __name__ == "__main__":
    main()
