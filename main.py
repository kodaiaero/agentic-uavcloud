"""
Agentic UAV Cloud — ドローン測量データ診断PoC
LangGraph + Gemini (Vertex AI) によるファイル診断・処理ルート提案CLI
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# .env をプロジェクトルートから自動ロード
load_dotenv()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FileInventory(BaseModel):
    images: list[str] = Field(default_factory=list)
    drone_obs: list[str] = Field(default_factory=list)
    timestamp: list[str] = Field(default_factory=list)
    base_obs: list[str] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)


class RouteCapability(BaseModel):
    route_a: bool = False  # 簡易写真処理
    route_b: bool = False  # 高精度基線解析
    route_c: bool = False  # フルPPK写真処理
    route_d: bool = False  # 精度検証付き処理


class GraphState(BaseModel):
    target_dir: str = ""
    inventory: FileInventory = Field(default_factory=FileInventory)
    capability: RouteCapability = Field(default_factory=RouteCapability)
    recommendation: str = ""
    selected_route: str = ""
    execution_result: str = ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def scan_files(state: GraphState) -> dict:
    """ディレクトリを走査し、ファイルをカテゴリ別に分類する。"""
    target = Path(state.target_dir)
    if not target.is_dir():
        print(f"\n[ERROR] ディレクトリが見つかりません: {target}")
        sys.exit(1)

    inv = FileInventory()

    # Images: photos/ フォルダ内のファイル
    photos_dir = target / "photos"
    if photos_dir.is_dir():
        inv.images = sorted(
            f.name for f in photos_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )

    # Drone OBS: ルートの .obs ファイル
    inv.drone_obs = sorted(
        f.name for f in target.iterdir()
        if f.is_file() and f.suffix.lower() == ".obs"
    )

    # Timestamp: ルートの .MRK ファイル
    inv.timestamp = sorted(
        f.name for f in target.iterdir()
        if f.is_file() and f.suffix.upper() == ".MRK"
    )

    # Base OBS: base_station_logs/ 内の .obs ファイル
    base_dir = target / "base_station_logs"
    if base_dir.is_dir():
        inv.base_obs = sorted(
            f.name for f in base_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".obs"
        )

    # Markers: aerobo_marker_logs/ 内の .log ファイル
    marker_dir = target / "aerobo_marker_logs"
    if marker_dir.is_dir():
        inv.markers = sorted(
            f.name for f in marker_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".log"
        )

    print("\n========== ファイルスキャン結果 ==========")
    print(f"  Images       : {len(inv.images)} 枚")
    print(f"  Drone OBS    : {len(inv.drone_obs)} 件")
    print(f"  Timestamp    : {len(inv.timestamp)} 件")
    print(f"  Base OBS     : {len(inv.base_obs)} 件")
    print(f"  Markers      : {len(inv.markers)} 件")
    print("==========================================")

    return {"inventory": inv}


def analyze_capability(state: GraphState) -> dict:
    """ファイル構成から各ルートの実行可否を判定する。"""
    inv = state.inventory
    has_images = len(inv.images) > 0
    has_drone_obs = len(inv.drone_obs) > 0
    has_base_obs = len(inv.base_obs) > 0
    has_timestamp = len(inv.timestamp) > 0
    has_markers = len(inv.markers) > 0

    cap = RouteCapability(
        route_a=has_images,
        route_b=has_drone_obs and has_base_obs,
        route_c=has_images and has_drone_obs and has_base_obs and has_timestamp,
        route_d=has_images and has_drone_obs and has_base_obs and has_timestamp and has_markers,
    )

    print("\n========== ルート判定結果 ==========")
    labels = {
        "route_a": "A: 簡易写真処理",
        "route_b": "B: 高精度基線解析",
        "route_c": "C: フルPPK写真処理",
        "route_d": "D: 精度検証付き処理",
    }
    for key, label in labels.items():
        status = "✓ 実行可能" if getattr(cap, key) else "✗ 条件未達"
        print(f"  [{status}] {label}")
    print("====================================")

    return {"capability": cap}


def recommend_agent(state: GraphState) -> dict:
    """Gemini APIを使い、現在の状態に基づいた提案を日本語で生成する。"""
    inv = state.inventory
    cap = state.capability

    # Gemini APIへのプロンプト構築
    prompt = _build_recommendation_prompt(inv, cap)

    recommendation = _call_gemini(prompt)
    print(f"\n========== AI アドバイザー ==========\n{recommendation}")
    print("====================================")

    return {"recommendation": recommendation}


def _build_recommendation_prompt(inv: FileInventory, cap: RouteCapability) -> str:
    """Geminiに送る診断プロンプトを組み立てる。"""

    file_status = f"""
## アップロード済みファイル状況
- 撮影画像 (photos/): {len(inv.images)}枚 {'✓' if inv.images else '✗ 未検出'}
- ドローンOBS観測データ (.obs): {len(inv.drone_obs)}件 {'✓' if inv.drone_obs else '✗ 未検出'}
- タイムスタンプ (.MRK): {len(inv.timestamp)}件 {'✓' if inv.timestamp else '✗ 未検出'}
- 基準局OBSデータ (base_station_logs/*.obs): {len(inv.base_obs)}件 {'✓' if inv.base_obs else '✗ 未検出'}
- 検証マーカーログ (aerobo_marker_logs/*.log): {len(inv.markers)}件 {'✓' if inv.markers else '✗ 未検出'}
"""

    route_status = f"""
## 各処理ルートの実行可否
- ルートA（簡易写真処理）: {'実行可能' if cap.route_a else '実行不可'} — 条件: 撮影画像
- ルートB（高精度基線解析）: {'実行可能' if cap.route_b else '実行不可'} — 条件: ドローンOBS + 基準局OBS
- ルートC（フルPPK写真処理）: {'実行可能' if cap.route_c else '実行不可'} — 条件: 撮影画像 + ドローンOBS + 基準局OBS + タイムスタンプ
- ルートD（精度検証付き処理）: {'実行可能' if cap.route_d else '実行不可'} — 条件: ルートCの全条件 + 検証マーカーログ
"""

    return f"""あなたはドローン測量の専門アドバイザーです。
ユーザーがアップロードしたファイル構成を分析し、最適な処理方法を提案してください。

{file_status}
{route_status}

## 回答ルール
1. まず、現在実行可能な最上位のルートを明確に伝えてください。
2. もし最上位ルート（ルートD）が実行できない場合、上位ルートに進むために「具体的に何のファイルが足りないか」をファイル種別・格納先フォルダ名とともに明示してください。
3. 各実行可能ルートの概要を簡潔に説明してください。
4. 回答は自然な日本語で、箇条書きを活用し、簡潔にまとめてください。
5. 推奨するルートを1つ選び、その理由も述べてください。
"""


def _call_gemini(prompt: str) -> str:
    """Vertex AI 経由で Gemini を呼び出す。認証未設定時はフォールバック。"""

    # Vertex AI 設定（.env から読み込み）
    project = os.environ.get("VERTEX_PROJECT")
    location = os.environ.get("VERTEX_LOCATION")
    model = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash")

    if not project or not location:
        print("\n[WARNING] .env に VERTEX_PROJECT / VERTEX_LOCATION が設定されていません。")
        print("  → .env.example を参考に .env を作成してください。")
        return _fallback_recommendation()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            project=project,
            location=location,
        )
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"\n[WARNING] Vertex AI 呼び出しに失敗しました: {e}")
        print("  → 以下を確認してください:")
        print(f"    1. gcloud auth application-default login を実行済みか")
        print(f"    2. プロジェクト '{project}' で Vertex AI API が有効か")
        print(f"    3. リージョン '{location}' が正しいか")
        return _fallback_recommendation()


def _fallback_recommendation() -> str:
    """Gemini が使えない場合のルールベースのフォールバック。"""
    return (
        "（Gemini API 未接続のためルールベースで回答）\n"
        "ファイルスキャン結果と上記のルート判定を参照し、\n"
        "実行可能な最上位ルートの利用を推奨します。"
    )


def wait_for_user(state: GraphState) -> dict:
    """ユーザーにCLIで実行ルートを選択させる。"""
    cap = state.capability
    available: list[str] = []
    menu: dict[str, str] = {}

    if cap.route_a:
        available.append("A")
        menu["A"] = "簡易写真処理"
    if cap.route_b:
        available.append("B")
        menu["B"] = "高精度基線解析"
    if cap.route_c:
        available.append("C")
        menu["C"] = "フルPPK写真処理"
    if cap.route_d:
        available.append("D")
        menu["D"] = "精度検証付き処理"

    if not available:
        print("\n実行可能なルートがありません。ファイルを追加してください。")
        sys.exit(0)

    print("\n========== ルート選択 ==========")
    for key in available:
        print(f"  [{key}] {menu[key]}")
    print("  [Q] 終了")
    print("================================")

    while True:
        choice = input("\n実行するルートを選択してください: ").strip().upper()
        if choice == "Q":
            print("終了します。")
            sys.exit(0)
        if choice in available:
            print(f"\n→ ルート{choice}（{menu[choice]}）を選択しました。")
            return {"selected_route": choice}
        print(f"無効な入力です。{', '.join(available)} または Q を入力してください。")


def execute_mock(state: GraphState) -> dict:
    """選択されたルートをモック実行する。"""
    route = state.selected_route
    inv = state.inventory

    print(f"\n========== ルート{route} モック実行 ==========")

    if route == "A":
        print(f"[1/3] {len(inv.images)} 枚の画像を読み込み中...")
        print("[2/3] SfM (Structure from Motion) 処理を実行中...")
        print("[3/3] オルソ画像・点群データを生成中...")
        result = f"簡易写真処理が完了しました。{len(inv.images)}枚の画像からオルソ画像を生成。"

    elif route == "B":
        print(f"[1/3] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[2/3] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print("[3/3] RTKLib による基線解析を実行中...")
        result = "高精度基線解析が完了しました。PPK座標を算出。"

    elif route == "C":
        print(f"[1/5] {len(inv.images)} 枚の画像を読み込み中...")
        print(f"[2/5] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[3/5] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print(f"[4/5] タイムスタンプ ({inv.timestamp[0]}) でPPK測位を実行中...")
        print("[5/5] PPK座標付きSfM処理を実行中...")
        result = f"フルPPK写真処理が完了しました。{len(inv.images)}枚をPPK補正済み座標で処理。"

    elif route == "D":
        print(f"[1/6] {len(inv.images)} 枚の画像を読み込み中...")
        print(f"[2/6] ドローンOBS ({inv.drone_obs[0]}) を読み込み中...")
        print(f"[3/6] 基準局OBS ({inv.base_obs[0]}) を読み込み中...")
        print(f"[4/6] タイムスタンプ ({inv.timestamp[0]}) でPPK測位を実行中...")
        print("[5/6] PPK座標付きSfM処理を実行中...")
        print(f"[6/6] {len(inv.markers)} 個のマーカーで精度検証を実行中...")
        result = (
            f"精度検証付き処理が完了しました。"
            f"{len(inv.images)}枚をPPK補正済みで処理し、"
            f"{len(inv.markers)}点のマーカーで精度検証。"
        )
    else:
        result = "不明なルートです。"

    print(f"\n[結果] {result}")
    print("=" * 44)

    return {"execution_result": result}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """LangGraphのワークフローを構築する。"""
    builder = StateGraph(GraphState)

    builder.add_node("scan_files", scan_files)
    builder.add_node("analyze_capability", analyze_capability)
    builder.add_node("recommend_agent", recommend_agent)
    builder.add_node("wait_for_user", wait_for_user)
    builder.add_node("execute_mock", execute_mock)

    builder.add_edge(START, "scan_files")
    builder.add_edge("scan_files", "analyze_capability")
    builder.add_edge("analyze_capability", "recommend_agent")
    builder.add_edge("recommend_agent", "wait_for_user")
    builder.add_edge("wait_for_user", "execute_mock")
    builder.add_edge("execute_mock", END)

    return builder


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("使い方: uv run main.py <ディレクトリパス>")
        print("例:     uv run main.py /path/to/drone_data")
        sys.exit(1)

    target_dir = sys.argv[1]
    print(f"\n🛩️  Agentic UAV Cloud — ドローンデータ診断")
    print(f"対象ディレクトリ: {target_dir}")

    graph = build_graph()
    app = graph.compile()

    initial_state = GraphState(target_dir=target_dir)
    app.invoke(initial_state)


if __name__ == "__main__":
    main()
