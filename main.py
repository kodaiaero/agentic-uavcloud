"""
Agentic UAV Cloud — ドローン測量データ診断PoC
LangGraph + Gemini (Vertex AI) による対話型ファイル診断・処理ルート提案CLI

ワークフロー:
  START → scan_files → analyze_capability → recommend_agent
            ↑                                      ↓
            │                                 chat_loop ──→ execute_mock → END
            │                               /     │
            └──────── (rescan) ────────────┘   (質問 → Gemini応答)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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


ROUTE_LABELS: dict[str, str] = {
    "A": "簡易写真処理",
    "B": "高精度基線解析",
    "C": "フルPPK写真処理",
    "D": "精度検証付き処理",
}


class GraphState(BaseModel):
    target_dir: str = ""
    inventory: FileInventory = Field(default_factory=FileInventory)
    capability: RouteCapability = Field(default_factory=RouteCapability)
    recommendation: str = ""
    # チャットループ用
    chat_history: list = Field(default_factory=list)
    user_input: str = ""
    next_action: str = ""  # "chat" | "rescan" | "execute" | "quit"
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
    cap_keys = {
        "route_a": "A: 簡易写真処理",
        "route_b": "B: 高精度基線解析",
        "route_c": "C: フルPPK写真処理",
        "route_d": "D: 精度検証付き処理",
    }
    for key, label in cap_keys.items():
        status = "✓ 実行可能" if getattr(cap, key) else "✗ 条件未達"
        print(f"  [{status}] {label}")
    print("====================================")

    return {"capability": cap}


def recommend_agent(state: GraphState) -> dict:
    """Gemini で初回の診断・提案を生成し、チャット履歴を初期化する。"""
    inv = state.inventory
    cap = state.capability

    system_prompt = _build_system_prompt(inv, cap)
    initial_user_msg = "現在のファイル構成を診断して、最適な処理ルートを提案してください。"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=initial_user_msg),
    ]

    response_text = _call_gemini_chat(messages)

    messages.append(AIMessage(content=response_text))

    print(f"\n========== AI アドバイザー ==========\n{response_text}")
    print("====================================")

    return {
        "recommendation": response_text,
        "chat_history": messages,
    }


def chat_loop(state: GraphState) -> dict:
    """ユーザー入力を受け取り、次のアクションを判定する。"""
    cap = state.capability
    available = _get_available_routes(cap)

    # ヘルプ表示
    print("\n========== 対話モード ==========")
    if available:
        print(f"  ルート実行: {', '.join(available)} を入力")
    print("  質問:       自由にテキストを入力")
    print("  再スキャン: rescan と入力")
    print("  終了:       quit と入力")
    print("================================")

    user_input = input("\n> ").strip()

    # 入力の分類
    upper = user_input.upper()
    if upper in ("QUIT", "Q", "EXIT"):
        return {"user_input": user_input, "next_action": "quit"}

    if upper in ("RESCAN", "RS", "再スキャン"):
        return {"user_input": user_input, "next_action": "rescan"}

    if upper in available:
        print(f"\n→ ルート{upper}（{ROUTE_LABELS[upper]}）を選択しました。")
        return {
            "user_input": user_input,
            "next_action": "execute",
            "selected_route": upper,
        }

    # それ以外は自由質問 → Gemini に送る
    return {"user_input": user_input, "next_action": "chat"}


def chat_respond(state: GraphState) -> dict:
    """ユーザーの自由質問に Gemini で回答する。"""
    history = list(state.chat_history)
    history.append(HumanMessage(content=state.user_input))

    response_text = _call_gemini_chat(history)
    history.append(AIMessage(content=response_text))

    print(f"\n========== AI アドバイザー ==========\n{response_text}")
    print("====================================")

    return {"chat_history": history}


def rescan_notify(state: GraphState) -> dict:
    """再スキャン前にユーザーに通知する。"""
    print("\n🔄 ディレクトリを再スキャンします...")
    # チャット履歴をリセット（新しいスキャン結果でコンテキストが変わるため）
    return {"chat_history": []}


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
# Router (conditional edge)
# ---------------------------------------------------------------------------

def route_after_chat(state: GraphState) -> str:
    """chat_loop の next_action に基づいてグラフの分岐先を決定する。"""
    action = state.next_action

    if action == "execute":
        return "execute_mock"
    elif action == "rescan":
        return "rescan_notify"
    elif action == "quit":
        return END
    else:  # "chat"
        return "chat_respond"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_available_routes(cap: RouteCapability) -> list[str]:
    """実行可能なルートのリストを返す。"""
    routes = []
    if cap.route_a:
        routes.append("A")
    if cap.route_b:
        routes.append("B")
    if cap.route_c:
        routes.append("C")
    if cap.route_d:
        routes.append("D")
    return routes


def _build_system_prompt(inv: FileInventory, cap: RouteCapability) -> str:
    """Gemini に渡すシステムプロンプトを構築する。"""

    file_status = f"""\
## アップロード済みファイル状況
- 撮影画像 (photos/): {len(inv.images)}枚 {'✓' if inv.images else '✗ 未検出'}
- ドローンOBS観測データ (.obs): {len(inv.drone_obs)}件 {'✓' if inv.drone_obs else '✗ 未検出'}
- タイムスタンプ (.MRK): {len(inv.timestamp)}件 {'✓' if inv.timestamp else '✗ 未検出'}
- 基準局OBSデータ (base_station_logs/*.obs): {len(inv.base_obs)}件 {'✓' if inv.base_obs else '✗ 未検出'}
- 検証マーカーログ (aerobo_marker_logs/*.log): {len(inv.markers)}件 {'✓' if inv.markers else '✗ 未検出'}"""

    route_status = f"""\
## 各処理ルートの実行可否
- ルートA（簡易写真処理）: {'実行可能' if cap.route_a else '実行不可'} — 条件: 撮影画像
- ルートB（高精度基線解析）: {'実行可能' if cap.route_b else '実行不可'} — 条件: ドローンOBS + 基準局OBS
- ルートC（フルPPK写真処理）: {'実行可能' if cap.route_c else '実行不可'} — 条件: 撮影画像 + ドローンOBS + 基準局OBS + タイムスタンプ
- ルートD（精度検証付き処理）: {'実行可能' if cap.route_d else '実行不可'} — 条件: ルートCの全条件 + 検証マーカーログ"""

    return f"""\
あなたはドローン測量の専門アドバイザーです。
ユーザーが測量データをクラウド処理するためにファイルをアップロードしました。
以下のファイル構成とルート判定結果を把握した上で、ユーザーの質問に的確に回答してください。

{file_status}

{route_status}

## ドメイン知識
- .obs ファイル: GNSS (衛星測位) の生観測データ。RINEX形式。PPK処理に必要。
- .MRK ファイル: ドローンのカメラシャッターを切った瞬間のGNSSタイムスタンプ。各画像の正確な撮影時刻と位置を紐づけるために使用。
- 基準局OBS: 既知座標の地上基準局で同時に記録したGNSS観測データ。ドローンOBSとペアで基線解析に使用。
- マーカーログ: 地上に設置した検証用ターゲット（GCP/検証点）の座標ログ。処理結果の精度を評価するために使用。
- SfM (Structure from Motion): 多数の写真からカメラ位置を推定し、3D点群やオルソ画像を生成する技術。
- PPK (Post-Processed Kinematic): 飛行後にドローンOBSと基準局OBSを基線解析し、センチメートル級の測位精度を実現する手法。

## 回答ルール
1. 質問に対して簡潔・正確に日本語で回答してください。
2. 上位ルートに進むために「何が足りないか」を聞かれた場合、ファイル種別・格納先フォルダ名・取得方法を具体的に説明してください。
3. ファイルの用途を聞かれた場合、ドメイン知識に基づいて実務的に分かりやすく説明してください。
4. 箇条書きを活用し、長くなりすぎないようにまとめてください。
5. 推奨ルートを提案する際は、理由も簡潔に述べてください。"""


def _call_gemini_chat(messages: list) -> str:
    """Vertex AI 経由で Gemini をチャット形式で呼び出す。"""

    # Vertex AI 設定（.env から読み込み）
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
    """Gemini が使えない場合のフォールバック。"""
    return (
        "（Gemini API 未接続のためルールベースで回答）\n"
        "ファイルスキャン結果とルート判定を参照し、\n"
        "実行可能な最上位ルートの利用を推奨します。"
    )


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """LangGraphのワークフローを構築する。

    START → scan_files → analyze_capability → recommend_agent → chat_loop
              ↑                                                 ↓  (conditional)
              │                                          ┌──────┼──────────┐
              │                                          ↓      ↓          ↓
              └──── rescan_notify ←─────────    chat_respond  execute_mock  END
                                                     │                │
                                                     └→ chat_loop ←──┘(no, → END)
    """
    builder = StateGraph(GraphState)

    # ノード登録
    builder.add_node("scan_files", scan_files)
    builder.add_node("analyze_capability", analyze_capability)
    builder.add_node("recommend_agent", recommend_agent)
    builder.add_node("chat_loop", chat_loop)
    builder.add_node("chat_respond", chat_respond)
    builder.add_node("rescan_notify", rescan_notify)
    builder.add_node("execute_mock", execute_mock)

    # 直線エッジ（初期フロー）
    builder.add_edge(START, "scan_files")
    builder.add_edge("scan_files", "analyze_capability")
    builder.add_edge("analyze_capability", "recommend_agent")
    builder.add_edge("recommend_agent", "chat_loop")

    # 条件分岐: chat_loop → 4方向
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

    # ループバック: chat_respond → chat_loop
    builder.add_edge("chat_respond", "chat_loop")

    # ループバック: rescan_notify → scan_files（再スキャン）
    builder.add_edge("rescan_notify", "scan_files")

    # 実行後 → 終了
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
    print(f"\n🛩️  Agentic UAV Cloud — ドローンデータ診断（対話モード）")
    print(f"対象ディレクトリ: {target_dir}")

    graph = build_graph()
    app = graph.compile()

    initial_state = GraphState(target_dir=target_dir)
    app.invoke(initial_state)


if __name__ == "__main__":
    main()
