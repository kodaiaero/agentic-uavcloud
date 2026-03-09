# Agentic UAV Cloud

ドローン測量データのアップロード状況を診断し、最適な処理パスを提案・実行する **CLIベースのPoC** です。
[LangGraph](https://github.com/langchain-ai/langgraph) と [Gemini (Vertex AI)](https://cloud.google.com/vertex-ai) を組み合わせたAgentic AIアーキテクチャで構成されています。

---

## 概要

ユーザーが指定したディレクトリをスキャンし、含まれるファイル種別に応じて **「今何ができるか」** と **「上位処理に進むために何が足りないか」** を Gemini が日本語でガイドします。
対話モードではユーザーの自由な質問に回答するほか、Gemini が **Tool Use（ReActパターン）** でファイルの中身を自律的に解析し、データ品質の問題を検出・報告します。

```
指定ディレクトリ
  → ファイルスキャン
  → 処理ルート判定（A〜D）
  → Gemini による初回診断・提案
  → 対話ループ（質問 / ルート実行 / 再スキャン）
      └─ ファイル品質の質問 → Gemini がツールを自律呼び出しして回答
```

---

## 主な機能

### 1. ファイルスキャン＆ルート判定
ディレクトリ内のファイルを5カテゴリに自動分類し、処理ルートA〜Dの実行可否を判定します。

### 2. 対話モード（チャットループ）
初回診断後、以下の操作を繰り返し実行できます：
- **自由質問** — Gemini がドローン測量の専門知識に基づいて回答
- **ルート実行** — A / B / C / D を入力してモック実行
- **再スキャン** — ファイルを追加・修正した後に `rescan` で再判定
- **終了** — `quit` で終了

### 3. Tool Use（ReActパターン）
ファイルの品質・整合性に関する質問を検出すると、Gemini が以下のツールを自律的に呼び出して実データを確認してから回答します：

| ツール | 機能 |
|---|---|
| `check_mrk_file` | MRKファイルのエントリ数・時間範囲・座標範囲・精度統計を解析 |
| `check_obs_file` | OBS（RINEX）ファイルのヘッダーを解析し、観測時間・衛星システム・ファイルサイズを返す |
| `validate_data_consistency` | 画像枚数とMRK数の整合性、OBS観測時間の十分性、ファイルサイズの妥当性を一括チェック |

---

## 処理ルート

アップロードされたファイル構成に応じて、実行可能な処理ルートが自動判定されます。

| ルート | 名称 | 必要なファイル |
|---|---|---|
| **A** | 簡易写真処理 | 撮影画像 (`photos/`) |
| **B** | 高精度基線解析 | Drone OBS (`.obs`) + 基準局 OBS (`base_station_logs/*.obs`) |
| **C** | フルPPK写真処理 | ルートBの条件 + 撮影画像 + タイムスタンプ (`.MRK`) |
| **D** | 精度検証付き処理 | ルートCの条件 + 検証マーカーログ (`aerobo_marker_logs/*.log`) |

### 期待するディレクトリ構成（フル構成）

```
your_drone_data/
├── photos/                    # 撮影画像
├── base_station_logs/
│   └── *.obs                  # 基準局OBSデータ
├── aerobo_marker_logs/
│   └── *.log                  # 検証マーカーログ
├── *.obs                      # ドローンOBS観測データ
└── *.MRK                      # タイムスタンプ
```

---

## 技術スタック

| 役割 | ライブラリ / サービス |
|---|---|
| Agentワークフロー | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Gemini 2.0 Flash (Google Vertex AI) |
| LLM クライアント | [langchain-google-genai](https://github.com/langchain-ai/langchain-google) |
| Tool Use | LangChain `@tool` + `bind_tools()` (ReActループ) |
| データモデル | [Pydantic v2](https://docs.pydantic.dev/) |
| 環境変数管理 | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| パッケージ管理 | [uv](https://github.com/astral-sh/uv) |
| 言語 | Python 3.12+ |

---

## アーキテクチャ

### LangGraphワークフロー

```
START
  └─► scan_files                    # ディレクトリ走査・ファイル分類
        └─► analyze_capability      # ルートA〜D の実行可否判定
              └─► recommend_agent   # Gemini による初回診断・提案
                    └─► chat_loop   # ユーザー入力を待機
                          │ (conditional)
                ┌─────────┼─────────────┬──────────┐
                ↓         ↓             ↓          ↓
          chat_respond  execute_mock  rescan_notify  END
              │             │             │
              └─► chat_loop └─► END       └─► scan_files (ループ)
```

### プロジェクト構成

```
agentic-uavcloud/
├── main.py                          # CLIエントリーポイント
├── pyproject.toml                   # プロジェクト設定・依存関係 (uv)
├── .env                             # 環境変数（Gitに含めない）
├── .env.example                     # 環境変数のテンプレート
├── .gitignore
└── agent/
    ├── __init__.py                  # build_graph をエクスポート
    ├── graph.py                     # LangGraphワークフロー定義
    ├── state.py                     # Pydantic状態モデル
    ├── llm/
    │   ├── client.py                # Gemini呼び出し + ReActループ
    │   └── prompts.py               # システムプロンプト構築
    ├── nodes/
    │   ├── scan.py                  # ファイルスキャン・ルート判定
    │   ├── recommend.py             # 初回診断・提案
    │   ├── chat.py                  # 対話ループ・ツールヒント注入
    │   └── execute.py               # ルートモック実行
    └── tools/
        ├── __init__.py              # ALL_TOOLS をエクスポート
        └── file_analysis.py         # MRK/OBS解析・整合性チェックツール
```

---

## セットアップ

### 前提条件

- Python 3.12 以上
- [uv](https://github.com/astral-sh/uv) インストール済み
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) インストール済み
- Vertex AI API が有効なGCPプロジェクト

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd agentic-uavcloud
```

### 2. 依存関係のインストール

```bash
uv sync
```

### 3. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を開き、GCPプロジェクト情報を入力します：

```env
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.0-flash
```

### 4. Vertex AI 認証

```bash
gcloud auth application-default login
```

---

## 実行方法

```bash
uv run main.py <ドローンデータのディレクトリパス>
```

**例：**

```bash
# 全ファイルが揃ったデータ（ルートD まで実行可能）
uv run main.py /path/to/full_drone_data

# 撮影画像のみのデータ（ルートA のみ実行可能）
uv run main.py /path/to/photos_only
```

### 対話モードの操作

```
========== 対話モード ==========
  ルート実行: A, B, C, D を入力
  質問:       自由にテキストを入力
  再スキャン: rescan と入力
  終了:       quit と入力
================================

> データに問題ない？
  🔧 ツール実行: validate_data_consistency({})
  🔧 ツール実行: check_mrk_file({"filename": "20220607_0405_ASTimestamp.MRK"})
  🔧 ツール実行: check_obs_file({"filename": "20220607_0405_ASRinexRtcm3.obs", "location": "root"})

========== AI アドバイザー ==========
データの品質チェック結果をお伝えします：
...
====================================
```

---

## Vertex AI 認証について

このプロジェクトは API キーではなく **Application Default Credentials (ADC)** を使用します。
`gcloud auth application-default login` を一度実行しておくだけで、以降は自動的に認証が通ります。

認証に失敗した場合は以下を確認してください：

1. `gcloud auth application-default login` を実行済みか
2. `.env` の `VERTEX_PROJECT` が正しいか
3. 該当プロジェクトで Vertex AI API が有効になっているか
