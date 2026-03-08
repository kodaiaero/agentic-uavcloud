# Agentic UAV Cloud

ドローン測量データのアップロード状況を診断し、最適な処理パスを提案・実行する **CLIベースのPoC** です。
[LangGraph](https://github.com/langchain-ai/langgraph) と [Gemini (Vertex AI)](https://cloud.google.com/vertex-ai) を組み合わせたAgentic AIアーキテクチャで構成されています。

---

## 概要

ユーザーが指定したディレクトリをスキャンし、含まれるファイル種別に応じて **「今何ができるか」** と **「上位処理に進むために何が足りないか」** を Gemini が日本語でガイドします。

```
指定ディレクトリ
  → ファイルスキャン
  → 処理ルート判定（A〜D）
  → Gemini による自然言語アドバイス
  → ユーザーがルートを選択
  → モック実行
```

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
| データモデル | [Pydantic v2](https://docs.pydantic.dev/) |
| 環境変数管理 | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| パッケージ管理 | [uv](https://github.com/astral-sh/uv) |
| 言語 | Python 3.12+ |

### LangGraphワークフロー

```
START
  └─► scan_files          # ディレクトリ走査・ファイル分類
        └─► analyze_capability   # ルートA〜D の実行可否判定
              └─► recommend_agent     # Gemini による日本語アドバイス生成
                    └─► wait_for_user      # CLIでルート選択
                          └─► execute_mock     # 選択ルートのモック実行
                                └─► END
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

### 実行例（写真のみの場合）

```
🛩️  Agentic UAV Cloud — ドローンデータ診断
対象ディレクトリ: /path/to/photos_only

========== ファイルスキャン結果 ==========
  Images       : 3 枚
  Drone OBS    : 0 件
  Timestamp    : 0 件
  Base OBS     : 0 件
  Markers      : 0 件
==========================================

========== ルート判定結果 ==========
  [✓ 実行可能] A: 簡易写真処理
  [✗ 条件未達] B: 高精度基線解析
  [✗ 条件未達] C: フルPPK写真処理
  [✗ 条件未達] D: 精度検証付き処理
====================================

========== AI アドバイザー ==========
現在実行可能な最上位ルートは ルートA（簡易写真処理）です。

上位ルートに進むために不足しているファイル：
- ドローンOBS観測データ (.obs) をルート直下に配置してください
- 基準局OBSデータ (base_station_logs/*.obs) を base_station_logs/ に配置してください
- タイムスタンプ (.MRK) をルート直下に配置してください
- 検証マーカーログ (aerobo_marker_logs/*.log) を aerobo_marker_logs/ に配置してください
...
====================================

========== ルート選択 ==========
  [A] 簡易写真処理
  [Q] 終了
================================
```

---

## ファイル構成

```
agentic-uavcloud/
├── main.py          # 全ロジック（LangGraphワークフロー）
├── pyproject.toml   # プロジェクト設定・依存関係 (uv)
├── .env             # 環境変数（Gitに含めない）
├── .env.example     # 環境変数のテンプレート
└── .gitignore
```

---

## Vertex AI 認証について

このプロジェクトは API キーではなく **Application Default Credentials (ADC)** を使用します。
`gcloud auth application-default login` を一度実行しておくだけで、以降は自動的に認証が通ります。

認証に失敗した場合は以下を確認してください：

1. `gcloud auth application-default login` を実行済みか
2. `.env` の `VERTEX_PROJECT` が正しいか
3. 該当プロジェクトで Vertex AI API が有効になっているか
