# Agentic UAV Cloud

A **CLI-based PoC** that diagnoses the upload status of drone survey data and suggests/executes optimal processing paths.
Built on an Agentic AI architecture combining [LangGraph](https://github.com/langchain-ai/langgraph) and [Gemini (Vertex AI)](https://cloud.google.com/vertex-ai).

---

## Overview

Scans a user-specified directory and, based on the file types found, Gemini guides you in English on **"what can be done now"** and **"what is missing to proceed to higher-level processing"**.
In interactive mode, the agent answers free-form questions. Gemini also autonomously analyzes file contents via **Tool Use (ReAct pattern)** to detect and report data quality issues.

```
Specified directory
  → File scan
  → Processing route determination (A–D)
  → Initial diagnosis & recommendations by Gemini
  → Interactive loop (questions / route execution / rescan)
      └─ File quality questions → Gemini autonomously calls tools to answer
```

---

## Key Features

### 1. File Scan & Route Determination
Automatically classifies files in a directory into 5 categories and determines which processing routes A–D are executable.

### 2. Interactive Mode (Chat Loop)
After the initial diagnosis, the following operations can be performed repeatedly:
- **Free questions** — Gemini answers based on drone survey expertise
- **Route execution** — Enter A / B / C / D to trigger a mock execution
- **Rescan** — After adding or modifying files, type `rescan` to re-evaluate
- **Exit** — Type `quit` to exit

### 3. Tool Use (ReAct Pattern)
When a question about file quality or consistency is detected, Gemini autonomously calls the following tools to inspect real data before responding:

| Tool | Function |
|---|---|
| `check_mrk_file` | Analyzes entry count, time range, coordinate range, and accuracy statistics of MRK files |
| `check_obs_file` | Parses OBS (RINEX) file headers, returning observation time, satellite systems, and file size |
| `validate_data_consistency` | Batch-checks image count vs. MRK count consistency, OBS observation duration sufficiency, and file size validity |

---

## Processing Routes

Executable processing routes are automatically determined based on the uploaded file structure.

| Route | Name | Required Files |
|---|---|---|
| **A** | Simple Photo Processing | Captured images (`photos/`) |
| **B** | High-Precision Baseline Analysis | Drone OBS (`.obs`) + Base station OBS (`base_station_logs/*.obs`) |
| **C** | Full PPK Photo Processing | Route B requirements + captured images + timestamps (`.MRK`) |
| **D** | Processing with Accuracy Verification | Route C requirements + verification marker logs (`aerobo_marker_logs/*.log`) |

### Expected Directory Structure (Full Configuration)

```
your_drone_data/
├── photos/                    # Captured images
├── base_station_logs/
│   └── *.obs                  # Base station OBS data
├── aerobo_marker_logs/
│   └── *.log                  # Verification marker logs
├── *.obs                      # Drone OBS observation data
└── *.MRK                      # Timestamps
```

---

## Tech Stack

| Role | Library / Service |
|---|---|
| Agent Workflow | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | Gemini 2.5 Flash (Google Vertex AI) |
| LLM Client | [langchain-google-genai](https://github.com/langchain-ai/langchain-google) |
| Tool Use | LangChain `@tool` + `bind_tools()` (ReAct loop) |
| Data Model | [Pydantic v2](https://docs.pydantic.dev/) |
| Environment Variables | [python-dotenv](https://github.com/theskumar/python-dotenv) |
| Package Manager | [uv](https://github.com/astral-sh/uv) |
| Language | Python 3.12+ |

---

## Architecture

### LangGraph Workflow

```
START
  └─► scan_files                    # Directory scan & file classification
        └─► analyze_capability      # Determine executable routes A–D
              └─► recommend_agent   # Initial diagnosis & recommendations by Gemini
                    └─► chat_loop   # Wait for user input
                          │ (conditional)
                ┌─────────┼─────────────┬──────────┐
                ↓         ↓             ↓          ↓
          chat_respond  execute_mock  rescan_notify  END
              │             │             │
              └─► chat_loop └─► END       └─► scan_files (loop)
```

### Project Structure

```
agentic-uavcloud/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project config & dependencies (uv)
├── .env                             # Environment variables (not in Git)
├── .env.example                     # Environment variable template
├── .gitignore
└── agent/
    ├── __init__.py                  # Exports build_graph
    ├── graph.py                     # LangGraph workflow definition
    ├── state.py                     # Pydantic state model
    ├── llm/
    │   ├── client.py                # Gemini invocation + ReAct loop
    │   └── prompts.py               # System prompt construction
    ├── nodes/
    │   ├── scan.py                  # File scan & route determination
    │   ├── recommend.py             # Initial diagnosis & recommendations
    │   ├── chat.py                  # Interactive loop & tool hint injection
    │   └── execute.py               # Route mock execution
    └── tools/
        ├── __init__.py              # Exports ALL_TOOLS
        └── file_analysis.py         # MRK/OBS analysis & consistency check tools
```

---

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) installed
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed
- A GCP project with the Vertex AI API enabled

### 1. Clone the Repository

```bash
git clone <repository-url>
cd agentic-uavcloud
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your GCP project information:

```env
VERTEX_PROJECT=your-gcp-project-id
VERTEX_LOCATION=us-central1
VERTEX_MODEL=gemini-2.5-flash
```

### 4. Authenticate with Vertex AI

```bash
gcloud auth application-default login
```

---

## Usage

```bash
uv run main.py <path-to-drone-data-directory>
```

**Examples:**

```bash
# Full dataset (all routes up to D are available)
uv run main.py /path/to/full_drone_data

# Photos-only dataset (only Route A is available)
uv run main.py /path/to/photos_only
```

### Interactive Mode Operations

```
========== Interactive Mode ==========
  Execute route: enter A, B, C, or D
  Ask a question: type any free-form text
  Rescan:         type rescan
  Exit:           type quit
======================================

> Any issues with the data?
  🔧 Tool call: validate_data_consistency({})
  🔧 Tool call: check_mrk_file({"filename": "20220607_0405_ASTimestamp.MRK"})
  🔧 Tool call: check_obs_file({"filename": "20220607_0405_ASRinexRtcm3.obs", "location": "root"})

========== AI Advisor ==========
Here are the results of the data quality check:
...
================================
```

---

## About Vertex AI Authentication

This project uses **Application Default Credentials (ADC)** rather than API keys.
Run `gcloud auth application-default login` once, and authentication will be handled automatically from that point on.

If authentication fails, check the following:

1. Have you run `gcloud auth application-default login`?
2. Is `VERTEX_PROJECT` in `.env` correct?
3. Is the Vertex AI API enabled for the specified project?
