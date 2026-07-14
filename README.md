# GovResearch-AI

GovResearch-AI is an autonomous, multi-agent research workflow for public-sector
AI topics. It plans a research task, gathers initial evidence, analyzes the
material, and produces a structured Markdown report. Sprint 1 intentionally
uses deterministic local agents and a dummy research source so the foundation
can run without external AI or search services.

## Features

- LangGraph `StateGraph` workflow: Planner → Research → Report
- Typed, shared workflow state
- File-based prompts and environment-based configuration
- Structured logging and a Markdown report artifact
- Dummy research agent designed to be replaced by live search in a later sprint

## Requirements

- Python 3.11+

## Installation

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python main.py
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` and
copy the environment template with `cp .env.example .env`.

## Usage

```bash
python main.py
python main.py --topic "AI platform implementation in local government"
```

The generated Markdown report is printed to the terminal and saved in
`reports/`.

## Architecture

```text
main.py
  └── app/workflow.py
        └── app/graph.py (LangGraph StateGraph)
              ├── agents/planner.py
              ├── agents/researcher.py
              └── agents/report_writer.py

Shared state: app/state.py
Configuration: config/settings.py
Prompts: prompts/*.txt
Logging: tools/logger.py
Reports: reports/
```

## Sprint 1 scope

This release establishes the orchestration foundation only. Ollama, Tavily,
ChromaDB, Streamlit, Docker, and a Reflection Agent are deliberately deferred
to future sprints.

## Testing

```bash
python -m unittest discover -s tests -v
```
