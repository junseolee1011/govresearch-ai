# GovResearch-AI

GovResearch-AI is an autonomous, multi-agent research workflow for identifying
and classifying AI services used by public institutions. It plans a research
task, gathers initial evidence, classifies services by institution, type, use
case, and maturity, and produces a structured Markdown report. Sprint 1 uses
deterministic local agents and dummy research data, so it runs without external
AI or search services.

## Features

- LangGraph `StateGraph` workflow: Planner → Research → Report
- Typed, shared workflow state
- File-based prompts and environment-based configuration
- Structured logging and a Markdown report artifact
- AI service inventory: institution, service type, use case, and maturity
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
python main.py --topic "AI services used by municipal governments"
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
