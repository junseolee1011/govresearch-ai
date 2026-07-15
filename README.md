# GovResearch-AI

GovResearch-AI is an autonomous, multi-agent research workflow for identifying
and classifying AI services used by public institutions. It plans a research
task, gathers research insights via local Ollama LLM, and produces a structured
Markdown report. Sprint 2 integrates Ollama for real local LLM inference without
external APIs or cloud services.

## Features

- LangGraph `StateGraph` workflow: Planner → Research → Report
- Ollama integration for local LLM inference (qwen3:8b default, llama3.1:8b supported)
- Typed, shared workflow state
- File-based prompts with variable substitution
- Structured logging with execution timing and response metrics
- AI service inventory: institution, service type, use case, and maturity
- Professional Markdown report generation

## Requirements

- Python 3.11+
- Ollama (local LLM runtime)

## Installation

### Step 1: Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai).

### Step 2: Download a Model

```bash
# Default model (qwen3:8b)
ollama pull qwen3:8b

# Alternative: llama3.1:8b
ollama pull llama3.1:8b
```

Ensure Ollama is running (default: `http://localhost:11434`).

### Step 3: Set Up Python Environment

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### Step 4: Configure (Optional)

Edit `.env` to customize Ollama settings:

```dotenv
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.2
TOP_P=0.9
MAX_TOKENS=2048
```

## Usage

```bash
# Run with default topic
python main.py

# Run with custom topic
python main.py --topic "AI services used by municipal governments"
```

The generated Markdown report is printed to the terminal and saved in `reports/`.

## Architecture

```text
main.py
  └── app/workflow.py
        └── app/graph.py (LangGraph StateGraph with Ollama)
              ├── agents/planner.py (Ollama: generate research plan)
              ├── agents/researcher.py (Ollama: generate research findings)
              └── agents/report_writer.py (Ollama: generate report)

Shared state: app/state.py
Configuration: config/settings.py
Prompts: prompts/*.txt (with variable substitution)
Logging: tools/logger.py (with execution metrics)
Ollama client: tools/ollama_client.py
Reports: reports/
```

## Sprint 2 Implementation

### Ollama Integration

- **Local-only inference**: No cloud APIs, no external LLM services
- **Configurable model**: Default `qwen3:8b`, supports `llama3.1:8b`
- **Error handling**: Meaningful exceptions when Ollama is unreachable
- **Timeout and context window**: Configurable via `.env`

### Agent Updates

- **Planner**: Generates research objectives and keywords via Ollama
- **Researcher**: LLM expands planner output into structured research data
- **Report Writer**: Produces professional Markdown via Ollama

### Prompt System

- **Variable replacement**: Prompts support `{query}`, `{research_plan}`, `{documents}`
- **Improved prompts**: Structured guidance for consistent LLM output

### Logging

- **Execution timing**: Tracks start/end for each agent node
- **Response metrics**: Logs LLM response length
- **Error tracking**: Detailed error logging with context

## Testing

```bash
python -m unittest discover -s tests -v
```

## Troubleshooting

### Ollama server is not running

```bash
Ollama server is not running at http://localhost:11434.
Please start Ollama and ensure the model is available.
```

**Solution**: Start Ollama:
```bash
ollama serve
```

### Model not found

```bash
ollama pull qwen3:8b
```

### Slow inference

- Check system resources (CPU/memory)
- Consider smaller model or increase `MAX_TOKENS`
- Reduce `TEMPERATURE` for faster responses

## Next Steps (Future Sprints)

- Tavily API for live web search
- ChromaDB for semantic search and memory
- Streamlit UI
- Docker containerization
- Reflection agent for self-critique and refinement
