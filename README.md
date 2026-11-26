# GovResearch-AI (PoC with MCP-style tools)

GovResearch-AI is a research agent that investigates how Korean public institutions use AI services and turns the findings into a structured Markdown report. It runs a small LangGraph workflow (Planner → Researcher → Report Writer), gathers evidence from a local PDF case book and from web search through an MCP-style tool layer, and extracts service records (institution, service, role, category) with source references.

## Features

- **Linear LangGraph workflow**: Planner → Researcher → Report Writer → END
- **MCP-style tool layer**: an MCPClient registers tool servers (Filesystem, Tavily) and dispatches tool calls. The servers run in-process; they follow the MCP tool-call pattern but do not use the MCP wire protocol yet.
- **Local LLM planning**: the Planner asks an Ollama model (qwen2.5:7b by default) for a schema-validated research plan
- **Local document reading**: PDF, Markdown and text files in data are read and chunked (1,500 characters, 200 overlap)
- **Web search**: Tavily search with URL deduplication
- **Rule-based extraction**: services are extracted from local chunks and web results with organization whitelists and patterns, classified into an L1/L2 taxonomy, and merged
- **Source references**: each service keeps its source chunk IDs (and URLs for web results); the report lists local and web references
- **Streamlit UI** and a command-line entry point

## Workflow

```
User Query
    ↓
Planner (Ollama: structured plan → search queries)
    ↓
Researcher (MCP Client)
    ├─→ Filesystem MCP  (topic has report/case/status/policy-style keywords)
    └─→ Tavily MCP      (always enabled)
    ↓
Report Writer (extract → classify → merge → Markdown template)
    ↓
END
```

### Planner (agents/planner.py)

- Requests a plan validated against a Pydantic schema (models/schemas.py): a main goal and at least three sub-tasks.
- Retries up to three times if the plan is too shallow. If every attempt fails (for example Ollama is unreachable), the topic itself becomes the only search query.
- For Korean public-sector AI topics, search queries are the plan's main goal followed by the predefined service-specific queries in config/service_constants.py, capped at MAX_SEARCH_QUERIES (default 70, which covers all of them). Other topics use the main goal and the plan's sub-tasks.

### Researcher (agents/researcher.py)

- Filesystem MCP is used when the topic contains keywords such as 보고서, 사례, 현황, 정책, report, case, status or policy.
- Tavily MCP web search is always used.
- Local files in data/ are read with read_pdf, read_markdown or read_text_file depending on the extension, then chunked.

### Report Writer (agents/report_writer.py, agents/information_extractor.py)

- Extracts services from all local chunks and, separately, from web results (deduplicated against local services).
- Classifies each service into an L1/L2 taxonomy by keyword mapping; unmatched services are marked 보류.
- Merges near-duplicate services and assembles the report from a deterministic Markdown template.
- Falls back to simpler rule-based extraction if Ollama cannot be initialized or extraction fails.

## MCP-style Structure

```python
mcp_client = MCPClient()
mcp_client.register_server("filesystem", filesystem_server)
mcp_client.register_server("tavily", tavily_server)
result = mcp_client.call_tool("tavily", "search", query="...", max_results=5)
```

- **Filesystem server** (mcp/servers/filesystem.py): list_files, read_text_file, read_markdown, read_pdf
- **Tavily server** (mcp/servers/tavily.py): search

## Requirements

- Python 3.11+
- Ollama (local LLM runtime)
- Tavily API key (free tier available at https://tavily.com)

## Installation

### Step 1: Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai).

### Step 2: Download a Model

```bash
ollama pull qwen2.5:7b
```

Ensure Ollama is running (default: http://localhost:11434).

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

### Step 4: Configure .env

```dotenv
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.2
TOP_P=0.9
MAX_TOKENS=2048
TAVILY_API_KEY=tvly-xxxxxxxx
SEARCH_MAX_RESULTS=5
SEARCH_TOPIC=general
MAX_SEARCH_QUERIES=70
```

MAX_SEARCH_QUERIES bounds how many search queries a Korean public-sector AI topic produces. Every query is one Tavily call, so lower it to shorten runs or reduce API usage.

## Usage

### Command Line

```bash
# Run with default topic
python main.py

# Run with custom topic
python main.py --topic "한국 공공기관 AI 서비스 현황"
```

### Streamlit UI

```bash
streamlit run ui.py
```

Reports are also saved as Markdown files in the reports/ directory.

## Report Structure

The generated Markdown report contains:

- **Executive Summary**: number of identified AI services
- **핵심 내용**: services grouped by operating organization, with role and L1 > L2 category
- **주요 사례**: numbered case list with organization, role, category and source chunk IDs
- **시사점**: short takeaways derived from the counts
- **References**: the local PDF and the web URLs used

## Testing

The unit tests are offline (no Ollama or Tavily key needed):

```bash
python -m unittest discover -s tests -v
```

They cover chunking, tool selection, query generation, the researcher with a fake MCP client, and configuration defaults.

## Troubleshooting

### Tavily API key not configured

```bash
TavilySearchError: Tavily API key is not configured.
```

**Solution**: Add your Tavily API key to .env:
```dotenv
TAVILY_API_KEY=tvly-xxxxxxxx
```

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
ollama pull qwen2.5:7b
```

## Project Structure

```
govresearch-ai/
├── agents/
│   ├── planner.py               # Structured plan and search queries
│   ├── researcher.py            # Tool selection, local PDF chunking, web search
│   ├── report_writer.py         # Extraction, merging and report assembly
│   ├── information_extractor.py # Rule-based extraction and L1/L2 classification
│   └── retriever.py, validator.py, reflection.py,
│       fact_extractor.py, summarizer.py   # Earlier-iteration modules, not wired into the graph
├── app/
│   ├── graph.py                 # LangGraph workflow
│   ├── state.py                 # Shared state definition
│   └── workflow.py              # Workflow run and report persistence
├── config/
│   ├── settings.py              # Environment-based settings
│   └── service_constants.py     # Target organizations and predefined search queries
├── data/                        # Local reference documents (PDF case book)
├── mcp/
│   ├── client.py                # MCP-style client
│   └── servers/                 # Filesystem and Tavily tool servers
├── models/                      # Pydantic schemas and document model
├── prompts/                     # Prompt templates
├── tests/                       # Offline unit tests
├── tools/                       # Logger, Ollama client, Tavily search, and helpers
│                                # (vector_store, embeddings, deduplicator, source_ranker, ... not wired into the graph)
├── main.py                      # CLI entry point
├── ui.py                        # Streamlit UI
└── requirements.txt
```

## Known Limitations and Future Work

- The MCP layer is in-process; moving the servers to the real MCP transport is future work.
- The graph is linear. Retrieval (ChromaDB), fact validation, reflection and source ranking modules exist but are not connected yet.
- Extraction and classification are rule-based, so coverage depends on the organization whitelist and keyword mappings.
- The predefined queries and target organizations come from one reference case book, which limits generality.
- The workflow does not record which path (normal or fallback) produced a report.
