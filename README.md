# GovResearch-AI (PoC with MCP)

GovResearch-AI PoC is a simplified research workflow demonstrating MCP (Model Context Protocol) integration. It uses MCP Client to invoke tools on MCP Servers for web search and local document access.

## Features

- **MCP-based Architecture**: Researcher uses MCP Client to invoke tools on MCP Servers
- **Simple LangGraph workflow**: Planner → Research → Report Writer → END
- **Tavily MCP Server**: Web search via MCP protocol
- **Filesystem MCP Server**: Local document reading via MCP protocol
- **Ollama integration**: Local LLM inference (qwen2.5:7b default)
- **Streamlit UI**: Simple web interface for research queries
- **Markdown reports**: Distinguishes web sources and local documents

## MCP Structure

### MCP Client (`mcp/client.py`)

The MCP Client is responsible for:
- Registering MCP Servers
- Invoking tools on registered servers
- Managing tool execution

```python
mcp_client = MCPClient()
mcp_client.register_server("filesystem", filesystem_server)
mcp_client.register_server("tavily", tavily_server)
result = mcp_client.call_tool("tavily", "search", query="...", max_results=5)
```

### Filesystem MCP Server (`mcp/servers/filesystem.py`)

The Filesystem MCP Server provides tools for:
- `read_text_file`: Read text files from local filesystem
- `read_markdown`: Read markdown files
- `list_files`: List files in a directory

**Purpose**: Enable Research Agent to access local reference documents.

### Tavily MCP Server (`mcp/servers/tavily.py`)

The Tavily MCP Server provides tools for:
- `search`: Perform web search with query and max_results parameters

**Purpose**: Enable Research Agent to perform web searches via MCP protocol.

## Workflow

```
User Query
    ↓
Planner (generate search keywords)
    ↓
Researcher (MCP Client)
    ├─→ Filesystem MCP (read local documents)
    └─→ Tavily MCP (web search)
    ↓
Report Writer (generate Markdown with web/local sources)
    ↓
END
```

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

### Step 4: Configure Tavily API

Get your free Tavily API key from [tavily.com](https://tavily.com).

Edit `.env` to add Tavily configuration:

```dotenv
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.2
TOP_P=0.9
MAX_TOKENS=2048
TAVILY_API_KEY=tvly-xxxxxxxx
SEARCH_MAX_RESULTS=5
SEARCH_TOPIC=general
```

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

## Demo Scenario

1. **User submits query**: "공공기관 AI 서비스 현황"
2. **Planner generates search keywords**: ["공공기관 AI 서비스 현황"]
3. **Researcher uses MCP Client with rule-based tool selection**:
   - Topic contains "현황" → Use Filesystem MCP
   - Topic contains "서비스" → Use Tavily MCP
   - Calls `Filesystem MCP` → `list_files("data/")` → Finds 2 files
   - Calls `Filesystem MCP` → `read_markdown("data/공공AI사례.md")` → Reads local document
   - Calls `Filesystem MCP` → `read_text_file("data/AI정책현황.txt")` → Reads local document
   - Calls `Tavily MCP` → `search(query="...", max_results=5)` → Performs web search
4. **Report Writer uses Ollama**:
   - Combines web results and local documents
   - Sends to Ollama with structured prompt
   - Generates professional Markdown report with:
     - Executive Summary
     - 핵심 내용
     - 주요 사례
     - 시사점
     - 로컬 참고자료
     - 웹 참고자료

## Demo Flow

```
User Query
    ↓
Planner (generate search keywords)
    ↓
Researcher (MCP Client)
    ├─→ Rule-based Tool Selection
    │   ├─→ Topic contains "현황" → Use Filesystem
    │   └─→ Topic contains "서비스" → Use Tavily
    ├─→ Filesystem MCP
    │   ├─→ list_files("data/")
    │   ├─→ read_markdown("data/공공AI사례.md")
    │   └─→ read_text_file("data/AI정책현황.txt")
    └─→ Tavily MCP
        └─→ search(query="...", max_results=5)
    ↓
Report Writer (Ollama)
    ├─→ Build prompt with web + local data
    ├─→ Call Ollama LLM
    └─→ Generate structured Markdown report
    ↓
END
```

## Architecture

```text
main.py
  └── app/workflow.py
        └── app/graph.py (LangGraph StateGraph with MCP)
              ├── mcp/client.py (MCP Client)
              ├── mcp/servers/filesystem.py (Filesystem MCP Server)
              ├── mcp/servers/tavily.py (Tavily MCP Server)
              ├── agents/planner.py (generate search keywords)
              ├── agents/researcher.py (use MCP Client for tools)
              └── agents/report_writer.py (generate Markdown report)

Shared state: app/state.py
Configuration: config/settings.py
Logging: tools/logger.py
Ollama client: tools/ollama_client.py
Tavily search: tools/tavily_search.py
UI: ui.py (Streamlit)
```

## Report Structure

The PoC generates simple Markdown reports with:
- **요약**: Total sources (web + local)
- **웹 검색 결과**: Top 5 web search results from Tavily MCP
- **로컬 참고자료**: Local documents from Filesystem MCP
- **참고 URL**: All web search URLs

## Troubleshooting

### Tavily API key not configured

```bash
TavilySearchError: Tavily API key is not configured.
```

**Solution**: Add your Tavily API key to `.env`:
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
│   ├── planner.py          # Search keyword generation
│   ├── researcher.py       # MCP-based research (uses MCP Client)
│   └── report_writer.py    # Markdown report generation
├── app/
│   ├── graph.py            # LangGraph workflow with MCP initialization
│   ├── state.py            # Shared state definition (includes local_documents)
│   └── workflow.py         # Workflow orchestration
├── config/
│   └── settings.py         # Configuration management
├── mcp/
│   ├── client.py           # MCP Client for tool invocation
│   └── servers/
│       ├── filesystem.py   # Filesystem MCP Server
│       └── tavily.py       # Tavily MCP Server
├── tools/
│   ├── logger.py           # Logging utilities
│   ├── ollama_client.py    # Ollama LLM client
│   └── tavily_search.py    # Tavily search integration
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI
└── requirements.txt        # Python dependencies
```

## Future Enhancements

This PoC demonstrates MCP integration. Future versions may add:
- More MCP Servers (Database, API, etc.)
- Advanced tool orchestration
- RAG with ChromaDB for semantic search
- PDF document processing via MCP
- Docker containerization
- Comprehensive testing
