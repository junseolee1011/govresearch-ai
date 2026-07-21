# GovResearch-AI

GovResearch-AI is an autonomous, multi-agent research workflow for identifying
and classifying AI services used by public institutions. It plans a research
task, performs web searches via Tavily, summarizes findings using local Ollama LLM,
and produces a structured Markdown report. Sprint 4 adds Retrieval-Augmented Generation (RAG)
with ChromaDB for intelligent document caching and retrieval.

## Features

- LangGraph `StateGraph` workflow: Planner → Retriever → Decision → Research → Store → Summarizer → Report
- **RAG with ChromaDB**: Intelligent document caching and semantic retrieval
- **Embedding model**: BAAI/bge-small-en-v1.5 for semantic similarity search
- **Conditional workflow**: Skips web search if sufficient relevant documents found in cache
- Tavily web search integration for real-time research data
- Ollama integration for local LLM inference (qwen3:8b default, llama3.1:8b supported)
- Source classification: Government, Research, News, Other
- Typed, shared workflow state
- File-based prompts with variable substitution
- Structured logging with execution timing and response metrics
- AI service inventory: institution, service type, use case, and maturity
- Professional Markdown report generation with RAG-specific sections

## Requirements

- Python 3.11+
- Ollama (local LLM runtime)
- Tavily API key (free tier available at https://tavily.com)
- ChromaDB (vector database, installed via pip)
- sentence-transformers (embedding model, installed via pip)

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

### Step 4: Configure Tavily API

Get your free Tavily API key from [tavily.com](https://tavily.com).

Edit `.env` to add Tavily and RAG configuration:

```dotenv
OLLAMA_MODEL=qwen3:8b
OLLAMA_BASE_URL=http://localhost:11434
TEMPERATURE=0.2
TOP_P=0.9
MAX_TOKENS=2048
TAVILY_API_KEY=tvly-xxxxxxxx
SEARCH_MAX_RESULTS=5
SEARCH_TOPIC=general
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PATH=chroma
RETRIEVAL_K=5
SIMILARITY_THRESHOLD=0.75
MIN_RETRIEVED_DOCS=3
```

## Usage

```bash
# Run with default topic
python main.py

# Run with custom topic (Korean)
python main.py --topic "국내 공공기관 생성형 AI 구축 사례 조사"

# Run with custom topic (English)
python main.py --topic "AI services used by municipal governments"
```

The generated Markdown report includes:
- Research objective
- Search queries used
- Key findings from web search
- **Retrieved Knowledge** (from ChromaDB cache with similarity scores)
- **New Findings** (from fresh web search if cache miss)
- **Comparison Analysis** (between cached and new findings)
- Service inventory with institution details
- Service classification by type
- **Recommendations** (based on integrated findings)
- References with URLs
- Next steps

## Architecture

```text
main.py
  └── app/workflow.py
        └── app/graph.py (LangGraph StateGraph)
              ├── agents/planner.py (generate research plan)
              ├── agents/retriever.py (ChromaDB semantic search)
              ├── agents/researcher.py (Tavily web search + source classification)
              ├── agents/store.py (store results in ChromaDB)
              ├── agents/summarizer.py (Ollama: summarize with RAG context)
              └── agents/report_writer.py (Ollama: generate RAG report)

Shared state: app/state.py
Configuration: config/settings.py
Prompts: prompts/*.txt (with variable substitution)
Logging: tools/logger.py (with execution metrics)
Ollama client: tools/ollama_client.py
Tavily search: tools/tavily_search.py
Vector store: tools/vector_store.py (ChromaDB)
Embeddings: tools/embeddings.py (sentence-transformers)
Document model: models/document.py
Reports: reports/
ChromaDB: chroma/
```

## RAG Architecture

### Embedding

The system uses **sentence-transformers** to generate embeddings for semantic search:
- **Default model**: BAAI/bge-small-en-v1.5
- **Singleton pattern**: Embedding model initialized once and reused
- **Configurable**: Model can be changed via `EMBEDDING_MODEL` environment variable

### Vector Database (ChromaDB)

ChromaDB is used for persistent document storage and retrieval:
- **Storage location**: `chroma/` directory (configurable via `CHROMA_PATH`)
- **Persistence**: Automatic persistence after document insertion
- **Metadata**: Each document includes title, URL, source, domain, retrieved_date, language
- **Deduplication**: Automatic duplicate detection by URL

### Document Chunking

Documents are chunked for efficient retrieval:
- **Chunk size**: 1000 characters
- **Overlap**: 150 characters
- **Splitter**: RecursiveCharacterTextSplitter from LangChain

### Retriever

The retriever agent performs semantic search:
- **Query embedding**: Converts research topic to embedding vector
- **Similarity search**: Finds top-k similar documents (default k=5)
- **Threshold filtering**: Only returns documents above similarity threshold (default 0.75)
- **Logging**: Logs retrieved document count and similarity scores

### Decision Node

Conditional logic determines workflow path:
- **Cache hit**: If ≥ `MIN_RETRIEVED_DOCS` found, skip web search
- **Cache miss**: If insufficient documents, perform web search via Tavily
- **Configurable thresholds**: `RETRIEVAL_K`, `SIMILARITY_THRESHOLD`, `MIN_RETRIEVED_DOCS`

### Workflow Flow

```
User Query
    ↓
Planner (research plan)
    ↓
Retriever (search ChromaDB)
    ↓
Decision (enough context?)
    ↓
    ├─ Yes → Summarizer → Report Writer
    └─ No → Research (Tavily) → Store (ChromaDB) → Retriever → Summarizer → Report Writer
```

## Sprint 4 Implementation

### RAG Pipeline

- **ChromaDB integration**: Persistent vector database for document storage
- **Semantic search**: Embedding-based similarity search using sentence-transformers
- **Document chunking**: RecursiveCharacterTextSplitter for efficient retrieval
- **Conditional workflow**: Decision node to skip web search on cache hit
- **Duplicate prevention**: URL-based deduplication before storage
- **Metadata tracking**: Comprehensive metadata for each document

### New Components

- **tools/embeddings.py**: Singleton embedding model manager
- **tools/vector_store.py**: ChromaDB wrapper for document operations
- **models/document.py**: Document dataclass with metadata
- **agents/retriever.py**: Semantic search agent
- **prompts/summarizer.txt**: RAG-aware summarization prompt
- **Updated prompts/report_writer.txt**: RAG-specific report sections

### Enhanced Summarizer

- **RAG context**: Integrates retrieved documents with new search results
- **Template-based**: Uses prompt template for consistent formatting
- **Dual input**: Processes both cached knowledge and fresh findings

### Enhanced Report Writer

- **Retrieved Knowledge section**: Shows cached documents with similarity scores
- **New Findings section**: Highlights fresh web search results
- **Comparison Analysis**: LLM-generated comparison between cached and new data
- **Recommendations section**: Actionable recommendations based on integrated findings

### Updated Workflow

```
Planner → Retriever → Decision → [Research → Store] → Summarizer → Report Writer
```

### State Extensions

- `retrieved_documents`: List of documents retrieved from ChromaDB
- `cache_hit`: Boolean indicating if sufficient documents were found
- `stored_documents_count`: Number of documents stored in ChromaDB

## Testing

```bash
python -m unittest discover -s tests -v
```

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
ollama pull qwen3:8b
```

### Slow inference

- Check system resources (CPU/memory)
- Consider smaller model or increase `MAX_TOKENS`
- Reduce `TEMPERATURE` for faster responses

## Next Steps (Future Sprints)

- ChromaDB for semantic search and memory
- Streamlit UI
- Docker containerization
- Reflection agent for self-critique and refinement
- PDF upload and document analysis
