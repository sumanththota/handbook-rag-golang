# Handbook RAG (Go)

Go-based RAG app for `grad-handbook-2025.pdf` using:
- Ollama embeddings (`qllama/bge-small-en-v1.5`, BAAI bge-small-en-v1.5)
- Qdrant vector search (cosine, 384 dims)
- Provider-aware streaming chat (OpenRouter or Groq)
- HTMX + SSE web interface

## Prerequisites

- Go 1.22+
- Qdrant running locally:
  - `docker run -p 6333:6333 qdrant/qdrant`
- Ollama with model:
  - `ollama pull qllama/bge-small-en-v1.5`

## Environment

Set:

- `HANDBOOK_PATH` (path to `grad-handbook-2025.pdf`)
- Provider keys (set one or both based on UI model choice):
  - `OPENROUTER_API_KEY` (required for OpenRouter-selected models)
  - `GROQ_API_KEY` (required for Groq-selected models)
- Optional:
  - `PORT` (default: `8080`)
  - `OLLAMA_HOST` (default: `http://localhost:11434`)
  - `QDRANT_HOST` (default: `http://localhost:6333`)
  - `QDRANT_COLLECTION` (default: `handbook_chunks`)

## Run

```bash
go run ./cmd/ingest
go run ./cmd/server
```

Run ingestion explicitly as an admin step (re-run whenever the handbook changes), then start the server and open `http://localhost:8080` to ask questions.
