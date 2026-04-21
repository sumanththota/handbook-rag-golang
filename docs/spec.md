# Handbook RAG Specification (Current)

## Overview

This project is a Go-based RAG application over `grad-handbook-2025.pdf` with:

- PDF extraction via Python (`pypdf` / `PyPDF2`)
- Embeddings from Ollama (`qllama/bge-small-en-v1.5`, 384-dim vectors)
- Vector storage and search in Qdrant
- Answer generation via OpenRouter or Groq through an OpenAI-compatible client
- Web UI with HTMX and SSE token streaming

## Architecture

1. User asks a question in the web UI.
2. Backend embeds the question using Ollama.
3. Backend retrieves top-K similar chunks from Qdrant.
4. Backend builds a grounded prompt with page citations.
5. Backend streams answer tokens from the selected LLM provider.

## Ingestion Pipeline

1. Extract text by page from the configured PDF using Python parser.
2. Split each page into fixed-size word chunks.
3. Embed each chunk with Ollama.
4. Upsert vectors into Qdrant with payload including chunk text and page number.

## Retrieval and Prompting

- Retrieval mode: similarity search
- Default `top_k`: `10`
- Prompt format includes context as `[Page X]: ...`
- Instruction enforces context-grounded answers with page citations

## Runtime Components

- Qdrant at `QDRANT_HOST` (default `http://localhost:6333`)
- Ollama at `OLLAMA_HOST` (default `http://localhost:11434`)
- Server entrypoint: `go run ./cmd/server`
- Eval entrypoint: `go run ./cmd/eval`

## Environment Variables

- `HANDBOOK_PATH` (required)
- `OPENROUTER_API_KEY` (optional, needed for OpenRouter models)
- `GROQ_API_KEY` (optional, needed for Groq models)
- `PORT`, `OLLAMA_HOST`, `QDRANT_HOST`, `QDRANT_COLLECTION`
