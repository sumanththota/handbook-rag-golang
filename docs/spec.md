Yes, we absolutely **can and should** use `BAAI/bge-small-en-v1.5`. It is one of the most efficient models for RAG due to its small dimension (384) and high retrieval accuracy.

In Golang, because there is no native `sentence-transformers` library like in Python, we use that **exact model** in one of two ways. I have updated the specification below to reflect the **Ollama** method, as it is the most stable way to run that specific BAAI model locally with Go.

---

# Revised Technical Specification: Go-Based Handbook RAG

## 1. Overview

The system is a Go-native RAG application that indexes the `grad-handbook-2025.pdf`. It uses **BAAI/bge-small-en-v1.5** for local embeddings, **Qdrant** for vector storage, and **OpenRouter** for inference.

## 2. Tech Stack Mapping

| Component         | Technology                 | Implementation Detail                                |
| :---------------- | :------------------------- | :--------------------------------------------------- |
| **Orchestration** | **Langchaingo**            | Manages the RAG workflow and LLM prompts.            |
| **PDF Parsing**   | **ledongthuc/pdf**         | Extracts text and page numbers from the source PDF.  |
| **Vector DB**     | **Qdrant**                 | Local Docker container on port 6333.                 |
| **Embeddings**    | **BAAI/bge-small-en-v1.5** | Running locally via **Ollama** (Model: `bge-small`). |
| **LLM Provider**  | **OpenRouter**             | Model: `nvidia/nemotron-3-nano-30b-a3b:free`.        |
| **Web UI**        | **HTMX + SSE**             | Provides a real-time streaming chat interface.       |

---

## 3. Implementation Details

### 3.1 Embedding Logic (BAAI/bge-small-en-v1.5)

To use the BAAI model locally in Go:

- **Provider**: Ollama acts as the local inference engine.
- **Model**: `bge-small` (which is the Ollama-compiled version of `BAAI/bge-small-en-v1.5`).
- **Integration**: The Go code uses the `embeddings/ollama` package to send text chunks to the local endpoint and receive 384-dimensional vectors.

### 3.2 Ingestion Pipeline

1.  **Text Extraction**: Use `ledongthuc/pdf` to read the handbook.
2.  **Chunking Strategy**:
    - **Size**: 300-400 tokens per chunk.
    - **Metadata**: Capture `page_number` for every chunk.
3.  **Vectorization**: Batch-process chunks through the local BAAI model.
4.  **Storage**: Upsert into Qdrant with the following collection config:
    - `Size: 384`
    - `Distance: Cosine`

### 3.3 Retrieval & LLM Pipeline

1.  **Search**: Perform a similarity search for the user's query against Qdrant.
2.  **Top-K**: Retrieve the **top 10** relevant chunks (No reranking).
3.  **Prompting**:
    - Context is injected as: `[Page X]: <text_content>`.
    - Strict Instruction: "Answer using only the provided context. Cite page numbers."
4.  **Streaming**: Response is streamed from OpenRouter using the Nvidia Nemotron model.

---

## 4. System Architecture Diagram

1.  **User** $\to$ **Web UI** (HTMX)
2.  **Web UI** $\to$ **Go Backend** (Chat Request)
3.  **Go Backend** $\to$ **Ollama** (Vectorize Query using BAAI/bge-small)
4.  **Go Backend** $\to$ **Qdrant** (Vector Search Top-10)
5.  **Go Backend** $\to$ **OpenRouter** (Nvidia Nemotron + Context)
6.  **OpenRouter** $\to$ **Go Backend** (Streamed Tokens)
7.  **Go Backend** $\to$ **Web UI** (SSE real-time update)

---

## 5. Environment & Dependencies

### Required Sidecars (Running via Docker/Local)

- **Qdrant**: `docker run -p 6333:6333 qdrant/qdrant`
- **Ollama**: `ollama pull bge-small` (This downloads the BAAI model).

### Go Environment Variables

- `OPENROUTER_API_KEY`: Your API key.
- `HANDBOOK_PATH`: Path to `grad-handbook-2025.pdf`.
- `OLLAMA_HOST`: `http://localhost:11434`.

---

## 6. Known Challenges & Solutions

- **LlamaIndex Parity**: Since we aren't using Python, we manually handle the metadata "Page Number" injection into the prompt, ensuring the Go version matches LlamaIndex's citation accuracy.
- **Streaming SSE**: Standard HTTP requests in Go will be upgraded to `text/event-stream` to allow HTMX to "type out" the response as it arrives from OpenRouter.

**Is this specification ready for implementation, or would you like to adjust the chunking size or the number of context results (Top-K)?**
