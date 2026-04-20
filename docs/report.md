# RAG Retrieval Analysis Report

_Generated: 2026-04-20_

---

## 1. CHUNKING

**Strategy:** Fixed word-count (no overlap, no semantic awareness)

| Parameter | Value |
|-----------|-------|
| Strategy | Fixed word-count |
| Chunk size | 180 words |
| Overlap | **0** (none) |
| Preprocessing | `strings.Fields()` whitespace split only |

**Key code:**

`internal/chunk/chunk.go:16–49`
```go
func Build(inputs []Input, wordsPerChunk int) []Chunk {
    if wordsPerChunk <= 0 {
        wordsPerChunk = 350
    }
    for _, in := range inputs {
        words := strings.Fields(in.Text)
        for start := 0; start < len(words); start += wordsPerChunk {
            // no overlap — each window advances by the full chunk size
```

`internal/rag/service.go:37`
```go
chunkWords: 180,
```

`internal/rag/service.go:67–68`
```go
chunks := chunk.Build(inputs, s.chunkWords)
log.Printf("[rag][ingest] built chunks=%d words_per_chunk=%d", len(chunks), s.chunkWords)
```

**Verdict:** Zero overlap means a sentence that straddles two chunk boundaries is split and neither chunk contains the full context. Semantic coherence is not considered.

---

## 2. EMBEDDING

**Model:** BAAI/bge-small-en-v1.5 via Ollama (384-dimensional)

| Parameter | Value |
|-----------|-------|
| Model | `qllama/bge-small-en-v1.5` |
| Dimensions | 384 |
| Batch size | **1** (no batching) |
| Caching | **None** |
| Timeout | 60 s per request |

**Key code:**

`internal/rag/service.go:36`
```go
embedModel: "qllama/bge-small-en-v1.5",
```

`internal/embed/ollama.go:28–35`
```go
func (c *OllamaClient) Embed(ctx context.Context, model, text string) ([]float64, error) {
    trimmed := strings.TrimSpace(text)
    if trimmed == "" {
        return nil, fmt.Errorf("empty text provided for embedding")
    }
    // single-request — no batch API
```

`internal/embed/ollama.go` — retry logic trims text progressively if context-length is exceeded (up to 4 attempts via `shrinkText()`).

**Verdict:** Every chunk and every query triggers an individual HTTP round-trip to Ollama. No embedding cache exists, so identical queries are re-embedded on every call.

---

## 3. VECTOR STORE

**Store:** Qdrant (default `http://localhost:6333`)

| Parameter | Value |
|-----------|-------|
| Store | Qdrant |
| Index type | Default Qdrant collection |
| Dimensions | 384 |
| Similarity metric | **Cosine** |
| Payload stored | `text`, `page` |

**Key code:**

`internal/qdrant/client.go:33–46`
```go
func (c *Client) EnsureCollection(ctx context.Context, name string) error {
    body := map[string]any{
        "vectors": map[string]any{
            "size":     384,
            "distance": "Cosine",
        },
    }
```

`internal/qdrant/client.go:85–115`
```go
func (c *Client) Search(ctx context.Context, collection string, vector []float64, limit int) ([]SearchResult, error) {
    body := map[string]any{
        "vector":       vector,
        "limit":        limit,
        "with_payload": true,
    }
```

---

## 4. RETRIEVAL

**Method:** Pure cosine similarity — no MMR, no hybrid, no filters

| Parameter | Value |
|-----------|-------|
| k | **10** |
| Method | Similarity search |
| MMR / hybrid | None |
| Filters | None |

**Key code:**

`internal/config/config.go:18,32`
```go
TopK int
// ...
TopK: 10,
```

`internal/rag/service.go:96–103`
```go
results, err := s.qdrant.Search(ctx, s.collection, queryEmbedding, s.topK)
if err != nil {
    return "", fmt.Errorf("search qdrant: %w", err)
}
log.Printf("[rag][query] retrieved context_chunks=%d", len(results))
```

All 10 results are used verbatim in prompt construction. No score threshold filtering is applied.

---

## 5. RERANKING

**None implemented.**

`docs/spec.md:50`
```
**Top-K**: Retrieve the **top 10** relevant chunks (No reranking).
```

Results from Qdrant flow directly into `contextBuilder` (`internal/rag/service.go:105–110`) without any cross-encoder scoring, MMR diversity pass, or relevance threshold.

---

## 6. QUERY PROCESSING

**Method:** Raw embed — no expansion, no HyDE, no preprocessing beyond whitespace trim

| Technique | Present |
|-----------|---------|
| Query expansion | No |
| HyDE | No |
| Multi-query | No |
| Pseudo-relevance feedback | No |
| Preprocessing | `strings.TrimSpace()` only |

**Key code:**

`cmd/server/main.go:127`
```go
question := strings.TrimSpace(r.FormValue("question"))
```

`internal/rag/service.go:90–94`
```go
queryEmbedding, err := s.embedClient.Embed(ctx, s.embedModel, question)
if err != nil {
    return "", fmt.Errorf("embed query: %w", err)
}
```

The user's raw question is embedded and used as the search vector directly.

---

## 7. LATENCY RISKS

### A. Sequential ingestion — no batching (HIGH)

`internal/rag/service.go:70–82`
```go
for _, c := range chunks {
    vec, err := s.embedClient.Embed(ctx, s.embedModel, c.Text)   // 1 HTTP call per chunk
    if err != nil {
        return 0, fmt.Errorf("embed chunk %d: %w", c.ID, err)
    }
    if err := s.qdrant.Upsert(ctx, s.collection, c.ID, vec, c.Text, c.Page); err != nil {
        return 0, fmt.Errorf("upsert chunk %d: %w", c.ID, err)
    }
```

Each chunk emits one Ollama HTTP round-trip followed by one Qdrant upsert. A 1 000-chunk document requires 2 000 sequential HTTP calls.

### B. Query re-embedding on every request — no cache (HIGH)

`internal/rag/service.go:90`
```go
queryEmbedding, err := s.embedClient.Embed(ctx, s.embedModel, question)
```

No in-memory or persistent cache. Repeated identical queries re-embed every time.

### C. Infinite LLM timeout (MEDIUM)

`internal/llm/openai_compatible.go:23`
```go
client: &http.Client{
    Timeout: 0,   // infinite — hangs indefinitely on a stalled connection
},
```

A stalled streaming connection blocks the goroutine forever.

### D. Synchronous vector search (LOW–MEDIUM)

`internal/rag/service.go:96`
```go
results, err := s.qdrant.Search(ctx, s.collection, queryEmbedding, s.topK)
```

Blocking HTTP call; acceptable for single requests but limits concurrency under load.

### E. No connection pooling or concurrency control

The ingestion loop has no goroutine pool or semaphore. Parallelising it naively would flood Ollama with unbounded concurrent requests.

---

## Verdict

This implementation is **functionally correct and readable**: Qdrant with cosine similarity is a solid choice for a 384-dimensional bge-small model, fixed-word chunking is simple to reason about, and the code is clean with good logging. Where it underperforms is in production readiness. Zero chunk overlap means context at boundaries is lost, directly harming recall for multi-sentence answers. There is no query-side intelligence — no HyDE, expansion, or reranking — so retrieval quality is entirely at the mercy of raw embedding similarity. Most critically, ingestion is fully sequential with one HTTP call per chunk and per upsert, making large document ingestion very slow; combined with the absence of any embedding cache and an infinite LLM timeout, the system has clear latency and reliability gaps that would surface immediately under real workloads. Priority fixes: add a simple LRU query-embedding cache, parallelise ingestion with a bounded worker pool, set a finite LLM timeout, and add a small chunk overlap (≈20 words) to recover boundary context.
