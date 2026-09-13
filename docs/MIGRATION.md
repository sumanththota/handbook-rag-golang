# Migration Spec: Go → Python FastAPI, Qdrant → Postgres (pgvector)

Agents: read this before touching any file. Don't re-derive conventions per-file — follow the mappings below. If a case isn't covered here, stop and ask rather than improvising a convention.

## 0. Repos

This is a two-repo migration, not a subdirectory port:

- **Source (this repo):** `handbook-rag-golang` — read-only reference during the port. This file lives here because it documents how to leave it.
- **Target:** [`rag-engine`](https://github.com/sumanththota/rag-engine) — a separate public repo, sibling directory on disk (`../rag-engine` relative to this repo). All `app/...` paths below are relative to that repo's root, not a subdirectory of this one.

Agents working in `rag-engine` reference this file and this repo's `internal/<module>` source by filesystem path across the sibling directories — there is no shared git history or worktree between the two repos, and none is needed.

## 1. Scope

- **Language port:** `handbook-rag` (Go, stdlib-only, 13 files / ~1400 LOC) → Python FastAPI.
- **Store port:** Qdrant (384-dim cosine, single collection `handbook_chunks`) → Postgres + `pgvector` extension.
- Both ports are collapsed into one target: the new Python storage module talks to Postgres directly. Do **not** write a Python Qdrant client as an intermediate step.
- Ollama (embeddings) and OpenRouter/Groq (chat) stay as external HTTP services — only the client code is ported, not the services themselves.
- Assumption locked in: `pgvector` extension is available on the target Postgres instance. If not, flag it — don't silently fall back to manual cosine-similarity SQL.

## 2. Target architecture

Mirror the existing package boundaries 1:1 so review can diff old-module → new-module directly:

| Go (source)              | Python (target)                    | Notes |
|---------------------------|-------------------------------------|-------|
| `internal/config`         | `app/config.py`                     | Pydantic `BaseSettings`, same env var names |
| `internal/chunk`          | `app/chunk.py`                      | Pure function, no I/O — easiest leaf |
| `internal/embed`          | `app/embed.py`                      | Ollama client, `httpx.AsyncClient` |
| `internal/llm`            | `app/llm.py`                        | OpenAI-compatible client (OpenRouter/Groq), async streaming |
| `internal/llamaparse`     | `app/llamaparse.py`                 | LlamaCloud client |
| `internal/pdfextract`     | `app/pdfextract.py`                 | Wraps llamaparse + local fallback |
| `internal/qdrant`         | `app/store.py`                      | **Replaced**, not ported — Postgres/pgvector implementation of the same retrieval interface |
| `internal/rag`            | `app/rag.py`                        | Orchestrator: ingest + retrieve + prompt build + rewrite |
| `cmd/ingest`              | `app/cli/ingest.py` or `scripts/ingest.py` | Admin ingestion entrypoint |
| `cmd/eval`                | `app/cli/eval.py`                   | Eval harness |
| `cmd/server`              | `app/main.py` (FastAPI app)         | HTTP + SSE, same routes/contract |

## 3. Coding-convention mappings (Go idiom → Python idiom)

| Go pattern | Python target | Rule |
|---|---|---|
| `func F(...) (T, error)` explicit error return | Exceptions (typed, e.g. `EmbedError`), or `Result`-style only where the caller must branch on error without unwinding | Default to exceptions; don't invent a Go-style tuple-return convention in Python |
| `context.Context` for timeout/cancellation | `httpx.Timeout`, `asyncio` cancellation, request-scoped `async with` | Every outbound HTTP call must carry an explicit timeout — same discipline as the 60s Ollama timeout in `internal/embed/ollama.go` |
| Structs with JSON tags | Pydantic `BaseModel` | Field names stay identical to the Go JSON tags — no casing changes |
| `log.Printf("[rag][ingest] ...")` prefixed logs | Python `logging` with matching logger names (`rag.ingest`, `rag.retrieve`) | Keep the same log-line semantics so ops greps still work |
| Manual retry loop (`shrinkText()` progressive shrink in embed client) | Same algorithm, ported literally — this is a business rule, not boilerplate | Flag for human review (see §6), don't "clean up" the retry logic while porting it |
| Single-request embedding (batch size 1) | Keep batch size 1 unless explicitly asked to fix it | Don't silently improve behavior mid-port — `docs/report.md` already tracks this as a known issue; fix it as a separate, reviewed change after parity is established, not during the port |
| SSE streaming (`cmd/server` hand-rolled) | FastAPI `StreamingResponse` with `text/event-stream`, same event names (`sources`, `error`, token chunks) | The HTMX frontend contract (`sourcesHint`, `streamerror`) must not change — see `docs/ui-execution-tracker.md` |

## 4. Store migration: Qdrant → Postgres/pgvector

| Qdrant concept | Postgres/pgvector target |
|---|---|
| Collection `handbook_chunks` | Table `handbook_chunks` |
| Point ID | `id` (uuid or bigserial) |
| Vector (384-dim, cosine) | `embedding vector(384)`, index `USING hnsw (embedding vector_cosine_ops)` |
| Payload (`text`, `page`) | Plain columns `text text`, `page int`, not JSONB — payload shape is fixed and small, no need for schemaless storage |
| Cosine similarity search, top_k | `ORDER BY embedding <=> :query_vector LIMIT :top_k` |
| Upsert on ingest | `INSERT ... ON CONFLICT (id) DO UPDATE` |

`app/store.py` exposes the same two operations `internal/qdrant/client.go` exposes today (upsert, search) — same function signatures conceptually, so `app/rag.py` doesn't need to know the store changed shape.

## 5. Dependency-ordered migration batches (leaves first)

Computed from the actual Go import graph:

**Tier 0 — no internal deps, fully parallel (6 independent agents possible):**
`config`, `chunk`, `embed`, `llm`, `llamaparse`, and the new `store` (built against Postgres from scratch, not a port of `qdrant`)

**Tier 1 — depends only on Tier 0:**
`pdfextract` (needs `llamaparse`)

**Tier 2 — depends on all of Tier 0 + Tier 1, single agent (this is the integration point, don't split it):**
`rag` (needs `chunk`, `embed`, `llm`, `pdfextract`, `store`)

**Tier 3 — depends on Tier 2, parallel across the 3 entrypoints:**
`cmd/server` → `app/main.py`, `cmd/ingest` → ingest script, `cmd/eval` → eval script

Rule: never start a tier until every module in the prior tier compiles and passes its characterization tests. Don't let an agent "peek ahead" into a higher tier to unblock itself — that's the fast path to the "half the repo won't build" failure mode.

## 6. Characterization tests — write before any port work starts

Repo currently has **zero** Go tests. Step 0, before touching Python:

- `chunk.Build`: golden test — fixed input text → exact expected chunk boundaries (180 words, no overlap). This is pure and deterministic, cheapest test to write.
- `rag.BuildPrompt` / retrieval ranking: snapshot test against the real `grad-handbook-2025.pdf` — fixed query → expected top-K page numbers + scores (within float tolerance).
- `cmd/server` HTTP/SSE contract: capture real request/response pairs (a handful of questions) as golden fixtures — status codes, SSE event sequence, `sources` payload shape. This is the test that actually protects the frontend contract in §3.
- Ingest pipeline end-to-end: run ingest once against the real PDF into Qdrant, record chunk count + a sample of vectors/payloads as the reference to diff the new Postgres path against.

These tests are written **in Go, against the current code**, then re-run conceptually (same fixtures, same expected outputs) against the Python port. The fixtures are the ground truth — they don't get rewritten mid-migration.

## 7. Execution model

- **Mechanical vs. semantic split:** an agent ports syntax/structure/HTTP-client boilerplate unsupervised (Tier 0/1/3 leaf modules). Anything touching a business rule — the embed retry/shrink logic, chunk boundary math, prompt template, SSE event contract — gets an explicit human diff review before merge, not just a passing test.
- **Parallel agents:** one worktree per Tier-0 module (up to 6 concurrent), one worktree per Tier-3 entrypoint (3 concurrent) once Tier 2 lands. Tier 2 (`rag`) is single-agent — it's the one module every other module depends on, splitting it creates merge conflicts with no payoff.
- **Definition of "done" per module:** compiles/runs (`python -m py_compile` / import succeeds) + passes its characterization test(s) against the golden fixtures from §6, before the agent reports the module finished. Not "looks right" — passes the test.
- **Review gate per tier:** don't open Tier N+1 until every Tier N module has passed its test *and* had human review sign-off on any business-logic module in that tier.
