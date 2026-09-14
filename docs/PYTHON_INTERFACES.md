# Python Interface Contract — Tier 0 Modules

Pins exact signatures for the 6 Tier-0 modules (`config`, `chunk`, `embed`, `llm`, `llamaparse`, `store`) so independently-spawned porting agents (tickets 03-08) produce compatible shapes. Derived from the Go source in `internal/{config,chunk,embed,llm,llamaparse}` and the store contract in `MIGRATION.md` §4. Conventions follow `MIGRATION.md` §3 — do not deviate without updating this doc first.

Stack assumptions pinned here (not stated elsewhere, needed to make signatures concrete): Pydantic v2 (`pydantic.BaseModel`, `pydantic-settings.BaseSettings`), `httpx.AsyncClient` for HTTP, `asyncpg` for Postgres.

Every module-level exception inherits directly from `Exception` (no shared base class — none of the Tier-2/3 call sites need to catch across modules, so a hierarchy would be pure speculation).

---

## Async/sync summary

| Module | Sync or async | Why |
|---|---|---|
| `config` | sync | env/file reads only, no network |
| `chunk` | sync | pure function, no I/O (per MIGRATION.md §2/§3) |
| `embed` | async | outbound HTTP to Ollama |
| `llm` | async | outbound HTTP to OpenRouter/Groq, incl. streaming |
| `llamaparse` | async | outbound HTTP to LlamaCloud (upload/start/poll) |
| `store` | async | Postgres I/O via asyncpg |

## Timeout summary

Rule (MIGRATION.md §3): every outbound HTTP call must carry an explicit timeout — same discipline as the 60s Ollama timeout in `internal/embed/ollama.go`. This applies even where the Go source has a gap (`internal/llm` currently sets none).

| Module | Client | Timeout | Source |
|---|---|---|---|
| `embed.OllamaClient` | `httpx.AsyncClient(timeout=60.0)` | 60s flat, all calls | matches Go `http.Client{Timeout: 60 * time.Second}` |
| `llm.OpenAICompatibleClient` | `httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0))` | 120s read (covers streaming), 10s connect | **new** — Go's `http.Client{}` has no timeout (a gap); §3 requires Python set one explicitly regardless |
| `llamaparse` upload/start | `httpx.AsyncClient(timeout=120.0)` | 120s | matches Go `http.Client{Timeout: 120 * time.Second}` |
| `llamaparse` poll (per request) | `httpx.AsyncClient(timeout=60.0)` | 60s per poll request | matches Go `pollClient := &http.Client{Timeout: 60 * time.Second}` |
| `llamaparse` poll (overall job) | `asyncio.timeout(25 * 60)` wrapping the poll loop | 25 min | matches Go `context.WithTimeout(ctx, 25*time.Minute)` |
| `store.PostgresStore` | `asyncpg.create_pool(..., command_timeout=30, timeout=10)` | 30s per command, 10s connect | **new** — no Go equivalent; extends §3 discipline to DB I/O |

---

## 1. `app/config.py` (from `internal/config/config.go`)

Go's `Config` struct has no JSON tags (it's env-loaded only, never serialized), so the "field names match Go JSON tags" rule doesn't constrain attribute casing here — Pythonic `snake_case` is used. **Env var names are unchanged** per MIGRATION.md §2 table.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    port: str = Field(default="8080", alias="PORT")
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    handbook_path: str = Field(alias="HANDBOOK_PATH")              # required, no default
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_api_key: str = Field(default="", alias="OLLAMA_API_KEY")  # trimmed of whitespace
    collection_name: str = Field(default="handbook_chunks", alias="QDRANT_COLLECTION")
    top_k: int = 10                                                  # not env-configurable in Go either

    # NEW — no Go equivalent. internal/config has QdrantHost; the Postgres store
    # (MIGRATION.md §4) needs a DSN instead. Flagged as an intentional addition,
    # not a silent scope-creep — ticket 08 (store) must consume this field.
    database_url: str = Field(alias="DATABASE_URL")

def load_config() -> Settings:
    """Equivalent of Go's config.Load(). Raises ConfigError (wrapping pydantic's
    ValidationError) if HANDBOOK_PATH or DATABASE_URL is missing."""
```

- `Config.QdrantHost` is dropped (no target — store is Postgres now); `DATABASE_URL` replaces it.
- `getEnvOrDefault` / `.env` loading is native `BaseSettings` behavior (`env_file=".env"`) — do not hand-roll the Go `loadDotEnv` scanner.
- `load_config()` is **sync** — no I/O beyond local file/env reads.

## 2. `app/chunk.py` (from `internal/chunk/chunk.go`)

Go's `Input`/`Chunk` structs also have no JSON tags and are never serialized — Pythonic field names used.

```python
from pydantic import BaseModel

class ChunkInput(BaseModel):
    page: int
    text: str

class Chunk(BaseModel):
    id: int
    page: int
    text: str

def build(inputs: list[ChunkInput], words_per_chunk: int = 350) -> list[Chunk]:
    """Pure, sync. Mirrors chunk.Build byte-for-byte: words_per_chunk <= 0 falls
    back to 350; splits on strings.Fields-equivalent whitespace tokenization;
    empty-text inputs are skipped; ids are assigned sequentially across all inputs."""
```

- No exceptions — Go's `Build` never errors; neither does this.
- **Sync**, per MIGRATION.md §2 ("pure function, no I/O").

## 3. `app/embed.py` (from `internal/embed/ollama.go`)

```python
import httpx

class EmbedError(Exception):
    """Raised for any embedding failure: empty input, HTTP failure, empty
    embedding, or exhausted shrink retries."""

class OllamaClient:
    def __init__(self, base_url: str) -> None:
        """Strips trailing '/'. Owns an httpx.AsyncClient(timeout=60.0)
        (see Timeout summary) — same as Go's 60s http.Client."""

    async def embed(self, model: str, text: str) -> list[float]:
        """Async. Raises EmbedError on: empty/whitespace-only text; non-2xx
        response that isn't a retryable context-length error; empty embedding
        vector; or exhaustion of the shrink-retry loop.

        Ports embedOnce's shrink retry LITERALLY (MIGRATION.md §3 — business
        rule, not boilerplate, flagged for human review before merge):
        up to 4 attempts; on a 500 whose body contains "context length"
        (case-insensitive), call _shrink_text(current) and retry; any other
        non-2xx is immediately fatal (no retry)."""

    async def healthy(self) -> None:
        """Async. GET {base_url}/api/tags. Raises EmbedError on transport
        failure or non-2xx status (message includes status + truncated body,
        matching Go's 2048-byte body cap)."""

    async def _embed_once(self, model: str, text: str) -> list[float]:
        """Internal. POST {base_url}/api/embeddings with {"model", "prompt"}.
        Distinguishes retryable (500 + "context length" in body) from fatal
        errors internally — embed() is the only public entrypoint that surfaces
        this distinction, matching Go's (vec, retry, err) return via control flow
        instead of a tuple (MIGRATION.md §3: no Go-style tuple returns)."""

def _shrink_text(text: str) -> str:
    """Sync, pure. Ports shrinkText literally: word-split, halve (floor),
    minimum 40 words; returns "" if input already has <= 40 words (Go: <= 40
    -> "" signals give-up)."""
```

- Batch size stays 1 request per `embed()` call (MIGRATION.md §3 — known issue, not fixed during port).
- Timeout: flat 60s via the client's `httpx.AsyncClient(timeout=60.0)` — see Timeout summary above.

## 4. `app/llm.py` (from `internal/llm/openai_compatible.go`)

Go's `ChatMessage` has JSON tags `role`/`content` — already lowercase, so Pydantic field names are unchanged.

```python
from collections.abc import AsyncIterator
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class LLMError(Exception):
    """Raised for any chat-completion failure: non-2xx provider response,
    empty choices array, or response-decode failure."""

class OpenAICompatibleClient:
    def __init__(self, base_url: str, headers: dict[str, str] | None = None) -> None:
        """Strips trailing '/'. Owns an httpx.AsyncClient with the timeout in
        the Timeout summary above (NEW relative to Go — see note)."""

    async def complete(
        self, api_key: str, model: str, temperature: float, messages: list[ChatMessage]
    ) -> str:
        """Async, non-streaming (stream=False). POST {base_url}/chat/completions.
        Sets Authorization: Bearer {api_key} only if api_key is non-blank after
        strip. Raises LLMError on non-2xx, empty choices, or decode failure.
        Returns choices[0].message.content, stripped."""

    def stream_answer(
        self, api_key: str, model: str, prompt: str
    ) -> AsyncIterator[str]:
        """Async generator (replaces Go's onToken callback with Python's native
        idiom — MIGRATION.md doesn't mandate a literal callback port here since
        this is HTTP-client boilerplate, not the business rule; the SSE event
        contract this feeds is untouched, see §3 SSE row).

        POSTs stream=True with the same fixed system prompt as Go (verbatim,
        including the "I couldn't find that in the handbook..." fallback
        sentence — this IS a business-rule string, port it unchanged).
        Parses "data: " lines; stops on "data: [DONE]"; skips lines that fail
        JSON decode or have an empty choices array or empty delta content;
        yields each non-empty delta.content token. Raises LLMError if the
        initial request fails (non-2xx) — errors during stream iteration
        propagate as whatever httpx raises (matching Go's scanner.Err() ->
        wrapped error, i.e. do not swallow mid-stream transport errors)."""
```

- Both methods are **async**.
- Timeout: see Timeout summary — this is the one module where Python is *stricter* than Go (Go has no timeout here; Python must set one per §3).

## 5. `app/llamaparse.py` (from `internal/llamaparse/client.go`)

Go's `Page` struct has no JSON tags — Pythonic field names used.

```python
class LlamaParseError(Exception):
    """Raised for any stage failure: missing API key, upload failure, job-start
    failure, poll failure, job FAILED/ERROR status, or a completed job with no
    markdown/text pages."""

class Page(BaseModel):
    number: int
    text: str

async def extract_pages(api_key: str, pdf_path: str, tier: str = "cost_effective") -> list[Page]:
    """Async. Raises LlamaParseError immediately if api_key is blank.
    base_url resolution: os.environ.get("LLAMA_CLOUD_BASE_URL", "").rstrip("/")
    or "https://api.cloud.llamaindex.ai" — read directly from the environment
    here, NOT via config.Settings, mirroring Go's os.Getenv call inside the
    llamaparse package rather than plumbing it through internal/config.

    Sequence (each sub-step raises LlamaParseError, wrapping the underlying
    cause, matching Go's fmt.Errorf("llamaparse upload: %w", err) wrapping):
      1. _upload_file(client, base, api_key, pdf_path) -> file_id
         multipart POST {base}/api/v1/files/, field "purpose"="parse",
         file field name "upload_file" (not "file" — LlamaCloud v1 quirk).
      2. _start_parse_job(client, base, api_key, file_id, tier) -> job_id
         POST {base}/api/v2/parse, JSON {"file_id", "tier", "version": "latest"}.
         Reads top-level "id", falling back to nested "job.id".
      3. _poll_parse_job(base, api_key, job_id) -> list[Page]
         GET {base}/api/v2/parse/{job_id}?expand=markdown every 1.5s until
         status (top-level or job.status, uppercased) is COMPLETED/COMPLETE/
         SUCCESS (-> parse pages) or FAILED/ERROR (-> raise). Prefers
         markdown.pages[].markdown over text.pages[].text; strips + collapses
         internal whitespace on each page's text (Go's normalizeWhitespace);
         drops pages with empty text after stripping.
      Raises LlamaParseError if the final page list is empty.
    """
```

- All three internal steps are **async**.
- Timeouts: see Timeout summary — upload/start client at 120s, poll-request client at 60s, poll-loop deadline at 25 min via `asyncio.timeout`.
- The 1.5s poll interval is ported literally (`asyncio.sleep(1.5)`), same as Go's `time.NewTicker(1500 * time.Millisecond)`.

## 6. `app/store.py` (new — Postgres/pgvector, per MIGRATION.md §4; no Go port, replaces `internal/qdrant/client.go`)

Field names for `SearchResult` are pinned to `text`/`page`/`score` (lowercase) because that shape flows unchanged into the existing SSE `sources` JSON payload in `cmd/server/main.go:384-386` (`json:"page"`, `json:"score"`, `json:"text"`) — the HTMX frontend contract this must not break (MIGRATION.md §3 SSE row). This is the concrete instance of "field names match Go JSON tags" even though the struct itself is new.

`upsert`/`search` signatures are kept **parameter-for-parameter identical** to `internal/qdrant/client.go`'s `Upsert`/`Search` (both call sites in `internal/rag/service.go:92,125` pass `collection`/`id`/`vector`/`text`/`page` and `collection`/`vector`/`limit` respectively) so `app/rag.py` (Tier 2) ports with a mechanical swap of `qdrant.Client` → `store.PostgresStore`, no call-site restructuring.

```python
import asyncpg

class StoreError(Exception):
    """Raised for any Postgres/pgvector failure: connection failure, schema
    setup failure, upsert failure, or search failure."""

class SearchResult(BaseModel):
    text: str
    page: int
    score: float

class PostgresStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        """Takes an already-created pool (created via asyncpg.create_pool(
        dsn=settings.database_url, command_timeout=30, timeout=10) — see
        Timeout summary). Pool ownership/lifecycle is main.py's (Tier 3)
        responsibility, not this class's — mirrors Go's qdrant.Client owning
        only an http.Client, not a process lifecycle."""

    async def ensure_schema(self, collection: str) -> None:
        """Async. Equivalent of Go's EnsureCollection. Idempotent (matches
        Go's 409-as-success behavior): CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {collection} (id bigint PRIMARY KEY,
        text text NOT NULL, page int NOT NULL, embedding vector(384) NOT NULL);
        CREATE INDEX IF NOT EXISTS ... USING hnsw (embedding vector_cosine_ops).
        `collection` is used as the table name (see param-naming note below).
        Raises StoreError if the pgvector extension is unavailable — MIGRATION.md
        §1 says flag this, don't silently fall back to manual cosine SQL."""

    async def upsert(
        self, collection: str, id: int, vector: list[float], text: str, page: int
    ) -> None:
        """Async. One row per call (matches Go's one-point-per-call Upsert —
        internal/rag/service.go's Ingest loop calls this once per chunk, same
        as today). INSERT ... ON CONFLICT (id) DO UPDATE SET text=EXCLUDED.text,
        page=EXCLUDED.page, embedding=EXCLUDED.embedding (MIGRATION.md §4).
        Raises StoreError on failure."""

    async def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[SearchResult]:
        """Async. SELECT text, page, 1 - (embedding <=> $1) AS score FROM
        {collection} ORDER BY embedding <=> $1 LIMIT $2 (cosine distance ->
        cosine similarity score, so score semantics match Qdrant's Cosine
        distance metric — higher is more similar, same as today). Raises
        StoreError on failure. Returns [] on no rows (matches Go: Qdrant
        returns an empty result list, not an error, for no matches)."""

    async def healthy(self) -> None:
        """Async. SELECT 1 (or pool.acquire() round-trip). Raises StoreError
        on failure — equivalent of Go's Healthy() hitting GET /collections."""
```

- `collection` is kept as the parameter name (not renamed to `table`) purely for call-site parity with `internal/qdrant` / future `app/rag.py` — it is documented here as meaning "table name" in the Postgres implementation.
- All methods **async** (asyncpg I/O).
- Timeouts: pool-level `command_timeout=30`, `timeout=10` (connect) — see Timeout summary; this is new discipline (no Go equivalent) extending §3's HTTP-timeout rule to DB I/O.

---

## Checklist self-check (ticket 01-interface-contract.md)

1. **Exact signatures matching §3 conventions (exceptions, Pydantic, unchanged field names)** — satisfied: §1-6 above give full method/function signatures for all 6 modules; every module has a typed exception (`ConfigError`, `EmbedError`, `LLMError`, `LlamaParseError`, `StoreError` — `chunk` has none, matching Go which never errors); every struct-equivalent is a Pydantic `BaseModel`; field-name casing is justified per module (§1/§2/§5 explain why Pythonic casing is safe — no Go JSON tags exist; §3/§4/§6 keep Go's already-lowercase JSON tags; §6 pins new field names to the existing downstream JSON contract).
2. **`store.py` upsert/search match what `app/rag.py` will call** — satisfied: §6's "parameter-for-parameter identical" paragraph cites the exact Go call sites (`internal/rag/service.go:92,125`) and keeps `upsert(collection, id, vector, text, page)` / `search(collection, vector, limit)` shaped identically so the Tier-2 port is a mechanical swap.
3. **Async vs sync specified per function** — satisfied: the "Async/sync summary" table gives the per-module default, and every function/method docstring in §1-6 states "sync" or "async" explicitly.
4. **Timeout convention stated explicitly per outbound-HTTP function** — satisfied: the "Timeout summary" table gives a concrete `httpx`/`asyncpg` timeout value and Go-parity note for every HTTP- or DB-touching client (embed, llm, llamaparse ×3, store), each cross-referenced from its module section.
5. **Reviewed and approved by Sumanth before tickets 03-08 start** — not yet satisfied by this doc itself (it's a human sign-off, not something this doc can self-certify); flagging explicitly so the gate isn't skipped: **this doc requires Sumanth's review/approval before any of tickets 03-08 begin.**
