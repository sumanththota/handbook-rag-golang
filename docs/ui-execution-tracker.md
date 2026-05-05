# UI execution tracker (Handbook RAG)

Use this file to see **what was planned**, **what shipped**, and **where the code lives**. Another agent can update checkboxes and the **Last updated** line as work continues.

**Last updated:** 2026-04-21 (UI polish pass: design tokens, empty state, HTMX ingest indicator, streaming chrome, `sourcesHint` / `streamerror` contract unchanged)

**Note:** Automated fetch/MCP cannot load `http://localhost` from the tool environment; verify layout in your browser after `go run ./cmd/server`.

## Scope

- **Stack:** Go `cmd/server`, HTMX, SSE (`EventSource`), no server-side chat persistence.
- **History:** Browser `localStorage` only, schema versioned under key `handbook-rag-state`.

## Files

| Area | File |
|------|------|
| Full page UI | `cmd/server/index.html` (embedded by `main.go`) |
| HTTP + SSE + chat fragment | `cmd/server/main.go` |
| Prompt + retrieval | `internal/rag/service.go` (`BuildPrompt` returns prompt + raw hits) |

## Feature checklist

### Core chat (C)

- [x] C1 Message list (user / assistant), scroll to latest
- [x] C2 User input (textarea), submit + Shift+Enter newline
- [x] C3 Block send while streaming / HTMX in flight (`busy` flag)
- [x] C4 Streaming reply via SSE + token join heuristic (spaces between tokens)
- [x] C5 Stop / cancel stream (closes `EventSource`, frees UI)
- [x] C6 Errors visible (SSE `error` event + connection errors)
- [x] C7 Loading / streaming status text in header strip

### RAG UI (R)

- [x] R1 Sources panel (page, score, snippet) from SSE `sources` (base64 JSON)
- [x] R2 Sources refresh per answer (cleared when new stream starts)
- [x] R3 Empty retrieval still surfaces as HTTP/SSE error from server (existing behavior)

### Browser history (H)

- [x] H1 Thread list in sidebar
- [x] H2 New chat
- [x] H3 Switch thread (load messages + sources last state)
- [x] H4 Persist after each completed turn (user + assistant + sources)
- [x] H5 Thread title from first user line (truncated)
- [x] H6 Delete thread
- [x] H7 Storage errors / quota (try/catch + `storage` event warning banner)
- [x] H8 Schema `version: 1` in stored JSON

### Rendering (M)

- [x] M1 Markdown for assistant (after stream ends + on thread load) via `marked`
- [x] M2 Sanitize with `DOMPurify`
- [x] M3 Code blocks (default `marked` + copy buttons on fenced blocks after render)
- [x] M4 External links `target="_blank"` + `rel` via renderer hook

### Presentation (P)

- [x] P1 Responsive layout (sidebar collapses under 900px)
- [x] P2 Title + subtitle branding
- [x] P3 Status strip (model, streaming, storage warning)
- [x] P4 Focus input after new chat / example chip click

### Extras (X)

- [x] X1 Export active thread as `.md`
- [x] X2 Clear all local data (confirm)
- [x] X3 Example prompt chips

### Deferred / optional

- [ ] Thread rename inline (title auto from first message only for now)
- [ ] Dark mode
- [ ] Virtualized message list (only needed for very long threads)

## SSE events (contract)

| Event | Payload |
|-------|---------|
| `sources` | Base64-encoded JSON array: `[{page, score, text}]` (text truncated server-side) |
| `token` | Token string (newlines escaped as `\n` in protocol) |
| `streamerror` | Error string (named `streamerror` to avoid clashing with the EventSource connection `error` event) |
| `done` | `complete` |

## Notes for the next agent

1. If **sources** fail to parse on the client, check `handleChatStream` still base64-encodes after `json.Marshal`.
2. **Persistence** runs on `done` / user stop (partial assistant message still saved if you extend behavior).
3. **Eval / other binaries** do not call `BuildPrompt`; only `cmd/server` needed updating for the new signature.
