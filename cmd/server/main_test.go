package main

import (
	"encoding/base64"
	"encoding/json"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"handbook-rag/internal/embed"
	"handbook-rag/internal/llm"
	"handbook-rag/internal/qdrant"
	"handbook-rag/internal/rag"
)

// These tests lock in the CURRENT HTTP/SSE contract that the HTMX frontend
// depends on (see docs/ui-execution-tracker.md). They deliberately avoid
// requiring live Ollama/Qdrant/LLM-provider processes by exercising only the
// deterministic paths (validation, templating, and the dependency-unavailable
// error contract) — the success-path retrieval/streaming behavior is covered
// separately by internal/rag's build-tagged live characterization test.

// unreachableService builds a rag.Service pointed at a closed local port so
// any network call fails immediately (connection refused) instead of
// hanging on the client's real timeout.
func unreachableService() *rag.Service {
	embedClient := embed.NewOllamaClient("http://127.0.0.1:1")
	qdrantClient := qdrant.New("http://127.0.0.1:1")
	return rag.NewService("handbook_chunks", 10, "unused.pdf", embedClient, qdrantClient)
}

func TestHandleHealthLive(t *testing.T) {
	req := httptest.NewRequest("GET", "/health/live", nil)
	rec := httptest.NewRecorder()
	handleHealthLive(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var got healthResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Status != "ok" {
		t.Fatalf("status field = %q, want %q", got.Status, "ok")
	}
}

func TestHandleChatStart_ValidRequest(t *testing.T) {
	form := url.Values{}
	form.Set("question", "What is <b>bold</b>?")
	form.Set("model_id", "groq_llama31_8b")

	req := httptest.NewRequest("POST", "/chat/start", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	handleChatStart(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	body := rec.Body.String()

	// The question must be HTML-escaped in the rendered user bubble.
	if !strings.Contains(body, "What is &lt;b&gt;bold&lt;/b&gt;?") {
		t.Fatalf("expected escaped question in body, got: %s", body)
	}
	// The assistant placeholder + streaming trigger must be present with the
	// expected element id and query-escaped script args — the frontend JS
	// (window.startAnswerStream) depends on this exact shape.
	if !strings.Contains(body, `id="assistant-last"`) {
		t.Fatalf("expected assistant-last placeholder, got: %s", body)
	}
	if !strings.Contains(body, "window.startAnswerStream(") {
		t.Fatalf("expected startAnswerStream call, got: %s", body)
	}
}

func TestHandleChatStart_MissingQuestion(t *testing.T) {
	form := url.Values{}
	form.Set("model_id", "groq_llama31_8b")
	req := httptest.NewRequest("POST", "/chat/start", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	handleChatStart(rec, req)

	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

func TestHandleChatStart_InvalidModelID(t *testing.T) {
	form := url.Values{}
	form.Set("question", "hi")
	form.Set("model_id", "not_a_real_model")
	req := httptest.NewRequest("POST", "/chat/start", strings.NewReader(form.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	rec := httptest.NewRecorder()
	handleChatStart(rec, req)

	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

func TestHandleChatStart_WrongMethod(t *testing.T) {
	req := httptest.NewRequest("GET", "/chat/start", nil)
	rec := httptest.NewRecorder()
	handleChatStart(rec, req)

	if rec.Code != 405 {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

func TestHandleIngest_DependencyUnavailable(t *testing.T) {
	svc := unreachableService()
	req := httptest.NewRequest("POST", "/ingest", nil)
	rec := httptest.NewRecorder()
	handleIngest(svc)(rec, req)

	// Current behavior: the handler does NOT set a non-200 status on failure
	// — it renders an HTMX-swappable error fragment with implicit 200. This
	// is a real characteristic of the contract, not an oversight to "fix"
	// silently during the port.
	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200 (error is rendered in-body, not via status code)", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `<p class="error">Ingestion failed:`) {
		t.Fatalf("expected error fragment, got: %s", body)
	}
	if !strings.Contains(body, "ensure qdrant collection") {
		t.Fatalf("expected error to be wrapped with 'ensure qdrant collection' context, got: %s", body)
	}
}

func TestHandleChatStream_MissingQuestion(t *testing.T) {
	req := httptest.NewRequest("GET", "/chat/stream?model_id=groq_llama31_8b", nil)
	rec := httptest.NewRecorder()
	handleChatStream(unreachableService(), nil, nil)(rec, req)

	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

func TestHandleChatStream_InvalidModelID(t *testing.T) {
	req := httptest.NewRequest("GET", "/chat/stream?question=hi&model_id=nope", nil)
	rec := httptest.NewRecorder()
	handleChatStream(unreachableService(), nil, nil)(rec, req)

	if rec.Code != 400 {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

func TestHandleChatStream_DependencyUnavailableEmitsStreamError(t *testing.T) {
	req := httptest.NewRequest("GET", "/chat/stream?question=hi&model_id=groq_llama31_8b", nil)
	rec := httptest.NewRecorder()

	// A configured provider client + API key must be present for the handler
	// to reach BuildPrompt at all (it 500s earlier otherwise) — but retrieval
	// fails before the provider client is ever called (embed step fails
	// first), so the client itself just needs to exist, not succeed.
	providerClients := map[string]*llm.OpenAICompatibleClient{
		"groq": llm.NewOpenAICompatibleClient("http://127.0.0.1:1", nil),
	}
	apiKeys := map[string]string{"GROQ_API_KEY": "test-key"}
	handleChatStream(unreachableService(), providerClients, apiKeys)(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200 (SSE responses stay 200; errors are events)", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	body := rec.Body.String()
	if !strings.Contains(body, "event: streamerror") {
		t.Fatalf("expected a streamerror SSE event, got: %s", body)
	}

	payload := extractSSEData(t, body, "streamerror")
	raw, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		t.Fatalf("streamerror payload is not valid base64: %v", err)
	}
	var decoded streamErrorPayload
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("streamerror payload is not valid JSON: %v", err)
	}
	if decoded.Code != "dependency_unavailable" {
		t.Fatalf("code = %q, want %q", decoded.Code, "dependency_unavailable")
	}
	if decoded.Message != "Search is temporarily unavailable because the retrieval service is offline. Please try again shortly." {
		t.Fatalf("unexpected user message: %q", decoded.Message)
	}
}

func extractSSEData(t *testing.T, body, event string) string {
	t.Helper()
	marker := "event: " + event + "\ndata: "
	idx := strings.Index(body, marker)
	if idx == -1 {
		t.Fatalf("event %q not found in SSE body: %s", event, body)
	}
	rest := body[idx+len(marker):]
	end := strings.Index(rest, "\n")
	if end == -1 {
		end = len(rest)
	}
	return rest[:end]
}
