package main

import (
	_ "embed"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"log"
	"math/rand"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"handbook-rag/internal/config"
	"handbook-rag/internal/embed"
	"handbook-rag/internal/llm"
	"handbook-rag/internal/qdrant"
	"handbook-rag/internal/rag"
)

type ModelConfig struct {
	Provider string
	Model    string
	EnvKey   string
}

var modelConfigs = map[string]ModelConfig{
	"openrouter_nemotron": {
		Provider: "openrouter",
		Model:    "nvidia/nemotron-3-nano-30b-a3b:free",
		EnvKey:   "OPENROUTER_API_KEY",
	},
	"openrouter_llama4_scout": {
		Provider: "openrouter",
		Model:    "meta-llama/llama-4-scout-17b-16e-instruct",
		EnvKey:   "OPENROUTER_API_KEY",
	},
	"groq_llama31_8b": {
		Provider: "groq",
		Model:    "llama-3.1-8b-instant",
		EnvKey:   "GROQ_API_KEY",
	},
}

func main() {
	rand.Seed(time.Now().UnixNano())

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}
	log.Printf("[boot] config loaded port=%s ollama_host=%s qdrant_host=%s collection=%s top_k=%d handbook=%s openrouter_key=%t groq_key=%t llamaparse=%t", cfg.Port, cfg.OllamaHost, cfg.QdrantHost, cfg.CollectionName, cfg.TopK, cfg.HandbookPath, cfg.OpenRouterAPIKey != "", cfg.GroqAPIKey != "", os.Getenv("LLAMA_CLOUD_API_KEY") != "")

	embedClient := embed.NewOllamaClient(cfg.OllamaHost)
	qdrantClient := qdrant.New(cfg.QdrantHost)
	ragSvc := rag.NewService(
		cfg.CollectionName,
		cfg.TopK,
		cfg.HandbookPath,
		embedClient,
		qdrantClient,
	)

	providerClients := map[string]*llm.OpenAICompatibleClient{
		"openrouter": llm.NewOpenAICompatibleClient("https://openrouter.ai/api/v1", map[string]string{
			"HTTP-Referer": "http://localhost",
			"X-Title":      "handbook-rag",
		}),
		"groq": llm.NewOpenAICompatibleClient("https://api.groq.com/openai/v1", nil),
	}
	apiKeys := map[string]string{
		"OPENROUTER_API_KEY": cfg.OpenRouterAPIKey,
		"GROQ_API_KEY":       cfg.GroqAPIKey,
	}

	if cfg.OpenRouterAPIKey != "" {
		ragSvc.SetQueryRewriter(providerClients["openrouter"], cfg.OpenRouterAPIKey, "openai/gpt-4o-mini")
		log.Printf("[boot] query rewrite enabled provider=openrouter model=openai/gpt-4o-mini")
	} else {
		log.Printf("[boot] query rewrite disabled reason=missing OPENROUTER_API_KEY")
	}

	experimentsPath := "docs/experiments.jsonl"

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/ingest", handleIngest(ragSvc))
	mux.HandleFunc("/chat/start", handleChatStart)
	mux.HandleFunc("/chat/stream", handleChatStream(ragSvc, providerClients, apiKeys))
	mux.HandleFunc("/health/live", handleHealthLive)
	mux.HandleFunc("/health/ready", handleHealthReady(embedClient, qdrantClient, cfg))
	mux.HandleFunc("/dashboard", handleDashboard())
	mux.HandleFunc("/dashboard/data", handleDashboardData(experimentsPath))

	addr := ":" + cfg.Port
	log.Printf("server listening on %s", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

type healthResponse struct {
	Status       string            `json:"status"`
	Dependencies map[string]string `json:"dependencies,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func handleHealthLive(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, healthResponse{Status: "ok"})
}

func handleHealthReady(embedClient *embed.OllamaClient, qdrantClient *qdrant.Client, cfg config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		deps := map[string]string{
			"ollama": "ok",
			"qdrant": "ok",
		}
		overall := "ok"

		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		if err := embedClient.Healthy(ctx); err != nil {
			deps["ollama"] = "unavailable"
			overall = "degraded"
		}
		if err := qdrantClient.Healthy(ctx); err != nil {
			deps["qdrant"] = "unavailable"
			overall = "degraded"
		}
		if strings.TrimSpace(cfg.OpenRouterAPIKey) == "" && strings.TrimSpace(cfg.GroqAPIKey) == "" {
			deps["llm_provider"] = "unconfigured"
			overall = "degraded"
		} else {
			deps["llm_provider"] = "configured"
		}

		statusCode := http.StatusOK
		if overall != "ok" {
			statusCode = http.StatusServiceUnavailable
		}
		writeJSON(w, statusCode, healthResponse{
			Status:       overall,
			Dependencies: deps,
		})
	}
}

//go:embed index.html
var indexHTML string

func handleIndex(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, indexHTML)
}

func handleIngest(svc *rag.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		reqID := newReqID()
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		log.Printf("[http][%s] ingest start", reqID)

		count, err := svc.Ingest(r.Context())
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		if err != nil {
			log.Printf("[http][%s] ingest failed: %v", reqID, err)
			fmt.Fprintf(w, `<p class="error">Ingestion failed: %s</p>`, html.EscapeString(err.Error()))
			return
		}

		log.Printf("[http][%s] ingest success chunks=%d", reqID, count)
		fmt.Fprintf(w, `<p class="ok">Ingestion complete. Indexed %d chunks.</p>`, count)
	}
}

func handleChatStart(w http.ResponseWriter, r *http.Request) {
	reqID := newReqID()
	startedAtMs := time.Now().UnixMilli()
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if err := r.ParseForm(); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}

	question := strings.TrimSpace(r.FormValue("question"))
	modelID := strings.TrimSpace(r.FormValue("model_id"))
	if question == "" {
		http.Error(w, "question is required", http.StatusBadRequest)
		return
	}
	if _, ok := modelConfigs[modelID]; !ok {
		http.Error(w, "invalid model selection", http.StatusBadRequest)
		return
	}
	log.Printf("[http][%s] chat start question_chars=%d model_id=%s", reqID, len(question), modelID)

	escapedQuestion := html.EscapeString(question)
	escapedQuery := url.QueryEscape(question)
	escapedModelID := url.QueryEscape(modelID)
	escapedStartedAt := url.QueryEscape(fmt.Sprintf("%d", startedAtMs))

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprintf(
		w,
		`<div class="msg user">%s</div><div class="msg assistant raw" id="assistant-last" data-streaming="1"></div><script>window.startAnswerStream("%s","%s","%s");</script>`,
		escapedQuestion,
		escapedQuery,
		escapedModelID,
		escapedStartedAt,
	)
}

func handleChatStream(svc *rag.Service, providerClients map[string]*llm.OpenAICompatibleClient, apiKeys map[string]string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		reqID := newReqID()
		streamStart := time.Now()

		question := strings.TrimSpace(r.URL.Query().Get("question"))
		modelID := strings.TrimSpace(r.URL.Query().Get("model_id"))
		startedAtRaw := strings.TrimSpace(r.URL.Query().Get("started_at_ms"))
		if question == "" {
			http.Error(w, "question is required", http.StatusBadRequest)
			return
		}
		cfg, ok := modelConfigs[modelID]
		if !ok {
			http.Error(w, "invalid model_id", http.StatusBadRequest)
			return
		}
		client := providerClients[cfg.Provider]
		if client == nil {
			http.Error(w, "provider client not configured", http.StatusInternalServerError)
			return
		}
		apiKey := strings.TrimSpace(apiKeys[cfg.EnvKey])
		if apiKey == "" {
			http.Error(w, fmt.Sprintf("missing %s for selected provider", cfg.EnvKey), http.StatusBadRequest)
			return
		}
		log.Printf("[http][%s] stream start question_chars=%d provider=%s model=%s", reqID, len(question), cfg.Provider, cfg.Model)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}

		send := func(event, data string) error {
			if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, strings.ReplaceAll(data, "\n", "\\n")); err != nil {
				return err
			}
			flusher.Flush()
			return nil
		}

		prompt, results, err := svc.BuildPrompt(r.Context(), question)
		if err != nil {
			appErr := classifyError(err)
			log.Printf("[http][%s] retrieval failed code=%s retryable=%t err=%v", reqID, appErr.Code, appErr.Retryable, err)
			_ = send("streamerror", appErr.UserMessage)
			return
		}
		if len(results) == 0 {
			emptySourcesJSON, _ := json.Marshal([]any{})
			_ = send("sources", base64.StdEncoding.EncodeToString(emptySourcesJSON))
			if prompt != "" {
				_ = send("token", prompt)
			}
			_ = send("done", "complete")
			return
		}

		type sourceRow struct {
			Page  int     `json:"page"`
			Score float64 `json:"score"`
			Text  string  `json:"text"`
		}
		srcRows := make([]sourceRow, 0, len(results))
		const maxSourceRunes = 220
		for _, r := range results {
			t := r.Text
			if n := utf8.RuneCountInString(t); n > maxSourceRunes {
				t = string([]rune(t)[:maxSourceRunes]) + "…"
			}
			srcRows = append(srcRows, sourceRow{Page: r.Page, Score: r.Score, Text: t})
		}
		srcJSON, err := json.Marshal(srcRows)
		if err != nil {
			log.Printf("[http][%s] sources marshal: %v", reqID, err)
			_ = send("streamerror", "internal: could not encode sources")
			return
		}
		if err := send("sources", base64.StdEncoding.EncodeToString(srcJSON)); err != nil {
			return
		}

		if err := client.StreamAnswer(r.Context(), apiKey, cfg.Model, prompt, func(token string) error {
			return send("token", token)
		}); err != nil {
			appErr := classifyError(err)
			log.Printf("[http][%s] stream failed code=%s retryable=%t err=%v", reqID, appErr.Code, appErr.Retryable, err)
			_ = send("streamerror", appErr.UserMessage)
			return
		}

		streamDuration := time.Since(streamStart)
		totalFromQuery := streamDuration
		if startedAtRaw != "" {
			if startedAtMs, err := strconv.ParseInt(startedAtRaw, 10, 64); err == nil && startedAtMs > 0 {
				totalFromQuery = time.Since(time.UnixMilli(startedAtMs))
			}
		}
		log.Printf("[http][%s] stream complete duration=%s total_from_query=%s", reqID, streamDuration, totalFromQuery)
		_ = send("done", "complete")
	}
}

type AppError struct {
	Code       string
	UserMessage string
	Retryable  bool
}

func classifyError(err error) AppError {
	msg := strings.ToLower(strings.TrimSpace(err.Error()))
	if errors.Is(err, context.DeadlineExceeded) || strings.Contains(msg, "deadline exceeded") || strings.Contains(msg, "timeout") {
		return AppError{
			Code:        "timeout",
			UserMessage: "Request timed out while contacting search services. Please try again.",
			Retryable:   true,
		}
	}
	if strings.Contains(msg, "11434") || strings.Contains(msg, "ollama") || strings.Contains(msg, "embed query") {
		return AppError{
			Code:        "dependency_unavailable",
			UserMessage: "Search is temporarily unavailable because the retrieval service is offline. Please try again shortly.",
			Retryable:   true,
		}
	}
	if strings.Contains(msg, "qdrant") || strings.Contains(msg, "search qdrant") {
		return AppError{
			Code:        "dependency_unavailable",
			UserMessage: "Search index is temporarily unavailable. Please try again in a moment.",
			Retryable:   true,
		}
	}
	if strings.Contains(msg, "status 429") || strings.Contains(msg, "rate limit") {
		return AppError{
			Code:        "rate_limited",
			UserMessage: "The model is receiving too many requests right now. Please retry shortly.",
			Retryable:   true,
		}
	}
	if strings.Contains(msg, "no context found") {
		return AppError{
			Code:        "no_context",
			UserMessage: "I couldn't find matching handbook content for that question. Try a more specific academic or policy query.",
			Retryable:   true,
		}
	}
	return AppError{
		Code:        "internal",
		UserMessage: "Something went wrong while generating the answer. Please try again.",
		Retryable:   true,
	}
}

func newReqID() string {
	return fmt.Sprintf("%08x", rand.Uint32())
}
