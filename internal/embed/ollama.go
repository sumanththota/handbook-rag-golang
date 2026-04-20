package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type OllamaClient struct {
	baseURL string
	client  *http.Client
}

func NewOllamaClient(baseURL string) *OllamaClient {
	return &OllamaClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *OllamaClient) Embed(ctx context.Context, model, text string) ([]float64, error) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil, fmt.Errorf("empty text provided for embedding")
	}

	// BGE-small models in Ollama can reject long prompts due to small context.
	// Retry with progressively shorter text so ingestion can continue.
	current := trimmed
	for attempt := 0; attempt < 4; attempt++ {
		vec, retry, err := c.embedOnce(ctx, model, current)
		if err == nil {
			return vec, nil
		}
		if !retry {
			return nil, err
		}

		current = shrinkText(current)
		if current == "" {
			return nil, err
		}
	}

	return nil, fmt.Errorf("embedding failed after retries due to context-length constraints")
}

func (c *OllamaClient) embedOnce(ctx context.Context, model, text string) ([]float64, bool, error) {
	reqBody := map[string]string{
		"model":  model,
		"prompt": text,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, false, fmt.Errorf("marshal embed request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("create embed request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, false, fmt.Errorf("send embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		bodyText := string(bodyBytes)
		if resp.StatusCode == http.StatusInternalServerError &&
			strings.Contains(strings.ToLower(bodyText), "context length") {
			return nil, true, fmt.Errorf("ollama embed input too long: %s", bodyText)
		}
		return nil, false, fmt.Errorf("ollama embed failed with status %d: %s", resp.StatusCode, strings.TrimSpace(bodyText))
	}

	var payload struct {
		Embedding []float64 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, false, fmt.Errorf("decode embed response: %w", err)
	}

	if len(payload.Embedding) == 0 {
		return nil, false, fmt.Errorf("empty embedding returned by ollama")
	}

	return payload.Embedding, false, nil
}

func shrinkText(in string) string {
	words := strings.Fields(in)
	if len(words) <= 40 {
		return ""
	}
	newLen := len(words) / 2
	if newLen < 40 {
		newLen = 40
	}
	return strings.Join(words[:newLen], " ")
}
