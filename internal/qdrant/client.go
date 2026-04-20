package qdrant

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	baseURL string
	client  *http.Client
}

type SearchResult struct {
	Text  string
	Page  int
	Score float64
}

func New(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		client: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *Client) EnsureCollection(ctx context.Context, name string) error {
	body := map[string]any{
		"vectors": map[string]any{
			"size":     384,
			"distance": "Cosine",
		},
	}

	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil {
		return fmt.Errorf("encode request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, c.baseURL+"/collections/"+name, &buf)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	// Collection already exists. Treat as success so ingestion is rerunnable.
	if resp.StatusCode == http.StatusConflict {
		return nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant request %s %s failed with status %d", http.MethodPut, "/collections/"+name, resp.StatusCode)
	}
	return nil
}

func (c *Client) Upsert(ctx context.Context, collection string, id int, vector []float64, text string, page int) error {
	body := map[string]any{
		"points": []map[string]any{
			{
				"id":     id,
				"vector": vector,
				"payload": map[string]any{
					"text": text,
					"page": page,
				},
			},
		},
	}

	return c.request(ctx, http.MethodPut, "/collections/"+collection+"/points?wait=true", body, nil)
}

func (c *Client) Search(ctx context.Context, collection string, vector []float64, limit int) ([]SearchResult, error) {
	body := map[string]any{
		"vector":       vector,
		"limit":        limit,
		"with_payload": true,
	}

	var response struct {
		Result []struct {
			Score   float64 `json:"score"`
			Payload struct {
				Text string `json:"text"`
				Page int    `json:"page"`
			} `json:"payload"`
		} `json:"result"`
	}

	if err := c.request(ctx, http.MethodPost, "/collections/"+collection+"/points/search", body, &response); err != nil {
		return nil, err
	}

	out := make([]SearchResult, 0, len(response.Result))
	for _, item := range response.Result {
		out = append(out, SearchResult{
			Text:  item.Payload.Text,
			Page:  item.Payload.Page,
			Score: item.Score,
		})
	}
	return out, nil
}

func (c *Client) request(ctx context.Context, method, path string, body any, out any) error {
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			return fmt.Errorf("encode request body: %w", err)
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, &buf)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("qdrant request %s %s failed with status %d", method, path, resp.StatusCode)
	}

	if out != nil {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}
