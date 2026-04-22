package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type OpenAICompatibleClient struct {
	baseURL string
	headers map[string]string
	client  *http.Client
}

func NewOpenAICompatibleClient(baseURL string, headers map[string]string) *OpenAICompatibleClient {
	return &OpenAICompatibleClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		headers: headers,
		client: &http.Client{
			Timeout: 0,
		},
	}
}

func (c *OpenAICompatibleClient) StreamAnswer(ctx context.Context, apiKey, model, prompt string, onToken func(string) error) error {
	payload := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": `You are a helpful assistant for the University Student Handbook.
Answer questions using only the provided handbook context in markdown format.
If the answer is not in the context, respond with: "I couldn't find that in the handbook. Please contact the relevant university office."
Never fabricate policies, dates, or procedures.`,
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"stream": true,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("provider returned status %d", resp.StatusCode)
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return nil
		}

		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		token := chunk.Choices[0].Delta.Content
		if token == "" {
			continue
		}
		if err := onToken(token); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read stream: %w", err)
	}
	return nil
}
