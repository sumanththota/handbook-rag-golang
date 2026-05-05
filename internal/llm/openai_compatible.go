package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"`
	Stream      bool          `json:"stream"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

type OpenAICompatibleClient struct {
	baseURL string
	headers map[string]string
	client  *http.Client
}

func NewOpenAICompatibleClient(baseURL string, headers map[string]string) *OpenAICompatibleClient {
	return &OpenAICompatibleClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		headers: headers,
		client:  &http.Client{},
	}
}

func (c *OpenAICompatibleClient) sendChatRequest(ctx context.Context, apiKey string, payload chatCompletionRequest) (*http.Response, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if key := strings.TrimSpace(apiKey); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.headers {
		req.Header.Set(k, v)
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if len(body) > 0 {
			return nil, fmt.Errorf("provider returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
		return nil, fmt.Errorf("provider returned status %d", resp.StatusCode)
	}
	return resp, nil
}

func (c *OpenAICompatibleClient) Complete(ctx context.Context, apiKey, model string, temperature float64, messages []ChatMessage) (string, error) {
	payload := chatCompletionRequest{
		Model:       model,
		Messages:    messages,
		Temperature: &temperature,
		Stream:      false,
	}

	resp, err := c.sendChatRequest(ctx, apiKey, payload)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var out chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("empty choices in completion response")
	}
	return strings.TrimSpace(out.Choices[0].Message.Content), nil
}

func (c *OpenAICompatibleClient) StreamAnswer(ctx context.Context, apiKey, model, prompt string, onToken func(string) error) error {
	payload := chatCompletionRequest{
		Model: model,
		Messages: []ChatMessage{
			{
				Role: "system",
				Content: `You are a helpful assistant for the University Student Handbook.
Answer questions using only the provided handbook context in markdown format.
If the answer is not in the context, respond with: "I couldn't find that in the handbook. Please contact the relevant university office."
Never fabricate policies, dates, or procedures.`,
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		Stream: true,
	}
	resp, err := c.sendChatRequest(ctx, apiKey, payload)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

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
