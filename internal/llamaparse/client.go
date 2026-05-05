package llamaparse

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const defaultBaseURL = "https://api.cloud.llamaindex.ai"

// Page is one page of extracted text (plain or markdown normalized for downstream chunking).
type Page struct {
	Number int
	Text   string
}

// ExtractPages uploads a PDF, runs a LlamaParse job, and returns per-page text.
// apiKey is a LlamaCloud API key (LLAMA_CLOUD_API_KEY). tier is e.g. cost_effective, agentic.
func ExtractPages(ctx context.Context, apiKey, pdfPath, tier string) ([]Page, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, fmt.Errorf("llamaparse: empty API key")
	}
	if tier == "" {
		tier = "cost_effective"
	}

	base := strings.TrimRight(os.Getenv("LLAMA_CLOUD_BASE_URL"), "/")
	if base == "" {
		base = defaultBaseURL
	}

	client := &http.Client{Timeout: 120 * time.Second}

	fileID, err := uploadFile(ctx, client, base, apiKey, pdfPath)
	if err != nil {
		return nil, fmt.Errorf("llamaparse upload: %w", err)
	}

	jobID, err := startParseJob(ctx, client, base, apiKey, fileID, tier)
	if err != nil {
		return nil, fmt.Errorf("llamaparse start job: %w", err)
	}

	pages, err := pollParseJob(ctx, base, apiKey, jobID)
	if err != nil {
		return nil, fmt.Errorf("llamaparse poll: %w", err)
	}
	if len(pages) == 0 {
		return nil, fmt.Errorf("llamaparse: no pages in result")
	}
	return pages, nil
}

func uploadFile(ctx context.Context, client *http.Client, base, apiKey, pdfPath string) (string, error) {
	f, err := os.Open(pdfPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	if err := mw.WriteField("purpose", "parse"); err != nil {
		return "", err
	}
	// LlamaCloud v1 files API expects multipart field "upload_file" (not "file").
	part, err := mw.CreateFormFile("upload_file", filepath.Base(pdfPath))
	if err != nil {
		return "", err
	}
	if _, err := io.Copy(part, f); err != nil {
		return "", err
	}
	if err := mw.Close(); err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/v1/files/", &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", mw.FormDataContentType())

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(body), 500))
	}

	var out struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return "", fmt.Errorf("decode upload response: %w", err)
	}
	if out.ID == "" {
		return "", fmt.Errorf("upload: missing file id in response: %s", truncate(string(body), 300))
	}
	return out.ID, nil
}

func startParseJob(ctx context.Context, client *http.Client, base, apiKey, fileID, tier string) (string, error) {
	payload := map[string]string{
		"file_id": fileID,
		"tier":    tier,
		"version": "latest",
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/api/v2/parse", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("status %d: %s", resp.StatusCode, truncate(string(body), 500))
	}

	var top struct {
		ID     string          `json:"id"`
		Status string          `json:"status"`
		Job    json.RawMessage `json:"job"`
	}
	if err := json.Unmarshal(body, &top); err != nil {
		return "", fmt.Errorf("decode parse start: %w", err)
	}
	if top.ID != "" {
		return top.ID, nil
	}
	var nested struct {
		ID string `json:"id"`
	}
	if len(top.Job) > 0 && json.Unmarshal(top.Job, &nested) == nil && nested.ID != "" {
		return nested.ID, nil
	}
	return "", fmt.Errorf("parse start: no job id in response: %s", truncate(string(body), 400))
}

func pollParseJob(ctx context.Context, base, apiKey, jobID string) ([]Page, error) {
	parseCtx, cancel := context.WithTimeout(ctx, 25*time.Minute)
	defer cancel()

	// Long-running parse: shorter per-request timeout; we poll repeatedly.
	pollClient := &http.Client{Timeout: 60 * time.Second}
	ticker := time.NewTicker(1500 * time.Millisecond)
	defer ticker.Stop()

	for {
		u := fmt.Sprintf("%s/api/v2/parse/%s?expand=markdown", base, jobID)
		req, err := http.NewRequestWithContext(parseCtx, http.MethodGet, u, nil)
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+apiKey)
		req.Header.Set("Accept", "application/json")

		resp, err := pollClient.Do(req)
		if err != nil {
			return nil, err
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			return nil, fmt.Errorf("poll status %d: %s", resp.StatusCode, truncate(string(body), 400))
		}

		var parsed parsePollResponse
		if err := json.Unmarshal(body, &parsed); err != nil {
			return nil, fmt.Errorf("decode poll: %w", err)
		}
		st := strings.ToUpper(parsed.status())
		switch st {
		case "FAILED", "ERROR":
			return nil, fmt.Errorf("llamaparse job failed: %s", truncate(string(body), 600))
		case "COMPLETED", "COMPLETE", "SUCCESS":
			pages, err := parsed.toPages()
			if err != nil {
				return nil, err
			}
			return pages, nil
		default:
			// PENDING, RUNNING, etc.
			select {
			case <-parseCtx.Done():
				return nil, parseCtx.Err()
			case <-ticker.C:
			}
		}
	}
}

type parsePollResponse struct {
	ID       string   `json:"id"`
	Status   string   `json:"status"`
	Job      *jobWrap `json:"job"`
	Markdown *struct {
		Pages []struct {
			PageNumber int    `json:"page_number"`
			Markdown   string `json:"markdown"`
			Text       string `json:"text"`
		} `json:"pages"`
	} `json:"markdown"`
	Text *struct {
		Pages []struct {
			PageNumber int    `json:"page_number"`
			Text       string `json:"text"`
		} `json:"pages"`
	} `json:"text"`
}

type jobWrap struct {
	ID     string `json:"id"`
	Status string `json:"status"`
}

func (p parsePollResponse) status() string {
	if p.Job != nil && p.Job.Status != "" {
		return p.Job.Status
	}
	return p.Status
}

func (p parsePollResponse) toPages() ([]Page, error) {
	if p.Markdown != nil && len(p.Markdown.Pages) > 0 {
		out := make([]Page, 0, len(p.Markdown.Pages))
		for _, pg := range p.Markdown.Pages {
			t := strings.TrimSpace(pg.Markdown)
			if t == "" {
				continue
			}
			out = append(out, Page{Number: pg.PageNumber, Text: normalizeWhitespace(t)})
		}
		if len(out) > 0 {
			return out, nil
		}
	}
	if p.Text != nil && len(p.Text.Pages) > 0 {
		out := make([]Page, 0, len(p.Text.Pages))
		for _, pg := range p.Text.Pages {
			t := strings.TrimSpace(pg.Text)
			if t == "" {
				continue
			}
			out = append(out, Page{Number: pg.PageNumber, Text: normalizeWhitespace(t)})
		}
		if len(out) > 0 {
			return out, nil
		}
	}
	return nil, fmt.Errorf("completed job has no markdown/text pages")
}

func normalizeWhitespace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
