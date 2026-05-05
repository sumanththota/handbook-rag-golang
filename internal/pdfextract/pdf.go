package pdfextract

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"handbook-rag/internal/llamaparse"
)

type PageText struct {
	Page int
	Text string
}

// ExtractByPage reads the PDF at path. If LLAMA_CLOUD_API_KEY is set, uses LlamaCloud LlamaParse;
// otherwise falls back to local Python (pypdf / PyPDF2).
func ExtractByPage(ctx context.Context, path string) ([]PageText, error) {
	log.Printf("[pdfextract] start path=%s", path)

	key := strings.TrimSpace(os.Getenv("LLAMA_CLOUD_API_KEY"))
	if key != "" {
		tier := strings.TrimSpace(os.Getenv("LLAMAPARSE_TIER"))
		if tier == "" {
			tier = "cost_effective"
		}
		pages, err := llamaparse.ExtractPages(ctx, key, path, tier)
		if err != nil {
			log.Printf("[pdfextract] parser=llamaparse failed err=%v", err)
			return nil, err
		}
		out := make([]PageText, 0, len(pages))
		for _, p := range pages {
			t := strings.TrimSpace(p.Text)
			if t == "" {
				continue
			}
			n := p.Number
			if n <= 0 {
				n = len(out) + 1
			}
			out = append(out, PageText{Page: n, Text: t})
		}
		if len(out) == 0 {
			return nil, fmt.Errorf("llamaparse: no non-empty pages")
		}
		log.Printf("[pdfextract] parser=llamaparse pages=%d tier=%s", len(out), tier)
		return out, nil
	}

	pages, err := extractWithPython(path)
	if err != nil {
		log.Printf("[pdfextract] parser=python failed err=%v", err)
		return nil, fmt.Errorf("extract text with python parser: %w", err)
	}
	log.Printf("[pdfextract] parser=python pages=%d", len(pages))
	return pages, nil
}

func extractWithPython(path string) ([]PageText, error) {
	script := `
import json, sys
pdf_path = sys.argv[1]
reader = None
err = None
try:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
except Exception as e1:
    err = e1
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
    except Exception as e2:
        raise Exception(f"cannot import pypdf/PyPDF2: {e1}; {e2}")

pages = []
for i, p in enumerate(reader.pages, start=1):
    text = p.extract_text() or ""
    text = " ".join(text.split())
    if text:
        pages.append({"page": i, "text": text})
print(json.dumps(pages))
`

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "python3", "-c", script, path)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("python3 execution failed: %w", err)
	}

	var parsed []struct {
		Page int    `json:"page"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		return nil, fmt.Errorf("parse python output: %w", err)
	}

	res := make([]PageText, 0, len(parsed))
	for _, p := range parsed {
		if strings.TrimSpace(p.Text) == "" {
			continue
		}
		res = append(res, PageText{
			Page: p.Page,
			Text: p.Text,
		})
	}

	if len(res) == 0 {
		return nil, fmt.Errorf("no extractable text from python parser")
	}
	return res, nil
}
