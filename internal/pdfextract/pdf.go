package pdfextract

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/ledongthuc/pdf"
	rscpdf "rsc.io/pdf"
)

type PageText struct {
	Page int
	Text string
}

func ExtractByPage(path string) ([]PageText, error) {
	pages, err := extractWithLedongthuc(path)
	if err == nil && len(pages) > 0 {
		return pages, nil
	}

	// Fallback parser for PDFs that fail on ledongthuc stream handling.
	fallbackPages, fallbackErr := extractWithRSC(path)
	if fallbackErr == nil && len(fallbackPages) > 0 {
		return fallbackPages, nil
	}

	pythonPages, pythonErr := extractWithPython(path)
	if pythonErr == nil && len(pythonPages) > 0 {
		return pythonPages, nil
	}

	if err != nil && fallbackErr != nil && pythonErr != nil {
		return nil, fmt.Errorf("primary parser failed: %v; fallback parser failed: %v; python parser failed: %v", err, fallbackErr, pythonErr)
	}
	return nil, fmt.Errorf("could not extract text from pdf")
}

func extractWithLedongthuc(path string) ([]PageText, error) {
	f, reader, err := pdf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open pdf: %w", err)
	}
	defer f.Close()

	total := reader.NumPage()
	out := make([]PageText, 0, total)

	for i := 1; i <= total; i++ {
		page := reader.Page(i)
		if page.V.IsNull() {
			continue
		}

		text, err := page.GetPlainText(nil)
		if err != nil {
			return nil, fmt.Errorf("extract text for page %d: %w", i, err)
		}

		trimmed := strings.TrimSpace(text)
		if trimmed == "" {
			continue
		}

		out = append(out, PageText{
			Page: i,
			Text: trimmed,
		})
	}

	return out, nil
}

func extractWithRSC(path string) ([]PageText, error) {
	reader, err := rscpdf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open pdf: %w", err)
	}

	total := reader.NumPage()
	out := make([]PageText, 0, total)
	for i := 1; i <= total; i++ {
		p := reader.Page(i)
		if p.V.IsNull() {
			continue
		}

		content, err := safePageContent(p)
		if err != nil {
			continue
		}
		var b strings.Builder
		for _, t := range content.Text {
			if strings.TrimSpace(t.S) == "" {
				continue
			}
			b.WriteString(t.S)
			b.WriteByte(' ')
		}

		text := strings.TrimSpace(b.String())
		if text == "" {
			continue
		}

		out = append(out, PageText{
			Page: i,
			Text: text,
		})
	}

	if len(out) == 0 {
		return nil, fmt.Errorf("no extractable text from rsc parser")
	}

	return out, nil
}

func safePageContent(p rscpdf.Page) (content rscpdf.Content, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic while reading page content: %v", r)
		}
	}()
	content = p.Content()
	return content, nil
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
