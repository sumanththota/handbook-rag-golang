//go:build characterization

package rag

// Live characterization test for the Go -> Python port (docs/MIGRATION.md §6).
//
// This is NOT a unit test: it hits real Ollama + Qdrant and ingests the real
// handbook PDF, because retrieval ranking and the assembled prompt depend on
// the embedding model's actual output — there is nothing meaningful to fake
// here. It snapshots CURRENT behavior as ground truth; it does not judge
// whether that behavior is good (see docs/report.md for known issues).
//
// Run it explicitly, with services up and HANDBOOK_PATH set:
//
//   go test -tags characterization ./internal/rag/... -run Characterization -v
//
// First run with no testdata/characterization_golden.json present RECORDS a
// new fixture and passes — review the diff and commit it. Every run after
// that COMPARES against the committed fixture and fails on drift. When the
// Python port runs the equivalent flow, its output must match this same
// fixture (adapted for Python's client, not this file).
//
// Uses a dedicated collection ("handbook_chunks_characterization"), never the
// collection your dev server/UI points at, so running this can't clobber (or
// be clobbered by) manual testing.

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"handbook-rag/internal/embed"
	"handbook-rag/internal/qdrant"
)

const characterizationCollection = "handbook_chunks_characterization"

// characterizationQuestions are generic grad-handbook topics chosen to likely
// hit real content regardless of the specific handbook loaded. Tune this list
// for your actual handbook content, then re-record the fixture deliberately
// (delete testdata/characterization_golden.json and re-run) — don't hand-edit
// the golden file.
var characterizationQuestions = []string{
	"What is the minimum GPA required to remain in good academic standing?",
	"How many credit hours are required to graduate?",
	"What is the policy on academic probation or dismissal?",
}

type questionResult struct {
	Question   string         `json:"question"`
	PromptHash string         `json:"prompt_hash_md5"`
	Results    []resultGolden `json:"results"`
}

type resultGolden struct {
	Page  int     `json:"page"`
	Score float64 `json:"score"`
}

type characterizationFixture struct {
	ChunkCount int              `json:"chunk_count"`
	Questions  []questionResult `json:"questions"`
}

func TestCharacterization_IngestAndRetrieval(t *testing.T) {
	handbookPath := os.Getenv("HANDBOOK_PATH")
	if handbookPath == "" {
		t.Skip("HANDBOOK_PATH not set; skipping live characterization test")
	}

	ollamaHost := envOrDefault("OLLAMA_HOST", "http://localhost:11434")
	qdrantHost := envOrDefault("QDRANT_HOST", "http://localhost:6333")

	embedClient := embed.NewOllamaClient(ollamaHost)
	qdrantClient := qdrant.New(qdrantHost)

	healthCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := embedClient.Healthy(healthCtx); err != nil {
		t.Skipf("Ollama not reachable at %s: %v", ollamaHost, err)
	}
	if err := qdrantClient.Healthy(healthCtx); err != nil {
		t.Skipf("Qdrant not reachable at %s: %v", qdrantHost, err)
	}

	svc := NewServiceWithParams(characterizationCollection, 10, 180, handbookPath, embedClient, qdrantClient)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	chunkCount, err := svc.Ingest(ctx)
	if err != nil {
		t.Fatalf("ingest failed: %v", err)
	}

	actual := characterizationFixture{ChunkCount: chunkCount}
	for _, q := range characterizationQuestions {
		prompt, results, err := svc.BuildPrompt(ctx, q)
		if err != nil {
			t.Fatalf("BuildPrompt(%q) failed: %v", q, err)
		}

		qr := questionResult{
			Question:   q,
			PromptHash: md5Hex(prompt),
		}
		for _, r := range results {
			qr.Results = append(qr.Results, resultGolden{
				Page:  r.Page,
				Score: round6(r.Score),
			})
		}
		actual.Questions = append(actual.Questions, qr)
	}

	goldenPath := filepath.Join("testdata", "characterization_golden.json")
	if _, err := os.Stat(goldenPath); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatalf("create testdata dir: %v", err)
		}
		out, err := json.MarshalIndent(actual, "", "  ")
		if err != nil {
			t.Fatalf("marshal golden fixture: %v", err)
		}
		if err := os.WriteFile(goldenPath, out, 0o644); err != nil {
			t.Fatalf("write golden fixture: %v", err)
		}
		t.Logf("recorded new golden fixture at %s — review it and commit it", goldenPath)
		return
	}

	raw, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("read golden fixture: %v", err)
	}
	var expected characterizationFixture
	if err := json.Unmarshal(raw, &expected); err != nil {
		t.Fatalf("unmarshal golden fixture: %v", err)
	}

	if !reflect.DeepEqual(expected, actual) {
		expectedJSON, _ := json.MarshalIndent(expected, "", "  ")
		actualJSON, _ := json.MarshalIndent(actual, "", "  ")
		t.Fatalf("characterization drift detected.\n--- golden (%s) ---\n%s\n--- actual ---\n%s",
			goldenPath, expectedJSON, actualJSON)
	}
}

func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func round6(f float64) float64 {
	const scale = 1e6
	return float64(int64(f*scale+0.5)) / scale
}

func md5Hex(s string) string {
	sum := md5.Sum([]byte(s))
	return hex.EncodeToString(sum[:])
}
