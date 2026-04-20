package rag

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"handbook-rag/internal/chunk"
	"handbook-rag/internal/embed"
	"handbook-rag/internal/pdfextract"
	"handbook-rag/internal/qdrant"
)

type Service struct {
	collection  string
	topK        int
	embedModel  string
	chunkWords  int
	pdfPath     string
	embedClient *embed.OllamaClient
	qdrant      *qdrant.Client
}

func NewService(
	collection string,
	topK int,
	pdfPath string,
	embedClient *embed.OllamaClient,
	qdrantClient *qdrant.Client,
) *Service {
	return &Service{
		collection:  collection,
		topK:        topK,
		embedModel:  "qllama/bge-small-en-v1.5",
		chunkWords:  180,
		pdfPath:     pdfPath,
		embedClient: embedClient,
		qdrant:      qdrantClient,
	}
}

func NewServiceWithParams(
	collection string,
	topK int,
	chunkWords int,
	pdfPath string,
	embedClient *embed.OllamaClient,
	qdrantClient *qdrant.Client,
) *Service {
	return &Service{
		collection:  collection,
		topK:        topK,
		embedModel:  "qllama/bge-small-en-v1.5",
		chunkWords:  chunkWords,
		pdfPath:     pdfPath,
		embedClient: embedClient,
		qdrant:      qdrantClient,
	}
}

// Retrieve embeds the question and returns the raw top-K results with scores.
func (s *Service) Retrieve(ctx context.Context, question string) ([]qdrant.SearchResult, error) {
	vec, err := s.embedClient.Embed(ctx, s.embedModel, question)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	return s.qdrant.Search(ctx, s.collection, vec, s.topK)
}

func (s *Service) Ingest(ctx context.Context) (int, error) {
	start := time.Now()
	log.Printf("[rag][ingest] start collection=%s pdf=%s embed_model=%s", s.collection, s.pdfPath, s.embedModel)

	if err := s.qdrant.EnsureCollection(ctx, s.collection); err != nil {
		return 0, fmt.Errorf("ensure qdrant collection: %w", err)
	}
	log.Printf("[rag][ingest] qdrant collection ready: %s", s.collection)

	pages, err := pdfextract.ExtractByPage(s.pdfPath)
	if err != nil {
		return 0, fmt.Errorf("extract pdf: %w", err)
	}
	log.Printf("[rag][ingest] extracted pages=%d", len(pages))

	inputs := make([]chunk.Input, 0, len(pages))
	for _, p := range pages {
		inputs = append(inputs, chunk.Input{
			Page: p.Page,
			Text: p.Text,
		})
	}

	chunks := chunk.Build(inputs, s.chunkWords)
	log.Printf("[rag][ingest] built chunks=%d words_per_chunk=%d", len(chunks), s.chunkWords)
	for _, c := range chunks {
		vec, err := s.embedClient.Embed(ctx, s.embedModel, c.Text)
		if err != nil {
			return 0, fmt.Errorf("embed chunk %d: %w", c.ID, err)
		}
		if err := s.qdrant.Upsert(ctx, s.collection, c.ID, vec, c.Text, c.Page); err != nil {
			return 0, fmt.Errorf("upsert chunk %d: %w", c.ID, err)
		}
		if c.ID > 0 && c.ID%25 == 0 {
			log.Printf("[rag][ingest] progress upserted_chunks=%d/%d", c.ID+1, len(chunks))
		}
	}

	log.Printf("[rag][ingest] complete chunks=%d duration=%s", len(chunks), time.Since(start))
	return len(chunks), nil
}

func (s *Service) BuildPrompt(ctx context.Context, question string) (string, error) {
	start := time.Now()
	log.Printf("[rag][query] start question_chars=%d top_k=%d", len(question), s.topK)

	queryEmbedding, err := s.embedClient.Embed(ctx, s.embedModel, question)
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}
	log.Printf("[rag][query] embedded question vector_dim=%d", len(queryEmbedding))

	results, err := s.qdrant.Search(ctx, s.collection, queryEmbedding, s.topK)
	if err != nil {
		return "", fmt.Errorf("search qdrant: %w", err)
	}
	if len(results) == 0 {
		return "", fmt.Errorf("no context found; run ingestion first")
	}
	log.Printf("[rag][query] retrieved context_chunks=%d", len(results))

	var contextBuilder strings.Builder
	for _, r := range results {
		contextBuilder.WriteString(fmt.Sprintf("[Page %d]: %s\n\n", r.Page, r.Text))
	}

	prompt := fmt.Sprintf(
		"Question:\n%s\n\nContext:\n%s\nRespond only from context and include page citations.",
		question,
		contextBuilder.String(),
	)
	log.Printf("[rag][query] assembled prompt_chars=%d", len(prompt))
	log.Printf("[rag][query] retrieval complete duration=%s", time.Since(start))
	return prompt, nil
}
