package rag

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"handbook-rag/internal/chunk"
	"handbook-rag/internal/embed"
	"handbook-rag/internal/llm"
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
	rewriter    *QueryRewriter
}

type QueryRewriter struct {
	Client *llm.OpenAICompatibleClient
	APIKey string
	Model  string
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

func (s *Service) SetQueryRewriter(client *llm.OpenAICompatibleClient, apiKey, model string) {
	apiKey = strings.TrimSpace(apiKey)
	model = strings.TrimSpace(model)
	if client == nil || apiKey == "" || model == "" {
		s.rewriter = nil
		return
	}

	s.rewriter = &QueryRewriter{
		Client: client,
		APIKey: apiKey,
		Model:  model,
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

	pages, err := pdfextract.ExtractByPage(ctx, s.pdfPath)
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

func (s *Service) BuildPrompt(ctx context.Context, question string) (string, []qdrant.SearchResult, error) {
	start := time.Now()
	log.Printf("[rag][query] start question_chars=%d top_k=%d", len(question), s.topK)

	retrievalQuery := question
	if s.rewriter != nil {
		rewrittenQuery, noRewrite, err := rewriteQueryWithLLM(ctx, s.rewriter.Client, s.rewriter.APIKey, s.rewriter.Model, question)
		if err != nil {
			log.Printf("[rag][query] rewrite failed fallback_original=true error=%v", err)
		} else {
			retrievalQuery = rewrittenQuery
			log.Printf("[rag][query] rewrite no_rewrite=%t original=%q rewritten=%q", noRewrite, question, retrievalQuery)
		}
	}

	queryEmbedding, err := s.embedClient.Embed(ctx, s.embedModel, retrievalQuery)
	if err != nil {
		return "", nil, fmt.Errorf("embed query: %w", err)
	}
	log.Printf("[rag][query] embedded question vector_dim=%d", len(queryEmbedding))

	results, err := s.qdrant.Search(ctx, s.collection, queryEmbedding, s.topK)
	if err != nil {
		return "", nil, fmt.Errorf("search qdrant: %w", err)
	}
	if len(results) == 0 {
		return "", nil, fmt.Errorf("no context found; run ingestion first")
	}
	log.Printf("[rag][query] retrieved context_chunks=%d", len(results))

	var contextBuilder strings.Builder
	for _, r := range results {
		contextBuilder.WriteString(fmt.Sprintf("[Page %d]: %s\n\n", r.Page, r.Text))
	}


	userPrompt := fmt.Sprintf(
		`Context (retrieved from handbook):
	%s
	
	---
	
	Question: %s
	
	Think through the following before responding:
	1. Is this a casual or conversational question that doesn't require handbook knowledge?
	2. Does the context contain a clear answer to the question?
	3. Is a citation actually necessary to support this answer?
	4. Use normal English spacing between words (never merge words, e.g. write "must satisfy" not "mustsatisfy").
	
	Then respond using this format:
	
	[Your response here. Only append a citation like (Section X.X, p.N) if the answer references a specific policy, rule, date, or procedure from the handbook. Do not cite for greetings, simple clarifications, or conversational replies.]
	`,
		contextBuilder.String(),
		question,
	)




	log.Printf("[rag][query] assembled prompt_chars=%d", len(userPrompt))
	log.Printf("[rag][query] retrieval complete duration=%s", time.Since(start))
	return userPrompt, results, nil
}
