package main

import (
	"context"
	"log"

	"handbook-rag/internal/config"
	"handbook-rag/internal/embed"
	"handbook-rag/internal/qdrant"
	"handbook-rag/internal/rag"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	log.Printf(
		"[ingest] start qdrant_host=%s collection=%s handbook=%s ollama_host=%s",
		cfg.QdrantHost,
		cfg.CollectionName,
		cfg.HandbookPath,
		cfg.OllamaHost,
	)

	svc := rag.NewService(
		cfg.CollectionName,
		cfg.TopK,
		cfg.HandbookPath,
		embed.NewOllamaClient(cfg.OllamaHost),
		qdrant.New(cfg.QdrantHost),
	)

	count, err := svc.Ingest(context.Background())
	if err != nil {
		log.Fatalf("[ingest] failed: %v", err)
	}

	log.Printf("[ingest] complete indexed_chunks=%d", count)
}
