package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"handbook-rag/internal/config"
	"handbook-rag/internal/embed"
	"handbook-rag/internal/llm"
	"handbook-rag/internal/qdrant"
	"handbook-rag/internal/rag"
)

type chunkResult struct {
	Text  string  `json:"text"`
	Page  int     `json:"page"`
	Score float64 `json:"score"`
}

type experimentRecord struct {
	Timestamp       string        `json:"timestamp"`
	ChunkWords      int           `json:"chunk_words"`
	TopK            int           `json:"top_k"`
	Collection      string        `json:"collection"`
	Question        string        `json:"question"`
	RetrievedChunks []chunkResult `json:"retrieved_chunks"`
	AvgScore        float64       `json:"avg_score"`
	Answer          string        `json:"answer,omitempty"`
}

type modelDef struct {
	provider string
	model    string
	envKey   string
}

var models = map[string]modelDef{
	"openrouter_nemotron": {
		provider: "openrouter",
		model:    "nvidia/nemotron-3-nano-30b-a3b:free",
		envKey:   "OPENROUTER_API_KEY",
	},
	"openrouter_llama4_scout": {
		provider: "openrouter",
		model:    "meta-llama/llama-4-scout-17b-16e-instruct",
		envKey:   "OPENROUTER_API_KEY",
	},
	"groq_llama31_8b": {
		provider: "groq",
		model:    "llama-3.1-8b-instant",
		envKey:   "GROQ_API_KEY",
	},
}

func main() {
	chunkWords := flag.Int("chunk-words", 180, "words per chunk")
	topK := flag.Int("top-k", 10, "number of chunks to retrieve")
	modelID := flag.String("model", "groq_llama31_8b", "LLM model ID for answer generation (empty to skip)")
	questionsPath := flag.String("questions", "docs/eval_questions.txt", "path to questions file")
	outputPath := flag.String("output", "docs/experiments.jsonl", "path to append-only JSONL results file")
	flag.Parse()

	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	collection := fmt.Sprintf("handbook_eval_w%d_k%d", *chunkWords, *topK)
	log.Printf("[eval] chunk_words=%d top_k=%d collection=%s", *chunkWords, *topK, collection)

	svc := rag.NewServiceWithParams(
		collection, *topK, *chunkWords, cfg.HandbookPath,
		embed.NewOllamaClient(cfg.OllamaHost),
		qdrant.New(cfg.QdrantHost),
	)

	ctx := context.Background()

	log.Printf("[eval] ingesting PDF into collection=%s ...", collection)
	count, err := svc.Ingest(ctx)
	if err != nil {
		log.Fatalf("ingest: %v", err)
	}
	log.Printf("[eval] ingested chunks=%d", count)

	questions, err := loadQuestions(*questionsPath)
	if err != nil {
		log.Fatalf("load questions from %s: %v", *questionsPath, err)
	}
	log.Printf("[eval] loaded questions=%d", len(questions))

	llmClient, llmModel, llmAPIKey := setupLLM(*modelID, cfg)

	out, err := os.OpenFile(*outputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		log.Fatalf("open output file %s: %v", *outputPath, err)
	}
	defer out.Close()

	enc := json.NewEncoder(out)

	for i, q := range questions {
		log.Printf("[eval] [%d/%d] %s", i+1, len(questions), q)

		results, err := svc.Retrieve(ctx, q)
		if err != nil {
			log.Printf("[eval] retrieve failed: %v — skipping", err)
			continue
		}

		chunks := make([]chunkResult, len(results))
		var scoreSum float64
		for j, r := range results {
			chunks[j] = chunkResult{Text: r.Text, Page: r.Page, Score: r.Score}
			scoreSum += r.Score
		}
		avgScore := 0.0
		if len(results) > 0 {
			avgScore = scoreSum / float64(len(results))
		}

		record := experimentRecord{
			Timestamp:       time.Now().UTC().Format(time.RFC3339),
			ChunkWords:      *chunkWords,
			TopK:            *topK,
			Collection:      collection,
			Question:        q,
			RetrievedChunks: chunks,
			AvgScore:        avgScore,
		}

		if llmClient != nil {
			answer, err := getAnswer(ctx, llmClient, llmAPIKey, llmModel, q, results)
			if err != nil {
				log.Printf("[eval] LLM failed: %v", err)
			} else {
				record.Answer = answer
			}
		}

		if err := enc.Encode(record); err != nil {
			log.Printf("[eval] write record failed: %v", err)
		}
		log.Printf("[eval] [%d/%d] done avg_score=%.4f", i+1, len(questions), avgScore)
	}

	log.Printf("[eval] complete — results in %s", *outputPath)
}

func getAnswer(ctx context.Context, client *llm.OpenAICompatibleClient, apiKey, model, question string, results []qdrant.SearchResult) (string, error) {
	var ctx2 strings.Builder
	for _, r := range results {
		ctx2.WriteString(fmt.Sprintf("[Page %d]: %s\n\n", r.Page, r.Text))
	}
	prompt := fmt.Sprintf(
		"Question:\n%s\n\nContext:\n%s\nRespond only from context and include page citations.",
		question, ctx2.String(),
	)
	var answer strings.Builder
	err := client.StreamAnswer(ctx, apiKey, model, prompt, func(token string) error {
		answer.WriteString(token)
		return nil
	})
	return answer.String(), err
}

func loadQuestions(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		out = append(out, line)
	}
	return out, scanner.Err()
}

func setupLLM(modelID string, cfg config.Config) (*llm.OpenAICompatibleClient, string, string) {
	if modelID == "" {
		return nil, "", ""
	}
	md, ok := models[modelID]
	if !ok {
		log.Printf("[eval] unknown model_id=%s — skipping LLM", modelID)
		return nil, "", ""
	}

	apiKeys := map[string]string{
		"OPENROUTER_API_KEY": cfg.OpenRouterAPIKey,
		"GROQ_API_KEY":       cfg.GroqAPIKey,
	}
	apiKey := apiKeys[md.envKey]
	if apiKey == "" {
		log.Printf("[eval] missing %s — skipping LLM", md.envKey)
		return nil, "", ""
	}

	var client *llm.OpenAICompatibleClient
	switch md.provider {
	case "openrouter":
		client = llm.NewOpenAICompatibleClient("https://openrouter.ai/api/v1", map[string]string{
			"HTTP-Referer": "http://localhost",
			"X-Title":      "handbook-rag-eval",
		})
	case "groq":
		client = llm.NewOpenAICompatibleClient("https://api.groq.com/openai/v1", nil)
	default:
		log.Printf("[eval] unknown provider=%s", md.provider)
		return nil, "", ""
	}
	return client, md.model, apiKey
}
