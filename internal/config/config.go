package config

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

type Config struct {
	Port             string
	OpenRouterAPIKey string
	GroqAPIKey       string
	HandbookPath     string
	OllamaHost       string
	OllamaAPIKey     string
	QdrantHost       string
	CollectionName   string
	TopK             int
}

func Load() (Config, error) {
	loadDotEnv(".env")

	cfg := Config{
		Port:             getEnvOrDefault("PORT", "8080"),
		OpenRouterAPIKey: os.Getenv("OPENROUTER_API_KEY"),
		GroqAPIKey:       os.Getenv("GROQ_API_KEY"),
		HandbookPath:     os.Getenv("HANDBOOK_PATH"),
		OllamaHost:       getEnvOrDefault("OLLAMA_HOST", "http://localhost:11434"),
		OllamaAPIKey:     strings.TrimSpace(os.Getenv("OLLAMA_API_KEY")),
		QdrantHost:       getEnvOrDefault("QDRANT_HOST", "http://localhost:6333"),
		CollectionName:   getEnvOrDefault("QDRANT_COLLECTION", "handbook_chunks"),
		TopK:             10,
	}

	if cfg.HandbookPath == "" {
		return Config{}, fmt.Errorf("HANDBOOK_PATH is required")
	}

	return cfg, nil
}

func getEnvOrDefault(key, fallback string) string {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	return v
}

func loadDotEnv(path string) {
	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		val = strings.Trim(val, `"`)

		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		_ = os.Setenv(key, val)
	}
}
