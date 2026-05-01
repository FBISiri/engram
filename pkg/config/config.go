// Package config provides configuration loading from environment variables.
package config

import (
	"os"
	"strconv"

	"github.com/FBISiri/engram/pkg/memory"
)

// Config holds all Engram configuration.
type Config struct {
	// Storage
	QdrantURL      string
	QdrantAPIKey   string
	QdrantUseTLS   bool
	CollectionName string

	// Embedding
	EmbedderProvider   string // "openai" | "voyage"
	EmbeddingModel     string
	EmbeddingDimension int
	OpenAIAPIKey       string
	OpenAIBaseURL      string
	VoyageAPIKey       string

	// Scoring
	Weights memory.ScoringWeights
	Decay   memory.DecayConfig
	MMRLambda      float64
	DedupThreshold float64

	// Server
	Transport string // "stdio", "http", "both"
	HTTPPort  int
	APIKey    string

	// Reflection
	ReflectionEnabled  bool
	ReflectionTrigger  string // "count", "cron", "manual"
	ReflectionCount    int
	ReflectionModel    string
}

// Load reads configuration from environment variables with sensible defaults.
func Load() *Config {
	return &Config{
		// Storage
		QdrantURL:      envStr("ENGRAM_QDRANT_URL", "localhost:6334"),
		QdrantAPIKey:   envStr("ENGRAM_QDRANT_API_KEY", ""),
		QdrantUseTLS:   envBool("ENGRAM_QDRANT_USE_TLS", false),
		CollectionName: envStr("ENGRAM_COLLECTION_NAME", "engram"),

		// Embedding
		EmbedderProvider:   envStr("ENGRAM_EMBEDDER_PROVIDER", "openai"),
		EmbeddingModel:     envStr("ENGRAM_EMBEDDING_MODEL", "text-embedding-3-small"),
		EmbeddingDimension: envInt("ENGRAM_EMBEDDING_DIMENSION", 1536),
		OpenAIAPIKey:       envStr("ENGRAM_OPENAI_API_KEY", ""),
		OpenAIBaseURL:      envStr("ENGRAM_OPENAI_BASE_URL", "https://api.openai.com/v1"),
		VoyageAPIKey:       envStr("ENGRAM_VOYAGE_API_KEY", ""),

		// Scoring
		Weights: memory.ScoringWeights{
			Relevance:  envFloat("ENGRAM_WEIGHT_RELEVANCE", 1.0),
			Recency:    envFloat("ENGRAM_WEIGHT_RECENCY", 0.5),
			Importance: envFloat("ENGRAM_WEIGHT_IMPORTANCE", 0.3),
		},
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      envFloat("ENGRAM_MMR_LAMBDA", 0.5),
		DedupThreshold: envFloat("ENGRAM_DEDUP_THRESHOLD", 0.92),

		// Server
		Transport: envStr("ENGRAM_TRANSPORT", "stdio"),
		HTTPPort:  envInt("ENGRAM_HTTP_PORT", 8080),
		APIKey:    envStr("ENGRAM_API_KEY", ""),

		// Reflection
		ReflectionEnabled: envBool("ENGRAM_REFLECTION_ENABLED", false),
		ReflectionTrigger: envStr("ENGRAM_REFLECTION_TRIGGER", "count"),
		ReflectionCount:   envInt("ENGRAM_REFLECTION_COUNT", 10),
		ReflectionModel:   envStr("ENGRAM_REFLECTION_MODEL", "claude-sonnet-4-20250514"),
	}
}

func envStr(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

func envInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return defaultVal
}

func envFloat(key string, defaultVal float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

func envBool(key string, defaultVal bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return defaultVal
}
