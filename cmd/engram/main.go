// Engram — Long-term memory for AI agents.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/qdrant"
	"github.com/FBISiri/engram/pkg/server"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cfg := config.Load()

	switch os.Args[1] {
	case "serve":
		if err := serve(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "migrate":
		fmt.Println("Migration tool not yet implemented.")
		os.Exit(1)
	case "version":
		fmt.Println("engram v0.1.0")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func serve(cfg *config.Config) error {
	fmt.Fprintf(os.Stderr, "Starting Engram server (transport: %s)\n", cfg.Transport)
	fmt.Fprintf(os.Stderr, "  Qdrant:     %s (collection: %s)\n", cfg.QdrantURL, cfg.CollectionName)
	fmt.Fprintf(os.Stderr, "  Embedding:  %s (%dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	fmt.Fprintf(os.Stderr, "  Scoring:    relevance=%.1f recency=%.1f importance=%.1f\n",
		cfg.Weights.Relevance, cfg.Weights.Recency, cfg.Weights.Importance)

	// 1. Create Qdrant store
	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		return fmt.Errorf("create qdrant store: %w", err)
	}
	defer store.Close()

	// Ensure collection exists
	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collection: %w", err)
	}
	fmt.Fprintf(os.Stderr, "  Collection: ready\n")

	// 2. Create embedder
	embedder := embedding.NewOpenAI(embedding.OpenAIConfig{
		APIKey:    cfg.OpenAIAPIKey,
		Model:     cfg.EmbeddingModel,
		BaseURL:   cfg.OpenAIBaseURL,
		Dimension: cfg.EmbeddingDimension,
	})

	// 3. Create and start server
	srv := server.NewServer(store, embedder, cfg)

	switch cfg.Transport {
	case "stdio":
		fmt.Fprintf(os.Stderr, "  Transport:  stdio (ready)\n")
		return srv.ServeStdio()
	case "http":
		return fmt.Errorf("HTTP transport not yet implemented")
	case "both":
		return fmt.Errorf("dual transport not yet implemented")
	default:
		return fmt.Errorf("unknown transport: %s", cfg.Transport)
	}
}

func printUsage() {
	fmt.Println(`Usage: engram <command>

Commands:
  serve     Start the memory server (MCP and/or REST)
  migrate   Migrate from chat2mem to Engram
  version   Print version`)
}
