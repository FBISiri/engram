// Engram — Long-term memory for AI agents.
package main

import (
	"fmt"
	"os"

	"github.com/anthropics/engram/pkg/config"
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
	fmt.Printf("Starting Engram server (transport: %s)\n", cfg.Transport)
	fmt.Printf("  Qdrant:     %s (collection: %s)\n", cfg.QdrantURL, cfg.CollectionName)
	fmt.Printf("  Embedding:  %s (%dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	fmt.Printf("  Scoring:    relevance=%.1f recency=%.1f importance=%.1f\n",
		cfg.Weights.Relevance, cfg.Weights.Recency, cfg.Weights.Importance)

	// TODO: Initialize store, embedder, and server
	// 1. Create Qdrant store
	// 2. Create OpenAI embedder
	// 3. Start MCP/HTTP server based on transport config

	return fmt.Errorf("server not yet implemented — scaffold only")
}

func printUsage() {
	fmt.Println(`Usage: engram <command>

Commands:
  serve     Start the memory server (MCP and/or REST)
  migrate   Migrate from chat2mem to Engram
  version   Print version`)
}
