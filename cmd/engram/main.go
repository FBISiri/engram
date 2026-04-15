// Engram — Long-term memory for AI agents.
package main

import (
	"context"
	"fmt"
	"os"

	"encoding/json"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/dream"
	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/qdrant"
	"github.com/FBISiri/engram/pkg/reflection"
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
	case "dream-check":
		if err := dreamCheck(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "dream-run":
		if err := dreamRun(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "reflection-check":
		if err := reflectionCheck(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "reflection-run":
		if err := reflectionRun(cfg); err != nil {
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
	var embedder embedding.Embedder
	switch cfg.EmbedderProvider {
	case "voyage":
		embedder = embedding.NewVoyage(embedding.VoyageConfig{
			APIKey:    cfg.VoyageAPIKey,
			Model:     cfg.EmbeddingModel,
			Dimension: cfg.EmbeddingDimension,
		})
		fmt.Fprintf(os.Stderr, "  Embedder:   voyage (%s, %dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	default:
		embedder = embedding.NewOpenAI(embedding.OpenAIConfig{
			APIKey:    cfg.OpenAIAPIKey,
			Model:     cfg.EmbeddingModel,
			BaseURL:   cfg.OpenAIBaseURL,
			Dimension: cfg.EmbeddingDimension,
		})
		fmt.Fprintf(os.Stderr, "  Embedder:   openai (%s, %dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	}

	// 3. Create and start server
	srv := server.NewServer(store, embedder, cfg)

	// 4. Start background expiry cleanup goroutine.
	// Uses context.Background() since ServeStdio blocks until process exit;
	// the goroutine will be cleaned up when the process terminates.
	serverCtx, serverCancel := context.WithCancel(context.Background())
	defer serverCancel()
	server.StartExpiryCleanup(serverCtx, store, 0) // 0 = use DefaultExpiryInterval (10 min)

	switch cfg.Transport {
	case "stdio":
		fmt.Fprintf(os.Stderr, "  Transport:  stdio (ready)\n")
		return srv.ServeStdio()
	case "http":
		fmt.Fprintf(os.Stderr, "  Transport:  http (port %d)\n", cfg.HTTPPort)
		httpSrv := server.NewHTTPServer(srv, cfg.HTTPPort, cfg.APIKey)
		return httpSrv.ListenAndServe(serverCtx)
	case "both":
		// Start HTTP in background; MCP stdio in foreground.
		fmt.Fprintf(os.Stderr, "  Transport:  stdio + http (port %d)\n", cfg.HTTPPort)
		httpSrv := server.NewHTTPServer(srv, cfg.HTTPPort, cfg.APIKey)
		go func() {
			if err := httpSrv.ListenAndServe(serverCtx); err != nil {
				fmt.Fprintf(os.Stderr, "http server error: %v\n", err)
			}
		}()
		return srv.ServeStdio()
	default:
		return fmt.Errorf("unknown transport: %s", cfg.Transport)
	}
}

func dreamCheck(cfg *config.Config) error {
	// Connect to Qdrant for Gate 2 (new memories count).
	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		// Non-fatal: run gates without store (Gate 2 will pass by default).
		fmt.Fprintf(os.Stderr, "warning: could not connect to qdrant for gate2: %v\n", err)
		result, err2 := dream.CheckGates(nil)
		if err2 != nil {
			return err2
		}
		return dream.PrintGateResult(result)
	}
	defer store.Close()

	result, err := dream.CheckGates(store)
	if err != nil {
		return err
	}
	return dream.PrintGateResult(result)
}

func dreamRun(cfg *config.Config) error {
	// Parse flags from os.Args[2:].
	dryRun := false
	phase := ""
	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--dry-run":
			dryRun = true
		case "--phase":
			if i+1 < len(os.Args) {
				i++
				phase = os.Args[i]
			} else {
				return fmt.Errorf("--phase requires a value (orient, gather, consolidate, prune)")
			}
		default:
			return fmt.Errorf("unknown flag: %s", os.Args[i])
		}
	}

	// Connect to Qdrant.
	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collection: %w", err)
	}

	// Create embedder for consolidate phase (stores new insights with proper vectors).
	var embedder embedding.Embedder
	switch cfg.EmbedderProvider {
	case "voyage":
		embedder = embedding.NewVoyage(embedding.VoyageConfig{
			APIKey:    cfg.VoyageAPIKey,
			Model:     cfg.EmbeddingModel,
			Dimension: cfg.EmbeddingDimension,
		})
	default:
		embedder = embedding.NewOpenAI(embedding.OpenAIConfig{
			APIKey:    cfg.OpenAIAPIKey,
			Model:     cfg.EmbeddingModel,
			BaseURL:   cfg.OpenAIBaseURL,
			Dimension: cfg.EmbeddingDimension,
		})
	}

	eng := dream.NewEngine(store, embedder, dream.Config{
		DryRun: dryRun,
		Phase:  phase,
	})

	if err := eng.Run(ctx); err != nil {
		return err
	}

	return eng.PrintLog()
}

func printUsage() {
	fmt.Println(`Usage: engram <command>

Commands:
  serve             Start the memory server (MCP and/or REST)
  dream-check       Check Triple Gate (outputs JSON: should_run, reason, etc.)
  dream-run         Run Dream Engine (--dry-run, --phase <name>)
  reflection-check  Check if Reflection Engine should run (outputs JSON)
  reflection-run    Run Reflection Engine (--dry-run)
  migrate           Migrate from chat2mem to Engram
  version           Print version`)
}

// reflectionCheck evaluates whether the Reflection Engine should run now.
// Outputs JSON: {should_trigger, skip_reason, unreflected_count, accumulated_importance, ...}
func reflectionCheck(cfg *config.Config) error {
	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	eng := reflection.NewEngine(store, nil, reflection.DefaultConfig())
	ctx := context.Background()
	result, err := eng.Check(ctx)
	if err != nil {
		return fmt.Errorf("reflection check: %w", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// reflectionRun executes one reflection cycle.
// Outputs JSON RunResult.
func reflectionRun(cfg *config.Config) error {
	dryRun := false
	for i := 2; i < len(os.Args); i++ {
		if os.Args[i] == "--dry-run" {
			dryRun = true
		}
	}

	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collection: %w", err)
	}

	// Create embedder for storing new insights with proper vectors.
	var embedder embedding.Embedder
	switch cfg.EmbedderProvider {
	case "voyage":
		embedder = embedding.NewVoyage(embedding.VoyageConfig{
			APIKey:    cfg.VoyageAPIKey,
			Model:     cfg.EmbeddingModel,
			Dimension: cfg.EmbeddingDimension,
		})
	default:
		embedder = embedding.NewOpenAI(embedding.OpenAIConfig{
			APIKey:    cfg.OpenAIAPIKey,
			Model:     cfg.EmbeddingModel,
			BaseURL:   cfg.OpenAIBaseURL,
			Dimension: cfg.EmbeddingDimension,
		})
	}

	reflCfg := reflection.DefaultConfig()
	reflCfg.DryRun = dryRun

	eng := reflection.NewEngine(store, embedder, reflCfg)
	result, err := eng.Run(ctx)
	if err != nil {
		return fmt.Errorf("reflection run: %w", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}
