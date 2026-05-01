// W17 schema migration: adds ArchivedAt, ArchiveReason payload indexes to Qdrant.
// ReflectedAt and LastAccessedAt indexes are already created by EnsureCollection.
//
// Usage:
//
//	go run ./cmd/migrate/w17_schema.go [--dry-run]
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/qdrant"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "w17_schema: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	dryRun := false
	for _, arg := range os.Args[1:] {
		if arg == "--dry-run" {
			dryRun = true
		}
	}

	cfg := config.Load()

	if dryRun {
		fmt.Fprintf(os.Stderr, "w17_schema: dry-run mode — will only report planned actions\n")
		result := map[string]any{
			"dry_run": true,
			"actions": []string{
				"create payload index: archived_at (float)",
				"create payload index: archive_reason (keyword)",
			},
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(result)
	}

	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		UseTLS:         cfg.QdrantUseTLS,
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

	fmt.Fprintf(os.Stderr, "w17_schema: indexes created (archived_at, archive_reason + all existing)\n")

	result := map[string]any{
		"dry_run":         false,
		"indexes_ensured": []string{"archived_at", "archive_reason"},
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}
