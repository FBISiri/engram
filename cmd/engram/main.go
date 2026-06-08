// Engram — Long-term memory for AI agents.
package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"encoding/json"

	otelpkg "github.com/FBISiri/engram/internal/otel"
	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/dream"
	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/qdrant"
	"github.com/FBISiri/engram/pkg/reflection"
	"github.com/FBISiri/engram/pkg/server"
	"github.com/FBISiri/engram/pkg/trajectory"
)

var (
	version   = "dev"
	gitCommit = "unknown"
	buildTime = "unknown"
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
	case "migrate-collections":
		if err := migrateCollections(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "drop-legacy":
		if err := dropLegacy(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "migrate-extra-collections":
		if err := migrateExtraCollections(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "migrate-reflected":
		if err := migrateReflected(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "version":
		fmt.Printf("engram %s (commit=%s built=%s)\n", version, gitCommit, buildTime)
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func serve(cfg *config.Config) error {
	tp, err := otelpkg.NewTracerProvider(otelpkg.LoadConfigFromEnv())
	if err != nil {
		return fmt.Errorf("init tracing: %w", err)
	}
	if tp != nil {
		defer tp.Shutdown(context.Background())
	}

	fmt.Fprintf(os.Stderr, "Starting Engram server (transport: %s)\n", cfg.Transport)
	fmt.Fprintf(os.Stderr, "  Qdrant:     %s (collections: engram_user, engram_agent_self, engram_reflection)\n", cfg.QdrantURL)
	fmt.Fprintf(os.Stderr, "  Embedding:  %s (%dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	fmt.Fprintf(os.Stderr, "  Scoring:    relevance=%.1f recency=%.1f importance=%.1f\n",
		cfg.Weights.Relevance, cfg.Weights.Recency, cfg.Weights.Importance)

	// 1. Create one Qdrant store per physical collection (Phase 4: physical isolation).
	storeCfg := qdrant.Config{
		URL:       cfg.QdrantURL,
		APIKey:    cfg.QdrantAPIKey,
		UseTLS:    cfg.QdrantUseTLS,
		Dimension: uint64(cfg.EmbeddingDimension),
	}
	collectionNames := []string{collection.CollectionUser, collection.CollectionAgentSelf, collection.CollectionReflection}
	storeMap := make(map[string]*qdrant.Store, len(collectionNames))
	for _, col := range collectionNames {
		storeCfg.CollectionName = col
		s, err := qdrant.New(storeCfg)
		if err != nil {
			return fmt.Errorf("create qdrant store for %s: %w", col, err)
		}
		storeMap[col] = s
	}
	store := qdrant.NewMultiStore(storeMap, collection.CollectionUser)
	defer store.Close()

	// Ensure all physical collections exist.
	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collections: %w", err)
	}
	fmt.Fprintf(os.Stderr, "  Collections: ready\n")

	// 2. Create embedder
	var rawEmbedder embedding.Embedder
	switch cfg.EmbedderProvider {
	case "voyage":
		rawEmbedder = embedding.NewVoyage(embedding.VoyageConfig{
			APIKey:    cfg.VoyageAPIKey,
			Model:     cfg.EmbeddingModel,
			Dimension: cfg.EmbeddingDimension,
		})
		fmt.Fprintf(os.Stderr, "  Embedder:   voyage (%s, %dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	default:
		rawEmbedder = embedding.NewOpenAI(embedding.OpenAIConfig{
			APIKey:    cfg.OpenAIAPIKey,
			Model:     cfg.EmbeddingModel,
			BaseURL:   cfg.OpenAIBaseURL,
			Dimension: cfg.EmbeddingDimension,
		})
		fmt.Fprintf(os.Stderr, "  Embedder:   openai (%s, %dd)\n", cfg.EmbeddingModel, cfg.EmbeddingDimension)
	}

	// Wrap embedder with LRU cache (P5-A1).
	embedCache := memory.NewEmbedCache(0) // default 10k entries
	embedder := embedding.NewCachingEmbedder(rawEmbedder, embedCache, cfg.EmbeddingModel+"/v1")
	fmt.Fprintf(os.Stderr, "  EmbedCache: LRU 10k entries\n")

	// 3. Create and start server
	srv := server.NewServer(store, embedder, cfg)
	srv.SetEmbedCache(embedCache)

	// 4a. Start trajectory logger if ENGRAM_TRAJECTORY_DIR is set.
	trajectoryDir := os.Getenv("ENGRAM_TRAJECTORY_DIR")
	if trajectoryDir == "" {
		trajectoryDir = "/data/engram/trajectories"
	}
	tl := trajectory.New(trajectoryDir)
	defer tl.Close()
	srv.SetTrajectoryLogger(tl)
	fmt.Fprintf(os.Stderr, "  Trajectory: %s\n", trajectoryDir)

	// 4b. Start background expiry cleanup goroutine.
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

// newMultiStore creates a MultiStore backed by all three logical collections.
func newMultiStore(cfg *config.Config) (*qdrant.MultiStore, error) {
	baseCfg := qdrant.Config{
		URL:       cfg.QdrantURL,
		APIKey:    cfg.QdrantAPIKey,
		UseTLS:    cfg.QdrantUseTLS,
		Dimension: uint64(cfg.EmbeddingDimension),
	}
	collectionNames := []string{collection.CollectionUser, collection.CollectionAgentSelf, collection.CollectionReflection}
	storeMap := make(map[string]*qdrant.Store, len(collectionNames))
	for _, col := range collectionNames {
		baseCfg.CollectionName = col
		s, err := qdrant.New(baseCfg)
		if err != nil {
			return nil, fmt.Errorf("create qdrant store for %s: %w", col, err)
		}
		storeMap[col] = s
	}
	return qdrant.NewMultiStore(storeMap, collection.CollectionUser), nil
}

func dreamCheck(cfg *config.Config) error {
	// Connect to Qdrant for Gate 2 (new memories count).
	store, err := newMultiStore(cfg)
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
	tp, err := otelpkg.NewTracerProvider(otelpkg.LoadConfigFromEnv())
	if err != nil {
		return fmt.Errorf("init tracing: %w", err)
	}
	if tp != nil {
		defer tp.Shutdown(context.Background())
	}

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
	store, err := newMultiStore(cfg)
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collections: %w", err)
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
  serve                        Start the memory server (MCP and/or REST)
  dream-check                  Check Triple Gate (outputs JSON: should_run, reason, etc.)
  dream-run                    Run Dream Engine (--dry-run, --phase <name>)
  reflection-check             Check if Reflection Engine should run (outputs JSON)
  reflection-run               Run Reflection Engine (--dry-run, --force, --mode <v1|v2>)
  migrate                      Migrate from chat2mem to Engram
  migrate-collections          Migrate legacy "engram" collection to new 3-collection layout (--dry-run, --batch N)
  migrate-extra-collections    Migrate extra collections (e.g. siri,bmo) into engram_user (--source, --target, --dry-run, --reembed)
  drop-legacy                  Delete legacy collections (--collections, --confirm required)
  migrate-reflected            Backfill top-level reflected_at from legacy metadata["reflected"]=true (W17 T1)
  version                      Print version`)
}

// reflectionCheck evaluates whether the Reflection Engine should run now.
// Outputs JSON: {should_trigger, skip_reason, unreflected_count, accumulated_importance, ...}
func reflectionCheck(cfg *config.Config) error {
	store, err := newMultiStore(cfg)
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
	tp, err := otelpkg.NewTracerProvider(otelpkg.LoadConfigFromEnv())
	if err != nil {
		return fmt.Errorf("init tracing: %w", err)
	}
	if tp != nil {
		defer tp.Shutdown(context.Background())
	}

	dryRun := false
	force := false
	mode := ""
	debugEvidence := false
	for i := 2; i < len(os.Args); i++ {
		if os.Args[i] == "--dry-run" {
			dryRun = true
		}
		if os.Args[i] == "--force" {
			force = true
		}
		if os.Args[i] == "--mode" && i+1 < len(os.Args) {
			mode = os.Args[i+1]
			i++
		}
		if os.Args[i] == "--debug-evidence" {
			debugEvidence = true
		}
	}

	store, err := newMultiStore(cfg)
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collections: %w", err)
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
	reflCfg.Force = force
	reflCfg.DebugEvidence = debugEvidence
	if mode != "" {
		reflCfg.Mode = mode
	}

	eng := reflection.NewEngine(store, embedder, reflCfg)
	result, err := eng.Run(ctx)
	if err != nil {
		return fmt.Errorf("reflection run: %w", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// migrateReflected backfills the top-level ReflectedAt field from the legacy
// metadata["reflected"]=true boolean.
//
// W17 T1 Part 2: for every memory where metadata["reflected"]==true AND
// ReflectedAt == 0, set ReflectedAt to UpdatedAt (or CreatedAt if UpdatedAt
// is zero). Legacy metadata key is preserved for backward compatibility —
// this migration is additive only.
//
// Flags:
//
//	--dry-run   scan + report, don't write
//	--batch N   scroll batch size (default 100)
func migrateReflected(cfg *config.Config) error {
	dryRun := false
	batchSize := 100
	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--dry-run":
			dryRun = true
		case "--batch":
			if i+1 < len(os.Args) {
				var n int
				if _, err := fmt.Sscanf(os.Args[i+1], "%d", &n); err == nil && n > 0 {
					batchSize = n
				}
				i++
			}
		}
	}

	store, err := newMultiStore(cfg)
	if err != nil {
		return fmt.Errorf("connect qdrant: %w", err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureCollection(ctx); err != nil {
		return fmt.Errorf("ensure collections: %w", err)
	}

	fmt.Fprintf(os.Stderr, "migrate-reflected: dry_run=%v batch_size=%d\n", dryRun, batchSize)

	var (
		scanned int
		matched int
		updated int
		offset  string
	)
	for {
		batch, next, err := store.Scroll(ctx, memory.ScrollOptions{
			Limit:  batchSize,
			Offset: offset,
		})
		if err != nil {
			return fmt.Errorf("scroll: %w", err)
		}
		if len(batch) == 0 {
			break
		}
		scanned += len(batch)

		for _, m := range batch {
			// Already has top-level ReflectedAt — nothing to do.
			if m.ReflectedAt > 0 {
				continue
			}
			// Only migrate when legacy metadata["reflected"] == true.
			if m.Metadata == nil {
				continue
			}
			v, ok := m.Metadata["reflected"]
			if !ok {
				continue
			}
			b, ok := v.(bool)
			if !ok || !b {
				continue
			}
			matched++

			// Pick a sensible timestamp: UpdatedAt, else CreatedAt.
			ts := m.UpdatedAt
			if ts <= 0 {
				ts = m.CreatedAt
			}
			if ts <= 0 {
				// Last-resort: now. Shouldn't happen in practice.
				ts = float64(time.Now().Unix())
			}

			if dryRun {
				fmt.Fprintf(os.Stderr, "  would migrate id=%s reflected_at=%.0f\n", m.ID, ts)
				continue
			}

			if err := store.Update(ctx, m.ID, map[string]any{
				"reflected_at": ts,
			}); err != nil {
				fmt.Fprintf(os.Stderr, "  update %s failed: %v\n", m.ID, err)
				continue
			}
			updated++
		}

		if next == "" || next == offset {
			break
		}
		offset = next
	}

	result := map[string]any{
		"dry_run": dryRun,
		"scanned": scanned,
		"matched": matched,
		"updated": updated,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// migrateCollections scrolls the legacy "engram" collection and writes each
// memory into the correct new physical collection based on its payload.collection
// field. Memories with an empty or unrecognised collection field fall back to
// engram_user (Phase4 D1 spec).
//
// Uses ScrollWithVectors to copy points without re-calling the embedding API.
// Use --reembed to re-generate vectors for legacy records that have empty vectors.
//
// Flags: --dry-run, --reembed, --batch N (default 100)
func migrateCollections(cfg *config.Config) error {
	dryRun := false
	reembed := false
	batchSize := 100
	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--dry-run":
			dryRun = true
		case "--reembed":
			reembed = true
		case "--batch":
			if i+1 < len(os.Args) {
				var n int
				if _, err := fmt.Sscanf(os.Args[i+1], "%d", &n); err == nil && n > 0 {
					batchSize = n
				}
				i++
			}
		}
	}

	baseCfg := qdrant.Config{
		URL:       cfg.QdrantURL,
		APIKey:    cfg.QdrantAPIKey,
		UseTLS:    cfg.QdrantUseTLS,
		Dimension: uint64(cfg.EmbeddingDimension),
	}

	// Legacy source store — do not call EnsureCollection (must already exist).
	baseCfg.CollectionName = "engram"
	legacyStore, err := qdrant.New(baseCfg)
	if err != nil {
		return fmt.Errorf("connect legacy store: %w", err)
	}
	defer legacyStore.Close()

	// Target stores — one per new collection.
	targetStoreMap := map[string]*qdrant.Store{}
	for _, col := range []string{collection.CollectionUser, collection.CollectionAgentSelf, collection.CollectionReflection} {
		baseCfg.CollectionName = col
		s, err := qdrant.New(baseCfg)
		if err != nil {
			return fmt.Errorf("connect target store %s: %w", col, err)
		}
		defer s.Close()
		targetStoreMap[col] = s
	}

	ctx := context.Background()
	if !dryRun {
		for col, s := range targetStoreMap {
			if err := s.EnsureCollection(ctx); err != nil {
				return fmt.Errorf("ensure collection %s: %w", col, err)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "migrate-collections: dry_run=%v reembed=%v batch_size=%d\n", dryRun, reembed, batchSize)

	var embedder embedding.Embedder
	if reembed {
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
	}

	// validTargets is the set of recognised physical collection names.
	validTargets := map[string]bool{
		collection.CollectionUser:       true,
		collection.CollectionAgentSelf:  true,
		collection.CollectionReflection: true,
	}

	counts := map[string]int{} // source payload.collection value → count
	inserted := 0
	errCount := 0
	var offset string

	for {
		batch, next, err := legacyStore.ScrollWithVectors(ctx, memory.ScrollOptions{
			Limit:  batchSize,
			Offset: offset,
		})
		if err != nil {
			return fmt.Errorf("scroll legacy: %w", err)
		}
		if len(batch) == 0 {
			break
		}

		for i := range batch {
			sm := &batch[i]
			origCol := sm.Collection
			counts[origCol]++

			// Resolve target collection; empty/unknown → engram_user.
			target := origCol
			if !validTargets[target] {
				target = collection.CollectionUser
			}
			sm.Collection = target

			if dryRun {
				continue
			}

			if reembed && len(sm.Vector) == 0 {
				vec, err := embedder.Embed(ctx, sm.Content)
				if err != nil {
					fmt.Fprintf(os.Stderr, "  reembed id=%s err=%v\n", sm.ID, err)
					errCount++
					continue
				}
				sm.Vector = vec
			}

			targetStore := targetStoreMap[target]
			if err := targetStore.Insert(ctx, &sm.Memory, sm.Vector); err != nil {
				fmt.Fprintf(os.Stderr, "  insert id=%s target=%s err=%v\n", sm.ID, target, err)
				errCount++
				continue
			}
			inserted++
		}

		if next == "" || next == offset {
			break
		}
		offset = next
	}

	result := map[string]any{
		"dry_run":            dryRun,
		"source_collection":  "engram",
		"payload_col_counts": counts,
		"inserted":           inserted,
		"errors":             errCount,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}

// dropLegacy deletes one or more Qdrant collections permanently.
//
// Flags:
//
//	--collections  comma-separated collection names to delete (default: engram)
//	--confirm      required safety flag; command refuses to run without it
func dropLegacy(cfg *config.Config) error {
	collectionsArg := "engram"
	confirm := false
	for i := 2; i < len(os.Args); i++ {
		switch {
		case os.Args[i] == "--confirm":
			confirm = true
		case strings.HasPrefix(os.Args[i], "--collections="):
			collectionsArg = strings.TrimPrefix(os.Args[i], "--collections=")
		case os.Args[i] == "--collections" && i+1 < len(os.Args):
			i++
			collectionsArg = os.Args[i]
		}
	}
	if !confirm {
		return fmt.Errorf("drop-legacy requires --confirm flag; this operation is irreversible")
	}

	toDelete := strings.Split(collectionsArg, ",")
	baseCfg := qdrant.Config{
		URL:       cfg.QdrantURL,
		APIKey:    cfg.QdrantAPIKey,
		UseTLS:    cfg.QdrantUseTLS,
		Dimension: uint64(cfg.EmbeddingDimension),
	}
	ctx := context.Background()
	for _, col := range toDelete {
		col = strings.TrimSpace(col)
		if col == "" {
			continue
		}
		baseCfg.CollectionName = col
		s, err := qdrant.New(baseCfg)
		if err != nil {
			return fmt.Errorf("connect store %s: %w", col, err)
		}
		if err := s.DropCollection(ctx); err != nil {
			s.Close()
			return fmt.Errorf("drop collection %s: %w", col, err)
		}
		s.Close()
		fmt.Fprintf(os.Stderr, "drop-legacy: deleted collection %q\n", col)
	}
	return nil
}

// migrateExtraCollections migrates arbitrary source collections (e.g. "siri", "bmo")
// into a single target collection (default: engram_user).
//
// Design:
//   - Scrolls each source collection in full.
//   - For each point, checks whether the ID already exists in the target (target wins).
//   - Stamps payload.collection to the target collection name before inserting.
//   - Optionally re-generates embeddings for points with empty vectors (--reembed).
//
// Flags:
//
//	--source   comma-separated source collection names (required)
//	--target   target collection name (default: engram_user)
//	--dry-run  scan + report, don't write
//	--reembed  regenerate vectors for points with empty vectors
//	--batch N  scroll batch size (default: 100)
func migrateExtraCollections(cfg *config.Config) error {
	sourceArg := ""
	targetName := collection.CollectionUser
	dryRun := false
	reembed := false
	batchSize := 100
	for i := 2; i < len(os.Args); i++ {
		switch {
		case os.Args[i] == "--dry-run":
			dryRun = true
		case os.Args[i] == "--reembed":
			reembed = true
		case strings.HasPrefix(os.Args[i], "--source="):
			sourceArg = strings.TrimPrefix(os.Args[i], "--source=")
		case os.Args[i] == "--source" && i+1 < len(os.Args):
			i++
			sourceArg = os.Args[i]
		case strings.HasPrefix(os.Args[i], "--target="):
			targetName = strings.TrimPrefix(os.Args[i], "--target=")
		case os.Args[i] == "--target" && i+1 < len(os.Args):
			i++
			targetName = os.Args[i]
		case os.Args[i] == "--batch" && i+1 < len(os.Args):
			i++
			var n int
			if _, err := fmt.Sscanf(os.Args[i], "%d", &n); err == nil && n > 0 {
				batchSize = n
			}
		}
	}
	if sourceArg == "" {
		return fmt.Errorf("--source is required (comma-separated collection names, e.g. siri,bmo)")
	}
	sources := strings.Split(sourceArg, ",")
	for i, s := range sources {
		sources[i] = strings.TrimSpace(s)
	}

	baseCfg := qdrant.Config{
		URL:       cfg.QdrantURL,
		APIKey:    cfg.QdrantAPIKey,
		UseTLS:    cfg.QdrantUseTLS,
		Dimension: uint64(cfg.EmbeddingDimension),
	}
	ctx := context.Background()

	// Target store.
	baseCfg.CollectionName = targetName
	targetStore, err := qdrant.New(baseCfg)
	if err != nil {
		return fmt.Errorf("connect target store %s: %w", targetName, err)
	}
	defer targetStore.Close()
	if !dryRun {
		if err := targetStore.EnsureCollection(ctx); err != nil {
			return fmt.Errorf("ensure target collection %s: %w", targetName, err)
		}
	}

	var embedder embedding.Embedder
	if reembed {
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
	}

	fmt.Fprintf(os.Stderr, "migrate-extra-collections: sources=%v target=%s dry_run=%v reembed=%v batch_size=%d\n",
		sources, targetName, dryRun, reembed, batchSize)

	sourceCounts := map[string]int{}
	targetInserted := 0
	skippedDuplicate := 0
	errCount := 0

	for _, srcName := range sources {
		baseCfg.CollectionName = srcName
		srcStore, err := qdrant.New(baseCfg)
		if err != nil {
			return fmt.Errorf("connect source store %s: %w", srcName, err)
		}

		var offset string
		for {
			batch, next, err := srcStore.ScrollWithVectors(ctx, memory.ScrollOptions{
				Limit:  batchSize,
				Offset: offset,
			})
			if err != nil {
				srcStore.Close()
				return fmt.Errorf("scroll %s: %w", srcName, err)
			}
			if len(batch) == 0 {
				break
			}

			// Collect IDs to batch-check existence in target.
			ids := make([]string, len(batch))
			for i, sm := range batch {
				ids[i] = sm.ID
			}
			existingInTarget := map[string]bool{}
			if !dryRun {
				existing, err := targetStore.SearchByIDs(ctx, ids)
				if err != nil {
					fmt.Fprintf(os.Stderr, "  warn: SearchByIDs failed: %v\n", err)
				}
				for _, m := range existing {
					existingInTarget[m.ID] = true
				}
			}

			for i := range batch {
				sm := &batch[i]
				sourceCounts[srcName]++

				if existingInTarget[sm.ID] {
					skippedDuplicate++
					continue
				}

				sm.Collection = targetName

				if dryRun {
					continue
				}

				if reembed && len(sm.Vector) == 0 {
					vec, err := embedder.Embed(ctx, sm.Content)
					if err != nil {
						fmt.Fprintf(os.Stderr, "  reembed id=%s err=%v\n", sm.ID, err)
						errCount++
						continue
					}
					sm.Vector = vec
				}

				if err := targetStore.Insert(ctx, &sm.Memory, sm.Vector); err != nil {
					fmt.Fprintf(os.Stderr, "  insert id=%s src=%s err=%v\n", sm.ID, srcName, err)
					errCount++
					continue
				}
				targetInserted++
			}

			if next == "" || next == offset {
				break
			}
			offset = next
		}
		srcStore.Close()
	}

	result := map[string]any{
		"dry_run":           dryRun,
		"source_counts":     sourceCounts,
		"target_inserted":   targetInserted,
		"skipped_duplicate": skippedDuplicate,
		"errors":            errCount,
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(result)
}
