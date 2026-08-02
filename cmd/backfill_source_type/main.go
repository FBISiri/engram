// cmd/backfill_source_type — source_type C1 Residual P2 Gap #3
//
// Backfills metadata["source_type"] onto legacy memories that pre-date the
// server-side source_type default. Any memory whose metadata lacks a non-empty
// source_type gets one assigned, derived from its coarse `source` field:
//
//	user   -> user_input
//	agent  -> reflection
//	system -> reflection
//	*      -> reflection
//
// Two-phase flow:
//
//	SCAN  — Scroll ALL memories; client-side filter for those missing
//	        metadata.source_type. Group candidates by assigned source_type.
//	PATCH — for each candidate, Store.Update({"metadata": <merged>}) with the
//	        assigned source_type injected. Skipped when --dry-run (default).
//
// Flags:
//
//	--dry-run     (default true)   scan + report only, don't write
//	--batch-size  (default 20)     sleep 500ms every N patches (soft rate limit)
//	--apply       (default false)  actually write (inverse of dry-run; when set
//	                               it forces dry-run=false)
//	--qdrant-url  override cfg.QdrantURL
//	--collection  override collection name
//
// Exit codes:
//
//	0  success (including dry-run)
//	1  config / store / scroll / update hard error
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/qdrant"
)

// backfillStore is the narrow Store surface this tool needs. Defining it
// locally keeps unit tests light — a fake only needs these three methods.
type backfillStore interface {
	Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error)
	Update(ctx context.Context, id string, fields map[string]any) error
	EnsureCollection(ctx context.Context) error
}

type options struct {
	dryRun     bool
	batchSize  int
	qdrantURL  string
	collection string
}

type candidate struct {
	ID         string
	Source     string
	SourceType string
	OldMeta    map[string]any
}

type runResult struct {
	DryRun         bool           `json:"dry_run"`
	Scanned        int            `json:"scanned"`
	AlreadySet     int            `json:"already_set"`
	CandidateCount int            `json:"candidate_count"`
	BySourceType   map[string]int `json:"by_source_type"`
	Patched        int            `json:"patched"`
	Errors         []string       `json:"errors,omitempty"`
}

// frankDirectivePrefixes are case-insensitive content prefixes that indicate a
// memory records a directive/feedback from Frank (the user). When Siri records
// such a directive, Memory.Source is "agent" but the information provenance is
// the user, so the source_type should be user_input, not reflection.
var frankDirectivePrefixes = []string{
	"frank directive",
	"frank instructed",
	"frank feedback",
	"frank 指示",
	"frank 要求",
	"frank 明确",
	"frank prefers",
	"frank said",
	"frank wants",
	"frank asked",
	"frank told",
}

// isFrankDirective applies content- and tag-aware heuristics to detect a memory
// that records a directive/feedback originating from Frank (the user).
func isFrankDirective(m memory.Memory) bool {
	lower := strings.ToLower(strings.TrimSpace(m.Content))
	for _, p := range frankDirectivePrefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}

	var hasFrank, hasDirective bool
	for _, t := range m.Tags {
		switch strings.ToLower(strings.TrimSpace(t)) {
		case "frank-feedback":
			return true
		case "frank":
			hasFrank = true
		case "directive":
			hasDirective = true
		}
	}
	return hasFrank && hasDirective
}

// classifySourceType assigns a fine-grained source_type to a memory. It first
// applies content-aware heuristics (Frank directive detection) that override the
// coarse mapping, then falls back to sourceTypeForSource(m.Source). A memory
// whose Source is already "user" is left to the coarse mapping (user_input).
func classifySourceType(m memory.Memory) string {
	if m.Source != "user" && isFrankDirective(m) {
		return string(memory.SourceTypeUserInput)
	}
	return sourceTypeForSource(m.Source)
}

// sourceTypeForSource maps a coarse Memory.Source to the default fine-grained
// source_type. Pure + unit-testable.
func sourceTypeForSource(source string) string {
	switch source {
	case "user":
		return string(memory.SourceTypeUserInput)
	case "agent":
		return string(memory.SourceTypeReflection)
	case "system":
		return string(memory.SourceTypeReflection)
	default:
		return string(memory.DefaultSourceType)
	}
}

// hasSourceType reports whether metadata already carries a non-empty source_type.
func hasSourceType(md map[string]any) bool {
	if md == nil {
		return false
	}
	v, ok := md["source_type"]
	if !ok || v == nil {
		return false
	}
	s, isStr := v.(string)
	if !isStr {
		return true // present with non-string shape — treat as set, don't touch
	}
	return s != ""
}

func main() {
	opts := parseFlags()

	cfg := config.Load()
	if opts.qdrantURL != "" {
		cfg.QdrantURL = opts.qdrantURL
	}
	colName := "engram_user"
	if opts.collection != "" {
		colName = opts.collection
	}

	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		UseTLS:         cfg.QdrantUseTLS,
		CollectionName: colName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "connect qdrant: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = store.Close() }()

	res, err := run(context.Background(), store, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(res)
}

func parseFlags() options {
	fs := flag.NewFlagSet("backfill_source_type", flag.ExitOnError)
	dryRun := fs.Bool("dry-run", true, "scan + report only, don't write")
	apply := fs.Bool("apply", false, "actually write (forces dry-run=false)")
	batch := fs.Int("batch-size", 20, "sleep 500ms every N patches (soft rate limit)")
	url := fs.String("qdrant-url", "", "override QDRANT_URL env")
	coll := fs.String("collection", "", "override collection name")
	_ = fs.Parse(os.Args[1:])

	if *batch <= 0 {
		*batch = 20
	}
	dr := *dryRun
	if *apply {
		dr = false
	}
	return options{
		dryRun:     dr,
		batchSize:  *batch,
		qdrantURL:  *url,
		collection: *coll,
	}
}

// run executes SCAN then PATCH. Extracted from main so tests can drive it
// against a fake backfillStore without hitting Qdrant.
func run(ctx context.Context, store backfillStore, opts options) (*runResult, error) {
	if err := store.EnsureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure collection: %w", err)
	}

	res := &runResult{
		DryRun:       opts.dryRun,
		BySourceType: map[string]int{},
	}

	candidates, scanned, alreadySet, err := scan(ctx, store)
	if err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}
	res.Scanned = scanned
	res.AlreadySet = alreadySet
	res.CandidateCount = len(candidates)
	for _, c := range candidates {
		res.BySourceType[c.SourceType]++
	}

	fmt.Fprintf(os.Stderr, "[SCAN] total scrolled: %d\n", scanned)
	fmt.Fprintf(os.Stderr, "[SCAN] already have source_type: %d\n", alreadySet)
	fmt.Fprintf(os.Stderr, "[SCAN] candidates (missing source_type): %d\n", len(candidates))
	for _, st := range sortedKeys(res.BySourceType) {
		fmt.Fprintf(os.Stderr, "  %s: %d\n", st, res.BySourceType[st])
	}

	if opts.dryRun {
		fmt.Fprintf(os.Stderr, "\n[SUMMARY]\ncandidates=%d  would_patch=%d  (DRY RUN)\n",
			len(candidates), len(candidates))
		return res, nil
	}

	patched, errs := patch(ctx, store, candidates, opts.batchSize)
	res.Patched = patched
	res.Errors = errs

	fmt.Fprintf(os.Stderr, "\n[SUMMARY]\ncandidates=%d  patched=%d  errors=%d\n",
		len(candidates), patched, len(errs))

	return res, nil
}

// scan enumerates all memories and returns those missing metadata.source_type.
func scan(ctx context.Context, store backfillStore) ([]candidate, int, int, error) {
	var (
		scanned    int
		alreadySet int
		candidates []candidate
		offset     string
	)

	for {
		batch, next, err := store.Scroll(ctx, memory.ScrollOptions{
			Limit:  200,
			Offset: offset,
		})
		if err != nil {
			return nil, scanned, alreadySet, fmt.Errorf("scroll: %w", err)
		}
		if len(batch) == 0 {
			break
		}
		scanned += len(batch)

		for _, m := range batch {
			if hasSourceType(m.Metadata) {
				alreadySet++
				continue
			}
			candidates = append(candidates, candidate{
				ID:         m.ID,
				Source:     m.Source,
				SourceType: classifySourceType(m),
				OldMeta:    m.Metadata,
			})
		}

		if next == "" || next == offset {
			break
		}
		offset = next
	}

	return candidates, scanned, alreadySet, nil
}

// patch writes metadata updates injecting the assigned source_type.
func patch(ctx context.Context, store backfillStore, cs []candidate, batchSize int) (int, []string) {
	var (
		patched int
		errs    []string
	)
	for i, c := range cs {
		newMeta := map[string]any{}
		for k, v := range c.OldMeta {
			newMeta[k] = v
		}
		newMeta["source_type"] = c.SourceType

		if err := store.Update(ctx, c.ID, map[string]any{
			"metadata": newMeta,
		}); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", c.ID, err))
			fmt.Fprintf(os.Stderr, "  update %s FAILED: %v\n", c.ID, err)
			continue
		}
		patched++

		if batchSize > 0 && (i+1)%batchSize == 0 {
			time.Sleep(500 * time.Millisecond)
		}
	}
	return patched, errs
}

func sortedKeys(m map[string]int) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
