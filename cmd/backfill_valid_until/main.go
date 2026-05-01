// cmd/backfill_valid_until — R-S2
//
// Backfills the `valid_until` field onto reflection-origin insights that were
// stored before commit 1b3e482 wired WithValidUntil into the reflection engine.
// Those insights have valid_until=0 (missing key in Qdrant) but should carry a
// 30-day TTL computed from their created_at timestamp.
//
// Three-phase flow:
//
//	SCAN   — enumerate type=insight memories with source:reflection tag (or
//	         source=system + reflection_source_ids); filter for valid_until == 0.
//	PATCH  — for each candidate, set valid_until = created_at + 30d.
//	         Skipped when --dry-run=true (default).
//	VERIFY — re-fetch patched IDs; assert valid_until > 0 for each.
//
// Flags:
//
//	--dry-run     (default true)   scan + report only
//	--verify      (default false)  run VERIFY phase after PATCH
//	--qdrant-url  override cfg.QdrantURL
//	--collection  override cfg.CollectionName
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/qdrant"
)

const thirtyDays = 30 * 24 * 3600

type backfillStore interface {
	Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error)
	Update(ctx context.Context, id string, fields map[string]any) error
	SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error)
	EnsureCollection(ctx context.Context) error
}

type options struct {
	dryRun     bool
	verify     bool
	qdrantURL  string
	collection string
}

type candidate struct {
	ID           string  `json:"id"`
	ShortContent string  `json:"short_content"`
	CreatedAt    float64 `json:"created_at"`
	NewValidUntil float64 `json:"new_valid_until"`
}

type runResult struct {
	DryRun         bool        `json:"dry_run"`
	Scanned        int         `json:"scanned"`
	AlreadySet     int         `json:"already_set"`
	CandidateCount int         `json:"candidate_count"`
	Patched        int         `json:"patched"`
	Errors         []string    `json:"errors,omitempty"`
	VerifyPassed   bool        `json:"verify_passed,omitempty"`
	VerifyFailed   []string    `json:"verify_failed,omitempty"`
}

func main() {
	opts := parseFlags()

	cfg := config.Load()
	if opts.qdrantURL != "" {
		cfg.QdrantURL = opts.qdrantURL
	}
	if opts.collection != "" {
		cfg.CollectionName = opts.collection
	}

	store, err := qdrant.New(qdrant.Config{
		URL:            cfg.QdrantURL,
		APIKey:         cfg.QdrantAPIKey,
		UseTLS:         cfg.QdrantUseTLS,
		CollectionName: cfg.CollectionName,
		Dimension:      uint64(cfg.EmbeddingDimension),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "connect qdrant: %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	res, err := run(context.Background(), store, opts)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(res)

	if opts.verify && !opts.dryRun && !res.VerifyPassed {
		os.Exit(2)
	}
}

func parseFlags() options {
	fs := flag.NewFlagSet("backfill_valid_until", flag.ExitOnError)
	dryRun := fs.Bool("dry-run", true, "scan + report only")
	verify := fs.Bool("verify", false, "run VERIFY phase after PATCH")
	url := fs.String("qdrant-url", "", "override QDRANT_URL")
	coll := fs.String("collection", "", "override collection name")
	_ = fs.Parse(os.Args[1:])

	return options{
		dryRun:     *dryRun,
		verify:     *verify,
		qdrantURL:  *url,
		collection: *coll,
	}
}

func run(ctx context.Context, store backfillStore, opts options) (*runResult, error) {
	if err := store.EnsureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure collection: %w", err)
	}

	res := &runResult{DryRun: opts.dryRun}

	candidates, scanned, alreadySet, err := scan(ctx, store)
	if err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}
	res.Scanned = scanned
	res.AlreadySet = alreadySet
	res.CandidateCount = len(candidates)

	fmt.Fprintf(os.Stderr, "[SCAN] type=insight with reflection origin: %d\n", scanned)
	fmt.Fprintf(os.Stderr, "[SCAN] already have valid_until: %d\n", alreadySet)
	fmt.Fprintf(os.Stderr, "[SCAN] candidates (missing valid_until): %d\n", len(candidates))
	for _, c := range candidates {
		fmt.Fprintf(os.Stderr, "  %s  %q  created_at=%.0f  new_valid_until=%.0f\n",
			c.ID, c.ShortContent, c.CreatedAt, c.NewValidUntil)
	}

	if opts.dryRun {
		return res, nil
	}

	patched, errs := patch(ctx, store, candidates)
	res.Patched = patched
	res.Errors = errs

	fmt.Fprintf(os.Stderr, "\n[PATCH] patched=%d errors=%d\n", patched, len(errs))

	if opts.verify && patched > 0 {
		ids := make([]string, len(candidates))
		for i, c := range candidates {
			ids[i] = c.ID
		}
		ms, err := store.SearchByIDs(ctx, ids)
		if err != nil {
			return nil, fmt.Errorf("verify fetch: %w", err)
		}
		res.VerifyPassed = true
		for _, m := range ms {
			if m.ValidUntil <= 0 {
				res.VerifyPassed = false
				res.VerifyFailed = append(res.VerifyFailed, m.ID)
			}
		}
		fmt.Fprintf(os.Stderr, "\n[VERIFY] passed=%v failed=%v\n", res.VerifyPassed, res.VerifyFailed)
	}

	return res, nil
}

func scan(ctx context.Context, store backfillStore) ([]candidate, int, int, error) {
	var (
		scanned    int
		alreadySet int
		candidates []candidate
		offset     string
	)

	for {
		batch, next, err := store.Scroll(ctx, memory.ScrollOptions{
			Limit: 200,
			Filters: []memory.Filter{
				{Field: "type", Op: memory.OpEq, Value: "insight"},
			},
			Offset: offset,
		})
		if err != nil {
			return nil, 0, 0, fmt.Errorf("scroll: %w", err)
		}
		if len(batch) == 0 {
			break
		}

		for _, m := range batch {
			if !isReflectionOrigin(m) {
				continue
			}
			scanned++
			if m.ValidUntil > 0 {
				alreadySet++
				continue
			}
			candidates = append(candidates, candidate{
				ID:            m.ID,
				ShortContent:  shortContent(m.Content, 80),
				CreatedAt:     m.CreatedAt,
				NewValidUntil: m.CreatedAt + thirtyDays,
			})
		}

		if next == "" || next == offset {
			break
		}
		offset = next
	}

	return candidates, scanned, alreadySet, nil
}

func patch(ctx context.Context, store backfillStore, cs []candidate) (int, []string) {
	var (
		patched int
		errs    []string
	)
	for _, c := range cs {
		if err := store.Update(ctx, c.ID, map[string]any{
			"valid_until": c.NewValidUntil,
		}); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", c.ID, err))
			continue
		}
		patched++
	}
	return patched, errs
}

func isReflectionOrigin(m memory.Memory) bool {
	if m.Source == "system" {
		return true
	}
	if m.Metadata != nil {
		if v, ok := m.Metadata["reflection_source_ids"]; ok && v != nil {
			switch s := v.(type) {
			case []any:
				return len(s) > 0
			case []string:
				return len(s) > 0
			}
		}
	}
	for _, t := range m.Tags {
		if t == "source:reflection" {
			return true
		}
	}
	return false
}

func shortContent(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
