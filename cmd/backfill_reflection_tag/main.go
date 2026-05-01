// cmd/backfill_reflection_tag — W17 T8
//
// Backfills the `source:reflection` tag onto legacy reflection-origin insights
// that pre-date v1.1 (commit 1b3e482, 2026-04-17). From that commit onward,
// every Reflection Engine output carries the tag as an invariant; older
// insights have source="system" and/or metadata["reflection_source_ids"]
// populated, but no matching tag.
//
// Design: /vault/Engram/w17-t8-backfill-script-design.md
//
// Three-phase flow:
//
//	SCAN   — enumerate all type=insight via Scroll; client-side filter on
//	         (source==system || metadata.reflection_source_ids non-empty)
//	         AND tags does NOT contain "source:reflection". Print candidates.
//	PATCH  — for each candidate, read oldTags + append "source:reflection" +
//	         write back via Store.Update({"tags": newTags}). Skipped when
//	         --dry-run=true (default).
//	VERIFY — optional. Count tags contains source:reflection before + after,
//	         assert delta == patched_count. Sample 3 random patched IDs and
//	         re-fetch to eyeball-check tag list.
//
// Flags:
//
//	--dry-run     (default true)   scan + report only, don't write
//	--batch-size  (default 20)     sleep 500ms every N patches (soft rate limit)
//	--verify      (default false)  run VERIFY phase after PATCH
//	--qdrant-url  override cfg.QdrantURL
//	--collection  override cfg.CollectionName
//
// Exit codes:
//
//	0  success (including dry-run with candidates)
//	1  config / store / scroll / update hard error
//	2  VERIFY assertion mismatch
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/qdrant"
)

const sourceReflectionTag = "source:reflection"

// backfillStore is the narrow Store surface this tool needs. Defining it
// locally (instead of binding to memory.Store) keeps the unit tests light —
// a fake implementation only needs these four methods.
type backfillStore interface {
	Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error)
	Update(ctx context.Context, id string, fields map[string]any) error
	SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error)
	EnsureCollection(ctx context.Context) error
}

type options struct {
	dryRun     bool
	batchSize  int
	verify     bool
	qdrantURL  string
	collection string
}

type candidate struct {
	ID              string
	ShortContent    string
	Source          string
	TagCount        int
	HasReflectionSourceIDs bool
	OldTags         []string
}

type runResult struct {
	DryRun            bool        `json:"dry_run"`
	Verify            bool        `json:"verify"`
	Scanned           int         `json:"scanned"`
	AlreadyTagged     int         `json:"already_tagged"`
	CandidateCount    int         `json:"candidate_count"`
	Patched           int         `json:"patched"`
	Errors            []string    `json:"errors,omitempty"`
	VerifyPreCount    int         `json:"verify_pre_count,omitempty"`
	VerifyPostCount   int         `json:"verify_post_count,omitempty"`
	VerifyExpectedDelta int       `json:"verify_expected_delta,omitempty"`
	VerifyActualDelta int         `json:"verify_actual_delta,omitempty"`
	VerifyPassed      bool        `json:"verify_passed,omitempty"`
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
	fs := flag.NewFlagSet("backfill_reflection_tag", flag.ExitOnError)
	dryRun := fs.Bool("dry-run", true, "scan + report only, don't write")
	batch := fs.Int("batch-size", 20, "sleep 500ms every N patches (soft rate limit)")
	verify := fs.Bool("verify", false, "run VERIFY phase after PATCH")
	url := fs.String("qdrant-url", "", "override QDRANT_URL env")
	coll := fs.String("collection", "", "override collection name")
	_ = fs.Parse(os.Args[1:])

	if *batch <= 0 {
		*batch = 20
	}
	return options{
		dryRun:     *dryRun,
		batchSize:  *batch,
		verify:     *verify,
		qdrantURL:  *url,
		collection: *coll,
	}
}

// run executes all three phases. Extracted from main so tests can drive it
// against a fake backfillStore without hitting Qdrant.
func run(ctx context.Context, store backfillStore, opts options) (*runResult, error) {
	if err := store.EnsureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure collection: %w", err)
	}

	res := &runResult{
		DryRun: opts.dryRun,
		Verify: opts.verify,
	}

	// ── Optional pre-verify count ────────────────────────────────────────
	if opts.verify && !opts.dryRun {
		pre, err := countTaggedReflection(ctx, store)
		if err != nil {
			return nil, fmt.Errorf("verify pre-count: %w", err)
		}
		res.VerifyPreCount = pre
	}

	// ── SCAN ─────────────────────────────────────────────────────────────
	candidates, scanned, alreadyTagged, err := scan(ctx, store)
	if err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}
	res.Scanned = scanned
	res.AlreadyTagged = alreadyTagged
	res.CandidateCount = len(candidates)

	fmt.Fprintf(os.Stderr, "[SCAN] total type=insight scrolled: %d\n", scanned)
	fmt.Fprintf(os.Stderr, "[SCAN] already tagged (%s): %d\n", sourceReflectionTag, alreadyTagged)
	fmt.Fprintf(os.Stderr, "[SCAN] candidates (missing tag): %d\n", len(candidates))
	if len(candidates) > 0 {
		fmt.Fprintln(os.Stderr, "\n[CANDIDATES]")
		for _, c := range candidates {
			fmt.Fprintf(os.Stderr,
				"  %s  %q  source=%s  tags=%d  has_src_ids=%v\n",
				c.ID, c.ShortContent, c.Source, c.TagCount, c.HasReflectionSourceIDs)
		}
	}

	if opts.dryRun {
		fmt.Fprintf(os.Stderr, "\n[SUMMARY]\ncandidates=%d  would_patch=%d  already_tagged=%d  (DRY RUN)\n",
			len(candidates), len(candidates), alreadyTagged)
		return res, nil
	}

	// ── PATCH ────────────────────────────────────────────────────────────
	patched, errs := patch(ctx, store, candidates, opts.batchSize)
	res.Patched = patched
	res.Errors = errs

	fmt.Fprintf(os.Stderr, "\n[SUMMARY]\ncandidates=%d  patched=%d  errors=%d\n",
		len(candidates), patched, len(errs))

	// ── VERIFY ───────────────────────────────────────────────────────────
	if opts.verify {
		post, err := countTaggedReflection(ctx, store)
		if err != nil {
			return nil, fmt.Errorf("verify post-count: %w", err)
		}
		res.VerifyPostCount = post
		res.VerifyExpectedDelta = patched
		res.VerifyActualDelta = post - res.VerifyPreCount
		res.VerifyPassed = res.VerifyActualDelta == res.VerifyExpectedDelta

		fmt.Fprintf(os.Stderr, "\n[VERIFY]\npre=%d post=%d expected_delta=%d actual_delta=%d passed=%v\n",
			res.VerifyPreCount, post, patched, res.VerifyActualDelta, res.VerifyPassed)

		// Sample up to 3 patched IDs and eyeball-check tag list.
		if patched > 0 {
			sample := sampleIDs(candidates, 3)
			ms, err := store.SearchByIDs(ctx, sample)
			if err == nil {
				fmt.Fprintln(os.Stderr, "\n[VERIFY-SAMPLE]")
				for _, m := range ms {
					fmt.Fprintf(os.Stderr, "  %s tags=%v\n", m.ID, m.Tags)
				}
			}
		}
	}

	return res, nil
}

// scan enumerates all type=insight memories and returns the candidates that
// need the `source:reflection` tag.
//
// Candidate rule (all must hold):
//  1. type == "insight"  — enforced server-side via Filter
//  2. source == "system"  OR  metadata["reflection_source_ids"] non-empty
//  3. tags does not contain "source:reflection"
func scan(ctx context.Context, store backfillStore) ([]candidate, int, int, error) {
	var (
		scanned       int
		alreadyTagged int
		candidates    []candidate
		offset        string
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
			return nil, scanned, alreadyTagged, fmt.Errorf("scroll: %w", err)
		}
		if len(batch) == 0 {
			break
		}
		scanned += len(batch)

		for _, m := range batch {
			hasSrcIDs := hasReflectionSourceIDs(m.Metadata)
			// Condition 2: source fingerprint
			if m.Source != "system" && !hasSrcIDs {
				continue
			}
			// Condition 3: missing tag
			if containsTag(m.Tags, sourceReflectionTag) {
				alreadyTagged++
				continue
			}
			candidates = append(candidates, candidate{
				ID:                     m.ID,
				ShortContent:           shortContent(m.Content, 80),
				Source:                 m.Source,
				TagCount:               len(m.Tags),
				HasReflectionSourceIDs: hasSrcIDs,
				OldTags:                append([]string(nil), m.Tags...),
			})
		}

		if next == "" || next == offset {
			break
		}
		offset = next
	}

	return candidates, scanned, alreadyTagged, nil
}

// patch writes tag updates for each candidate.
func patch(ctx context.Context, store backfillStore, cs []candidate, batchSize int) (int, []string) {
	var (
		patched int
		errs    []string
	)
	for i, c := range cs {
		newTags := appendTag(c.OldTags, sourceReflectionTag)
		tagValues := make([]any, len(newTags))
		for j, t := range newTags {
			tagValues[j] = t
		}
		if err := store.Update(ctx, c.ID, map[string]any{
			"tags": tagValues,
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

// countTaggedReflection returns the number of memories whose tags contain
// "source:reflection". Used for the VERIFY delta assertion.
func countTaggedReflection(ctx context.Context, store backfillStore) (int, error) {
	var (
		total  int
		offset string
	)
	for {
		batch, next, err := store.Scroll(ctx, memory.ScrollOptions{
			Limit: 200,
			Filters: []memory.Filter{
				{Field: "tags", Op: memory.OpIn, Value: []string{sourceReflectionTag}},
			},
			Offset: offset,
		})
		if err != nil {
			return 0, err
		}
		if len(batch) == 0 {
			break
		}
		total += len(batch)
		if next == "" || next == offset {
			break
		}
		offset = next
	}
	return total, nil
}

// ── helpers ─────────────────────────────────────────────────────────────────

func containsTag(tags []string, target string) bool {
	for _, t := range tags {
		if t == target {
			return true
		}
	}
	return false
}

// appendTag returns a new slice with `tag` appended iff not already present.
// Never mutates the input — same semantics as ensureSourceReflectionTag
// in pkg/reflection/engine.go.
func appendTag(tags []string, tag string) []string {
	if containsTag(tags, tag) {
		return append([]string(nil), tags...)
	}
	out := make([]string, 0, len(tags)+1)
	out = append(out, tags...)
	out = append(out, tag)
	return out
}

// hasReflectionSourceIDs reports whether metadata contains a non-empty
// "reflection_source_ids" value. Accepts []any (Qdrant JSON decode) and
// []string (in-process fakes).
func hasReflectionSourceIDs(md map[string]any) bool {
	if md == nil {
		return false
	}
	v, ok := md["reflection_source_ids"]
	if !ok || v == nil {
		return false
	}
	switch s := v.(type) {
	case []any:
		return len(s) > 0
	case []string:
		return len(s) > 0
	case string:
		return s != ""
	default:
		return true // present with unknown shape — count as populated
	}
}

func shortContent(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func sampleIDs(cs []candidate, n int) []string {
	if len(cs) <= n {
		ids := make([]string, len(cs))
		for i, c := range cs {
			ids[i] = c.ID
		}
		return ids
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	perm := r.Perm(len(cs))[:n]
	ids := make([]string, n)
	for i, p := range perm {
		ids[i] = cs[p].ID
	}
	return ids
}
