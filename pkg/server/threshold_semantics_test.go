package server

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
)

// scriptedStore embeds mockStore but returns a fixed set of Search results whose
// RAW COSINE scores are chosen by the test. It lets the R1 tests prove that the
// memory_update / memory_delete threshold is compared against the raw cosine
// that Store.Search returns — never against the composite relevance score
// (memory.Score), which is applied ONLY on the read path (rerankResults) and
// routinely exceeds 1.0. Insert/Delete/etc. fall through to the embedded
// mockStore so the write side behaves normally.
type scriptedStore struct {
	*mockStore
	results []memory.ScoredMemory
}

func (s *scriptedStore) Search(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	limit := opts.Limit
	if limit <= 0 || limit > len(s.results) {
		limit = len(s.results)
	}
	out := make([]memory.ScoredMemory, limit)
	copy(out, s.results[:limit])
	return out, nil
}

// ── R1: delete/update thresholds compare RAW COSINE, never the composite ──────

func TestDeleteThreshold_RawCosineNotComposite(t *testing.T) {
	cases := []struct {
		name        string
		rawCosine   float64
		content     string
		wantStatus  string
		wantDeleted int
	}{
		{
			// A composite score > 1.0 must NOT auto-pass a 0.92 cosine threshold:
			// raw cosine 0.87 < 0.92 => nothing deleted.
			name:        "composite_over_one_does_not_autopass_0.92",
			rawCosine:   0.87,
			content:     "short note",
			wantStatus:  "no_matches",
			wantDeleted: 0,
		},
		{
			// A genuinely near-identical long item (raw cosine 0.95 >= 0.92) DOES match.
			name:        "near_identical_long_item_matches_0.92",
			rawCosine:   0.95,
			content:     strings.Repeat("a long detailed memory sentence with plenty of tokens. ", 20),
			wantStatus:  "deleted",
			wantDeleted: 1,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, store := newTestServer()
			m := memory.New(tc.content, memory.WithType(memory.TypeEvent), memory.WithImportance(5))
			m.ID = "target-1"
			vec, _ := srv.embedder.Embed(context.Background(), tc.content)
			if err := store.Insert(context.Background(), m, vec); err != nil {
				t.Fatalf("insert: %v", err)
			}

			// Setup invariant: the COMPOSITE score for this item exceeds 1.0, so
			// if the delete path (wrongly) used composite it would auto-pass any
			// cosine threshold. The raw cosine is what must actually govern.
			composite := memory.Score(m, tc.rawCosine, srv.weights, srv.decay, srv.evaporationConfig())
			if composite <= 1.0 {
				t.Fatalf("setup: composite=%.3f should exceed 1.0", composite)
			}

			srv.store = &scriptedStore{mockStore: store, results: []memory.ScoredMemory{
				{Memory: *m, Score: tc.rawCosine},
			}}

			result, err := srv.handleDelete(context.Background(), toolReq(map[string]any{
				"query":                "find the target",
				"similarity_threshold": 0.92,
				"limit":                float64(1),
			}))
			if err != nil {
				t.Fatalf("handleDelete: %v", err)
			}
			var resp struct {
				Status       string `json:"status"`
				DeletedCount int    `json:"deleted_count"`
			}
			if err := json.Unmarshal([]byte(extractText(result)), &resp); err != nil {
				t.Fatalf("parse: %v raw=%s", err, extractText(result))
			}
			if resp.Status != tc.wantStatus || resp.DeletedCount != tc.wantDeleted {
				t.Fatalf("raw=%.2f composite=%.2f thr=0.92: status=%q deleted=%d, want %q/%d",
					tc.rawCosine, composite, resp.Status, resp.DeletedCount, tc.wantStatus, tc.wantDeleted)
			}
		})
	}
}

func TestUpdateThreshold_RawCosineNotComposite(t *testing.T) {
	cases := []struct {
		name       string
		rawCosine  float64
		content    string
		wantStatus string
	}{
		{
			name:       "composite_over_one_does_not_autopass_0.92",
			rawCosine:  0.87,
			content:    "short note",
			wantStatus: "no_target_matched",
		},
		{
			name:       "near_identical_long_item_matches_0.92",
			rawCosine:  0.95,
			content:    strings.Repeat("a long detailed memory sentence with plenty of tokens. ", 20),
			wantStatus: "updated",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, store := newTestServer()
			m := memory.New(tc.content, memory.WithType(memory.TypeEvent), memory.WithImportance(5))
			m.ID = "u-target"
			vec, _ := srv.embedder.Embed(context.Background(), tc.content)
			if err := store.Insert(context.Background(), m, vec); err != nil {
				t.Fatalf("insert: %v", err)
			}
			composite := memory.Score(m, tc.rawCosine, srv.weights, srv.decay, srv.evaporationConfig())
			if composite <= 1.0 {
				t.Fatalf("setup: composite=%.3f should exceed 1.0", composite)
			}

			srv.store = &scriptedStore{mockStore: store, results: []memory.ScoredMemory{
				{Memory: *m, Score: tc.rawCosine},
			}}

			result, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{
				"old_content":          "find the target",
				"new_content":          "the corrected content",
				"similarity_threshold": 0.92,
			}))
			if err != nil {
				t.Fatalf("handleUpdate: %v", err)
			}
			if got := mustStatus(t, result); got != tc.wantStatus {
				t.Fatalf("raw=%.2f composite=%.2f thr=0.92: status=%q, want %q",
					tc.rawCosine, composite, got, tc.wantStatus)
			}
		})
	}
}

// ── R2: explicit id short-circuits memory_update / memory_delete to the exact ─
//        match (by-id) path, skipping semantic search + the cosine threshold. ──

func TestDelete_IDPriority_ShortCircuits(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "del-target", "content to remove", memory.TypeEvent)

	// id alone (no query, no threshold) drives an exact-match soft delete.
	result, err := srv.handleDelete(context.Background(), toolReq(map[string]any{
		"id": "del-target",
	}))
	if err != nil {
		t.Fatalf("handleDelete: %v", err)
	}
	if got := mustStatus(t, result); got != "archived" {
		t.Fatalf("status=%q, want archived (id short-circuit soft delete)", got)
	}
	got, _ := store.SearchByIDs(context.Background(), []string{"del-target"})
	if len(got) != 1 || got[0].LifecycleStatus != memory.LifecycleArchived {
		t.Fatalf("id delete must soft-archive in place: %+v", got)
	}
}

func TestUpdate_IDPriority_ShortCircuits_SkipsThresholdGuard(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "upd-target", "old content", memory.TypeEvent)

	// id + new_content; deliberately OMIT similarity_threshold. The semantic
	// path would reject threshold < 0.85, but the id path skips that guard and
	// the cosine search entirely.
	result, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{
		"id":          "upd-target",
		"new_content": "corrected content",
		"type":        "insight",
	}))
	if err != nil {
		t.Fatalf("handleUpdate: %v", err)
	}
	if result.IsError {
		t.Fatalf("id update must not hit the threshold guard: %s", extractText(result))
	}
	if got := mustStatus(t, result); got != "updated" {
		t.Fatalf("status=%q, want updated", got)
	}
	got, _ := store.SearchByIDs(context.Background(), []string{"upd-target"})
	if len(got) != 1 || got[0].Content != "corrected content" || got[0].ID != "upd-target" {
		t.Fatalf("id update must replace content and preserve id: %+v", got)
	}
}

// R2 isolation: the id short-circuit must preserve the byIDAllowed caller guard
// (SearchByIDs fans out across all stores) so an isolated caller cannot reach a
// cross-collection id via memory_update / memory_delete.
func TestSemanticTools_IDPriority_IsolatedCrossCollection_Rejected(t *testing.T) {
	cases := []struct {
		name string
		call func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error)
	}{
		{"memory_delete", func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error) {
			return srv.handleDelete(ctx, toolReq(map[string]any{"id": id}))
		}},
		{"memory_update", func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error) {
			return srv.handleUpdate(ctx, toolReq(map[string]any{"id": id, "new_content": "hijacked"}))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, store := newTestServer()
			seedMemoryInCollection(t, srv, store, "foreign", "foreign content", collection.CollectionUser)
			ctx := WithCallerType(context.Background(), "pigo") // own = engram_pigo
			beforeMut := store.mutations()

			result, err := tc.call(srv, ctx, "foreign")
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if got := mustStatus(t, result); got != "not_found" {
				t.Fatalf("status=%q, want not_found (cross-collection id must be rejected)", got)
			}
			if store.mutations() != beforeMut {
				t.Fatalf("cross-collection id op must be a no-op (mutations %d -> %d)", beforeMut, store.mutations())
			}
			// Target untouched.
			got, _ := store.SearchByIDs(context.Background(), []string{"foreign"})
			if len(got) != 1 || got[0].Content != "foreign content" || got[0].LifecycleStatus == memory.LifecycleArchived {
				t.Fatalf("target was mutated: %+v", got)
			}
		})
	}
}
