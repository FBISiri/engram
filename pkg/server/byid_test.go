package server

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
)

// seedMemory inserts a memory directly into the store with a vector equal to the
// embedding of its content, so a semantic search for the same content scores 1.0
// (cosine). Bypasses handleAdd's dedup so we can seed duplicates deliberately.
func seedMemory(t *testing.T, srv *Server, store *mockStore, id, content string, mtype memory.MemoryType) {
	t.Helper()
	vec, err := srv.embedder.Embed(context.Background(), content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	m := memory.New(content, memory.WithType(mtype))
	m.ID = id
	if err := store.Insert(context.Background(), m, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}
}

func toolReq(args map[string]any) mcp.CallToolRequest {
	return mcp.CallToolRequest{Params: mcp.CallToolParams{Arguments: args}}
}

// seedMemoryInCollection is seedMemory but stamps the memory's Collection, used
// by the isolation tests.
func seedMemoryInCollection(t *testing.T, srv *Server, store *mockStore, id, content, coll string) {
	t.Helper()
	vec, err := srv.embedder.Embed(context.Background(), content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	m := memory.New(content, memory.WithType(memory.TypeEvent))
	m.ID = id
	m.Collection = coll
	if err := store.Insert(context.Background(), m, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}
}

func mustStatus(t *testing.T, result *mcp.CallToolResult) string {
	t.Helper()
	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &resp); err != nil {
		t.Fatalf("parse status: %v\nraw: %s", err, extractText(result))
	}
	return resp.Status
}

// ── R1: handleUpdate must never silently degrade to Insert ────────────────────

func TestUpdate_ZeroMatch_NoTargetMatched_NoInsert(t *testing.T) {
	srv, store := newTestServer()
	// Empty store => the threshold loop matches nothing.
	before := store.count()

	result, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{
		"old_content":          "nonexistent memory to correct",
		"new_content":          "the correction",
		"similarity_threshold": 0.92,
	}))
	if err != nil {
		t.Fatalf("handleUpdate: %v", err)
	}
	if result.IsError {
		t.Fatalf("expected structured result, got tool error: %s", extractText(result))
	}
	if got := mustStatus(t, result); got != "no_target_matched" {
		t.Fatalf("status = %q, want no_target_matched", got)
	}
	if after := store.count(); after != before {
		t.Fatalf("store count changed %d -> %d: update must NOT insert on zero match", before, after)
	}
}

func TestUpdate_AmbiguousMatch_NoDeleteNoInsert(t *testing.T) {
	srv, store := newTestServer()
	// Two memories with identical content => identical vectors => both score 1.0.
	seedMemory(t, srv, store, "id-a", "duplicate target content", memory.TypeEvent)
	seedMemory(t, srv, store, "id-b", "duplicate target content", memory.TypeEvent)
	before := store.count()

	result, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{
		"old_content":          "duplicate target content",
		"new_content":          "the correction",
		"similarity_threshold": 0.92,
	}))
	if err != nil {
		t.Fatalf("handleUpdate: %v", err)
	}
	if got := mustStatus(t, result); got != "ambiguous_match" {
		t.Fatalf("status = %q, want ambiguous_match", got)
	}
	if after := store.count(); after != before {
		t.Fatalf("store count changed %d -> %d: ambiguous update must NOT delete/insert", before, after)
	}
	// Candidate list should carry both ids for the caller to disambiguate.
	var resp struct {
		MatchCount int `json:"match_count"`
		Candidates []struct {
			ID string `json:"id"`
		} `json:"candidates"`
	}
	_ = json.Unmarshal([]byte(extractText(result)), &resp)
	if resp.MatchCount != 2 || len(resp.Candidates) != 2 {
		t.Fatalf("expected 2 candidates, got match_count=%d candidates=%d", resp.MatchCount, len(resp.Candidates))
	}
}

func TestUpdate_SingleMatch_DeleteThenInsert(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "id-1", "single target content", memory.TypeEvent)
	before := store.count()

	result, err := srv.handleUpdate(context.Background(), toolReq(map[string]any{
		"old_content":          "single target content",
		"new_content":          "the corrected content",
		"similarity_threshold": 0.92,
	}))
	if err != nil {
		t.Fatalf("handleUpdate: %v", err)
	}
	if got := mustStatus(t, result); got != "updated" {
		t.Fatalf("status = %q, want updated", got)
	}
	if after := store.count(); after != before {
		t.Fatalf("store count changed %d -> %d: single update should delete 1 + insert 1", before, after)
	}
	// The old id must be gone and the new content present.
	if _, err := store.SearchByIDs(context.Background(), []string{"id-1"}); err == nil {
		got, _ := store.SearchByIDs(context.Background(), []string{"id-1"})
		if len(got) != 0 {
			t.Fatalf("old memory id-1 should have been deleted")
		}
	}
	found := false
	for _, m := range store.all() {
		if m.Content == "the corrected content" {
			found = true
		}
	}
	if !found {
		t.Fatalf("new content not inserted")
	}
}

// ── R1: handleDelete zero-match is an explicit no-op ──────────────────────────

func TestDelete_ZeroMatch_ExplicitNoOp(t *testing.T) {
	srv, _ := newTestServer()
	result, err := srv.handleDelete(context.Background(), toolReq(map[string]any{
		"query":                "nothing matches this",
		"similarity_threshold": 0.92,
		"limit":                float64(1),
	}))
	if err != nil {
		t.Fatalf("handleDelete: %v", err)
	}
	var resp struct {
		Status       string `json:"status"`
		Message      string `json:"message"`
		DeletedCount int    `json:"deleted_count"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.Status != "no_matches" || resp.DeletedCount != 0 {
		t.Fatalf("status=%q deleted_count=%d, want no_matches/0", resp.Status, resp.DeletedCount)
	}
	if resp.Message == "" {
		t.Fatalf("no-op delete must carry a clear message")
	}
}

// ── R2: by-id tools bypass cosine entirely ────────────────────────────────────

func TestDeleteByID_SoftArchives_RegardlessOfCosine(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "keep-me", "arbitrary content", memory.TypeEvent)

	result, err := srv.handleDeleteByID(context.Background(), toolReq(map[string]any{"id": "keep-me"}))
	if err != nil {
		t.Fatalf("handleDeleteByID: %v", err)
	}
	if got := mustStatus(t, result); got != "archived" {
		t.Fatalf("status = %q, want archived", got)
	}
	// Soft delete: still present, but lifecycle_status=archived.
	got, _ := store.SearchByIDs(context.Background(), []string{"keep-me"})
	if len(got) != 1 {
		t.Fatalf("memory should still exist after soft delete")
	}
	if got[0].LifecycleStatus != memory.LifecycleArchived {
		t.Fatalf("lifecycle_status = %q, want archived", got[0].LifecycleStatus)
	}
}

func TestDeleteByID_NotFound(t *testing.T) {
	srv, _ := newTestServer()
	result, err := srv.handleDeleteByID(context.Background(), toolReq(map[string]any{"id": "ghost"}))
	if err != nil {
		t.Fatalf("handleDeleteByID: %v", err)
	}
	if got := mustStatus(t, result); got != "not_found" {
		t.Fatalf("status = %q, want not_found", got)
	}
}

func TestUpdateByID_Replaces_PreservesID_RegardlessOfCosine(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "fixed-id", "old content here", memory.TypeEvent)

	result, err := srv.handleUpdateByID(context.Background(), toolReq(map[string]any{
		"id":      "fixed-id",
		"content": "brand new content",
		"type":    "insight",
	}))
	if err != nil {
		t.Fatalf("handleUpdateByID: %v", err)
	}
	if got := mustStatus(t, result); got != "updated" {
		t.Fatalf("status = %q, want updated", got)
	}
	got, _ := store.SearchByIDs(context.Background(), []string{"fixed-id"})
	if len(got) != 1 {
		t.Fatalf("memory should still exist under same id")
	}
	if got[0].Content != "brand new content" {
		t.Fatalf("content = %q, want replaced", got[0].Content)
	}
	if got[0].ID != "fixed-id" {
		t.Fatalf("id changed to %q, must be preserved", got[0].ID)
	}
}

// ── R2 ISOLATION: isolated caller cannot reach a cross-collection id ──────────

func TestByID_IsolatedCaller_CrossCollection_Rejected(t *testing.T) {
	// The isolated "pigo" caller owns engram_pigo; the target lives in a
	// different collection (engram_user) and must be invisible/immutable to it.
	cases := []struct {
		name string
		call func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error)
	}{
		{"delete_by_id", func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error) {
			return srv.handleDeleteByID(ctx, toolReq(map[string]any{"id": id}))
		}},
		{"update_by_id", func(srv *Server, ctx context.Context, id string) (*mcp.CallToolResult, error) {
			return srv.handleUpdateByID(ctx, toolReq(map[string]any{"id": id, "content": "hijacked"}))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, store := newTestServer()
			seedMemoryInCollection(t, srv, store, "other-col", "foreign content", collection.CollectionUser)
			ctx := WithCallerType(context.Background(), "pigo") // own = engram_pigo
			beforeMut := store.mutations()

			result, err := tc.call(srv, ctx, "other-col")
			if err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
			if result.IsError {
				t.Fatalf("expected structured not_found, got tool error: %s", extractText(result))
			}
			if got := mustStatus(t, result); got != "not_found" {
				t.Fatalf("status = %q, want not_found", got)
			}
			if after := store.mutations(); after != beforeMut {
				t.Fatalf("store mutated (%d -> %d): cross-collection by-id must be a no-op", beforeMut, after)
			}
			// Target untouched: still active, content unchanged.
			got, _ := store.SearchByIDs(context.Background(), []string{"other-col"})
			if len(got) != 1 || got[0].Content != "foreign content" || got[0].LifecycleStatus == memory.LifecycleArchived {
				t.Fatalf("target was mutated: %+v", got)
			}
		})
	}
}

func TestByID_IsolatedCaller_OwnCollection_Succeeds(t *testing.T) {
	// Positive control: same isolated caller CAN act on a target in its own
	// collection (proves the guard is not blanket-deny).
	t.Run("delete_by_id", func(t *testing.T) {
		srv, store := newTestServer()
		seedMemoryInCollection(t, srv, store, "mine-d", "my content", collection.CollectionPigo)
		ctx := WithCallerType(context.Background(), "pigo")
		result, err := srv.handleDeleteByID(ctx, toolReq(map[string]any{"id": "mine-d"}))
		if err != nil {
			t.Fatalf("handleDeleteByID: %v", err)
		}
		if got := mustStatus(t, result); got != "archived" {
			t.Fatalf("status = %q, want archived", got)
		}
	})
	t.Run("update_by_id", func(t *testing.T) {
		srv, store := newTestServer()
		seedMemoryInCollection(t, srv, store, "mine-u", "my content", collection.CollectionPigo)
		ctx := WithCallerType(context.Background(), "pigo")
		result, err := srv.handleUpdateByID(ctx, toolReq(map[string]any{"id": "mine-u", "content": "my new content"}))
		if err != nil {
			t.Fatalf("handleUpdateByID: %v", err)
		}
		if got := mustStatus(t, result); got != "updated" {
			t.Fatalf("status = %q, want updated", got)
		}
		got, _ := store.SearchByIDs(context.Background(), []string{"mine-u"})
		if len(got) != 1 || got[0].Content != "my new content" {
			t.Fatalf("expected content replaced, got %+v", got)
		}
	})
}

// ── R3: MCP search excludes archived by default ───────────────────────────────

func TestSearch_ExcludesArchivedByDefault(t *testing.T) {
	srv, store := newTestServer()
	seedMemory(t, srv, store, "arch-1", "searchable archived content", memory.TypeEvent)
	// Archive it via the by-id soft delete.
	if _, err := srv.deleteByID(context.Background(), "arch-1"); err != nil {
		t.Fatalf("deleteByID: %v", err)
	}

	// Default: archived hidden.
	result, err := srv.handleSearch(context.Background(), toolReq(map[string]any{
		"query": "searchable archived content",
		"limit": float64(10),
	}))
	if err != nil {
		t.Fatalf("handleSearch: %v", err)
	}
	if txt := extractText(result); txt != "[]" {
		t.Fatalf("archived memory leaked into default search: %s", txt)
	}

	// include_archived=true: surfaced.
	result, err = srv.handleSearch(context.Background(), toolReq(map[string]any{
		"query":            "searchable archived content",
		"limit":            float64(10),
		"include_archived": true,
	}))
	if err != nil {
		t.Fatalf("handleSearch: %v", err)
	}
	if txt := extractText(result); txt == "[]" {
		t.Fatalf("include_archived=true should surface archived memory, got empty")
	}
}
