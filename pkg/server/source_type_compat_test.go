package server

// C1 backward-compatibility integration tests (R1).
//
// The source_type provenance field is soft-required: memories created WITHOUT
// a source_type must keep working across the full CRUD surface. These tests
// verify read/search, update (with and without adding source_type), delete,
// and mixed result sets against the same test infra used by integration_test.go
// (injectMemory / newTestServer / callTool / extractText / mockStore).

import (
	"encoding/json"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// --- Read / search: no source_type still searchable -------------------------

func TestCompat_SearchReturnsNoSourceTypeMemories(t *testing.T) {
	srv, st := newTestServer()

	injectMemory(t, srv, st, "Legacy memory about the deploy pipeline",
		memory.TypeEvent, nil, 0)
	injectMemory(t, srv, st, "Legacy note about the deploy rollback plan",
		memory.TypeEvent, nil, 0)

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "deploy pipeline rollback",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("search returned error: %s", extractText(result))
	}

	var hits []struct {
		Content  string         `json:"content"`
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &hits); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 legacy memories returned, got %d: %s", len(hits), extractText(result))
	}
	for _, h := range hits {
		if _, ok := h.Metadata["source_type"]; ok {
			t.Errorf("expected no source_type on legacy memory, got %v", h.Metadata["source_type"])
		}
	}
}

// --- Update: legacy memory updated WITHOUT source_type defaults to unknown ---

func TestCompat_UpdateNoSourceTypeDefaultsReflection(t *testing.T) {
	srv, st := newTestServer()

	injectMemory(t, srv, st, "The staging host is host-alpha", memory.TypeEvent, nil, 0)

	result, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "The staging host is host-alpha",
		"new_content":          "The staging host is host-bravo",
		"similarity_threshold": float64(0.92),
	})
	if err != nil {
		t.Fatalf("update failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("update returned error: %s", extractText(result))
	}

	var resp struct {
		DeletedCount int           `json:"deleted_count"`
		NewMemory    memory.Memory `json:"new_memory"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if resp.DeletedCount != 1 {
		t.Errorf("expected old memory deleted, deleted_count=%d", resp.DeletedCount)
	}
	if got, _ := resp.NewMemory.Metadata["source_type"].(string); got != "unknown" {
		t.Errorf("expected source_type defaulted to unknown after plain update, got %v", resp.NewMemory.Metadata["source_type"])
	}
}

// --- Update: legacy memory gains source_type via update ----------------------

func TestCompat_UpdateNoSourceTypeAddsSourceType(t *testing.T) {
	srv, st := newTestServer()

	injectMemory(t, srv, st, "The staging host is host-alpha", memory.TypeEvent, nil, 0)

	result, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "The staging host is host-alpha",
		"new_content":          "The staging host is host-bravo",
		"similarity_threshold": float64(0.92),
		"source_type":          "user_input",
	})
	if err != nil {
		t.Fatalf("update failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("update returned error: %s", extractText(result))
	}

	var resp struct {
		NewMemory memory.Memory `json:"new_memory"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &resp); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got, _ := resp.NewMemory.Metadata["source_type"].(string); got != "user_input" {
		t.Errorf("expected source_type=user_input after update, got %v", resp.NewMemory.Metadata["source_type"])
	}
}

// --- Delete: legacy memory deletable via memory_delete ----------------------

func TestCompat_DeleteNoSourceType(t *testing.T) {
	srv, st := newTestServer()

	injectMemory(t, srv, st, "Legacy fact: the backup runs at midnight",
		memory.TypeEvent, nil, 0)
	if st.count() != 1 {
		t.Fatalf("expected 1 memory before delete, got %d", st.count())
	}

	result, err := callTool(srv, "memory_delete", map[string]any{
		"query":                "backup runs at midnight",
		"similarity_threshold": float64(0.5),
		"limit":                float64(1),
	})
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("delete returned error: %s", extractText(result))
	}
	if st.count() != 0 {
		t.Errorf("expected legacy memory deleted, count=%d", st.count())
	}
}

// --- Mixed: search returning BOTH source_type and legacy memories -----------

func TestCompat_SearchMixedSourceTypeAndLegacy(t *testing.T) {
	srv, st := newTestServer()

	// One memory WITH source_type (via the MCP add path), one legacy without.
	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "Berlin is the capital of Germany",
		"source_type": "web_search",
	}); err != nil {
		t.Fatalf("add web_search failed: %v", err)
	}
	injectMemory(t, srv, st, "Berlin is a nice city to live in", memory.TypeEvent, nil, 0)

	// Unfiltered search returns BOTH.
	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "Berlin Germany city",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	var all []struct {
		Content  string         `json:"content"`
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &all); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("expected mixed search to return 2 memories, got %d: %s", len(all), extractText(result))
	}
	withST, withoutST := 0, 0
	for _, h := range all {
		if _, ok := h.Metadata["source_type"]; ok {
			withST++
		} else {
			withoutST++
		}
	}
	if withST != 1 || withoutST != 1 {
		t.Errorf("expected 1 with + 1 without source_type, got with=%d without=%d", withST, withoutST)
	}

	// Filtering by source_type=[web_search] returns ONLY the provenance memory.
	result, err = callTool(srv, "memory_search", map[string]any{
		"query":       "Berlin Germany city",
		"limit":       float64(10),
		"source_type": []interface{}{"web_search"},
	})
	if err != nil {
		t.Fatalf("filtered search failed: %v", err)
	}
	var filtered []struct {
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &filtered); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(filtered) != 1 {
		t.Fatalf("expected 1 web_search hit, got %d: %s", len(filtered), extractText(result))
	}
	if got, _ := filtered[0].Metadata["source_type"].(string); got != "web_search" {
		t.Errorf("expected filtered hit source_type=web_search, got %v", filtered[0].Metadata["source_type"])
	}
}
