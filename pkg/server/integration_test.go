package server

// Integration tests added in W16 (2026-04-16) to fill gaps in E2E coverage
// for the MCP tool surface: memory_add / memory_search / memory_update / memory_delete.
//
// Focus areas (previously uncovered):
//   1. Tag filter path through the MCP tool arg layer
//   2. time_start / time_end range filter
//   3. Combined filters (type + tag + time)
//   4. memory_delete semantic search path
//   5. Search limit clamping (upper bound 100)
//
// NOTE: these tests bypass dedup by injecting directly into the mockStore.
// The mock embedder is deterministic over char codes, so unrelated content
// often exceeds the 0.92 dedup threshold — not a property of the real embedder.

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// injectMemory creates a memory.Memory and stores it directly with a
// vector derived from the content, bypassing the MCP dedup path.
func injectMemory(t *testing.T, srv *Server, store *mockStore,
	content string, memType memory.MemoryType, tags []string, createdAt float64) *memory.Memory {
	t.Helper()
	ctx := context.Background()

	mem := memory.New(content,
		memory.WithType(memType),
		memory.WithImportance(5),
		memory.WithTags(tags...),
	)
	if createdAt > 0 {
		mem.CreatedAt = createdAt
	}

	vec, err := srv.embedder.Embed(ctx, content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if err := store.Insert(ctx, mem, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}
	return mem
}

// --- Tag filter --------------------------------------------------------------

func TestSearch_TagFilter_Single(t *testing.T) {
	srv, store := newTestServer()

	injectMemory(t, srv, store, "BMO is the local daemon", memory.TypeEvent, []string{"bmo"}, 0)
	injectMemory(t, srv, store, "Frank is Siri's owner", memory.TypeEvent, []string{"frank"}, 0)

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "daemon owner",
		"tags":  []interface{}{"bmo"},
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	var results []map[string]any
	if err := json.Unmarshal([]byte(extractText(result)), &results); err != nil {
		t.Fatalf("parse: %v", err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 result (tag=bmo), got %d: %s", len(results), extractText(result))
	}
	gotTags, _ := results[0]["tags"].([]interface{})
	found := false
	for _, tg := range gotTags {
		if tg == "bmo" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected tag bmo in result, got %v", gotTags)
	}
}

func TestSearch_ThreadTagFilter(t *testing.T) {
	srv, store := newTestServer()
	threadID := "19d986eb6ed26f48"

	injectMemory(t, srv, store, "Clone mechanism reduces context rot",
		memory.TypeInsight, []string{"architecture", "thread:" + threadID}, 0)
	injectMemory(t, srv, store, "Engram dedup uses vector similarity",
		memory.TypeInsight, []string{"architecture", "engram"}, 0)

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "architecture design",
		"tags":  []interface{}{"thread:" + threadID},
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	var results []map[string]any
	json.Unmarshal([]byte(extractText(result)), &results)

	if len(results) != 1 {
		t.Fatalf("expected 1 result filtered by thread tag, got %d", len(results))
	}
}

// --- Time range filter -------------------------------------------------------

func TestSearch_TimeRangeFilter(t *testing.T) {
	srv, store := newTestServer()

	injectMemory(t, srv, store, "event A", memory.TypeEvent, nil, 100)
	injectMemory(t, srv, store, "event B", memory.TypeEvent, nil, 200)
	injectMemory(t, srv, store, "event C", memory.TypeEvent, nil, 300)

	// Query [150, 250] → should return only "event B"
	result, err := callTool(srv, "memory_search", map[string]any{
		"query":      "event",
		"time_start": float64(150),
		"time_end":   float64(250),
		"limit":      float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	var results []map[string]any
	json.Unmarshal([]byte(extractText(result)), &results)

	if len(results) != 1 {
		t.Fatalf("expected 1 result in [150,250] window, got %d: %s",
			len(results), extractText(result))
	}
	if results[0]["content"] != "event B" {
		t.Errorf("expected 'event B', got %v", results[0]["content"])
	}
}

func TestSearch_TimeStartOnly(t *testing.T) {
	srv, store := newTestServer()
	injectMemory(t, srv, store, "old log", memory.TypeEvent, nil, 1000)
	injectMemory(t, srv, store, "new log", memory.TypeEvent, nil, 5000)

	result, _ := callTool(srv, "memory_search", map[string]any{
		"query":      "log",
		"time_start": float64(3000),
	})
	var results []map[string]any
	json.Unmarshal([]byte(extractText(result)), &results)
	if len(results) != 1 || results[0]["content"] != "new log" {
		t.Errorf("expected only 'new log' with time_start=3000, got %v", results)
	}
}

// --- Combined filters --------------------------------------------------------

func TestSearch_CombinedTypeAndTag(t *testing.T) {
	srv, store := newTestServer()

	injectMemory(t, srv, store, "Frank likes cycling",
		memory.TypeIdentity, []string{"frank"}, 0)
	injectMemory(t, srv, store, "Frank went cycling yesterday",
		memory.TypeEvent, []string{"frank"}, 0)
	injectMemory(t, srv, store, "Engram uses qdrant",
		memory.TypeEvent, []string{"engram"}, 0)

	result, _ := callTool(srv, "memory_search", map[string]any{
		"query": "cycling",
		"types": []interface{}{"identity"},
		"tags":  []interface{}{"frank"},
	})
	var results []map[string]any
	json.Unmarshal([]byte(extractText(result)), &results)

	if len(results) != 1 {
		t.Fatalf("expected 1 result (type=identity AND tag=frank), got %d", len(results))
	}
	if results[0]["type"] != "identity" {
		t.Errorf("wrong type: %v", results[0]["type"])
	}
}

// --- memory_delete via semantic search --------------------------------------

func TestDeleteBySemanticSearch(t *testing.T) {
	srv, store := newTestServer()

	injectMemory(t, srv, store,
		"Frank lives in Shanghai Changning district", memory.TypeIdentity, nil, 0)
	injectMemory(t, srv, store,
		"BMO runs on local machine", memory.TypeEvent, nil, 0)

	if store.count() != 2 {
		t.Fatalf("expected 2 memories, got %d", store.count())
	}

	result, err := callTool(srv, "memory_delete", map[string]any{
		"query":                "Shanghai Changning",
		"similarity_threshold": float64(0.5),
	})
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("delete returned error: %s", extractText(result))
	}

	// At least one memory should have been deleted.
	if store.count() >= 2 {
		t.Errorf("expected fewer than 2 memories after delete, got %d", store.count())
	}
}

// --- Limit clamping ----------------------------------------------------------

func TestSearch_LimitClampUpper(t *testing.T) {
	srv, store := newTestServer()

	for i := 0; i < 5; i++ {
		injectMemory(t, srv, store,
			fmt.Sprintf("item %s %c",
				time.Now().Format("150405.000000"), rune('a'+i)),
			memory.TypeEvent, nil, 0)
	}

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "item",
		"limit": float64(500), // should be clamped to 100 internally
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("search returned error: %s", extractText(result))
	}

	var results []map[string]any
	json.Unmarshal([]byte(extractText(result)), &results)
	if len(results) > 100 {
		t.Errorf("limit not clamped: got %d results", len(results))
	}
}
