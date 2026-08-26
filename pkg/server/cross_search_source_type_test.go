package server

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// seedCrossSearchMem inserts a memory into engram_user directly via the mock
// store, bypassing the REST 0.92 dedup check (C1) that the toy mock embedder
// would otherwise trip for any two >20-char strings. The memory carries the
// given source_type in metadata so cross-search filters can find it.
func seedCrossSearchMem(t *testing.T, store *mockStore, emb *mockEmbedder, content, sourceType string) {
	t.Helper()
	mem := memory.New(content,
		memory.WithType(memory.TypeEvent),
		memory.WithImportance(5),
		memory.WithMetadata(map[string]any{"source_type": sourceType}),
	)
	mem.Collection = "engram_user"
	vec, err := emb.Embed(context.Background(), content)
	if err != nil {
		t.Fatalf("embed %s: %v", sourceType, err)
	}
	if err := store.Insert(context.Background(), mem, vec); err != nil {
		t.Fatalf("seed %s: insert failed: %v", sourceType, err)
	}
}

func crossSearch(t *testing.T, ts, body string) *http.Response {
	t.Helper()
	return doJSON(t, ts, "POST", "/memories/cross-search", "user", body)
}

// (a) source_type filter returns only matching memories.
// (d) response includes the source_type field in results.
func TestCrossSearch_SourceTypeFilter(t *testing.T) {
	ts, store, emb := buildHTTPTestServerWithStore(t)
	seedCrossSearchMem(t, store, emb, "Paris is the capital of France", "web_search")
	seedCrossSearchMem(t, store, emb, "Berlin has an excellent public transit system", "user_input")

	body := `{"query":"European cities","collections":["engram_user"],"limit":10,"source_type":["web_search"]}`
	resp := crossSearch(t, ts.URL, body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected exactly 1 web_search hit, got %d", len(hits))
	}
	if got, _ := hits[0]["source_type"].(string); got != "web_search" {
		t.Fatalf("expected hit source_type=web_search, got %v", hits[0]["source_type"])
	}
}

// (b) WITHOUT source_type filter returns all memories (backward compat).
func TestCrossSearch_NoSourceTypeFilter_ReturnsAll(t *testing.T) {
	ts, store, emb := buildHTTPTestServerWithStore(t)
	seedCrossSearchMem(t, store, emb, "Paris is the capital of France", "web_search")
	seedCrossSearchMem(t, store, emb, "Berlin has an excellent public transit system", "user_input")

	body := `{"query":"European cities","collections":["engram_user"],"limit":10}`
	resp := crossSearch(t, ts.URL, body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits without source_type filter, got %d", len(hits))
	}
}

// (c) invalid source_type returns 400.
func TestCrossSearch_InvalidSourceType_400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"query":"Paris","collections":["engram_user"],"source_type":["not_a_real_type"]}`
	resp := crossSearch(t, ts.URL, body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 for invalid source_type, got %d", resp.StatusCode)
	}
}
