package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
)

// newTestServerWithMode builds a test server whose ProvenanceMode is set to the
// given value, so write-path strict gating can be exercised.
func newTestServerWithMode(mode string) (*Server, *mockStore) {
	store := newMockStore()
	embedder := newMockEmbedder()
	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
		ProvenanceMode: mode,
	}
	return NewServer(store, embedder, cfg), store
}

func TestProvenance_Strict_AddWithoutSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	res, err := callTool(srv, "memory_add", map[string]any{"content": "no provenance"})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected error, got: %s", extractText(res))
	}
	if !strings.Contains(extractText(res), "ENGRAM_PROVENANCE_MODE=strict") {
		t.Fatalf("unexpected message: %s", extractText(res))
	}
}

func TestProvenance_Strict_AddWithSourceType_Succeeds(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	res, err := callTool(srv, "memory_add", map[string]any{
		"content":     "has provenance",
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", extractText(res))
	}
}

func TestProvenance_Strict_UpdateWithoutSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	res, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "anything",
		"new_content":          "replacement",
		"type":                 "identity",
		"similarity_threshold": float64(0.92),
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected error, got: %s", extractText(res))
	}
	if !strings.Contains(extractText(res), "ENGRAM_PROVENANCE_MODE=strict") {
		t.Fatalf("unexpected message: %s", extractText(res))
	}
}

func TestProvenance_Strict_UpdateWithSourceType_Succeeds(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	res, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "anything",
		"new_content":          "replacement",
		"type":                 "identity",
		"similarity_threshold": float64(0.92),
		"source_type":          "user_input",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error: %s", extractText(res))
	}
}

func TestProvenance_Warn_AddWithoutSourceType_Succeeds(t *testing.T) {
	srv, _ := newTestServerWithMode("warn")
	res, err := callTool(srv, "memory_add", map[string]any{"content": "no provenance"})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("unexpected error in warn mode: %s", extractText(res))
	}
}

func TestProvenance_Strict_RestPostWithoutSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{"content": "no provenance"})
	resp, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("want 422, got %d", resp.StatusCode)
	}
}

func TestProvenance_Strict_RestPutWithoutSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	// Seed a memory (with provenance) to update.
	seed, err := callTool(srv, "memory_add", map[string]any{
		"content":     "seed",
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	id := parseAddMemory(t, seed).ID

	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{"content": "replacement, no provenance"})
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/memories/"+id, bytes.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("want 422, got %d", resp.StatusCode)
	}
}

func TestProvenance_Strict_RestPatchWithoutSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	// Seed a memory WITHOUT provenance via warn-mode store insertion is not
	// possible in strict mode; instead seed with source_type then PATCH content
	// while replacing metadata to a map lacking source_type.
	seed, err := callTool(srv, "memory_add", map[string]any{
		"content":     "seed",
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	id := parseAddMemory(t, seed).ID

	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	// PATCH replacing metadata with a map that has no source_type.
	body, _ := json.Marshal(map[string]any{"metadata": map[string]any{"foo": "bar"}})
	req, _ := http.NewRequest(http.MethodPatch, ts.URL+"/memories/"+id, bytes.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PATCH: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("want 422, got %d", resp.StatusCode)
	}
}

// =============================================================================
// Gap coverage (W-provenance): invalid source_type is mode-independent, strict
// warn-path negatives, and MCP search source_type filtering. These complement
// the missing/valid cases above and the default-mode cases in server_test.go by
// asserting behaviour that the existing suite does not: mode-independence of the
// enum check, that strict mode stores nothing, warn-mode store-side defaulting,
// and multi-type / mixed-set search filtering.
// =============================================================================

// TestProvenance_AddWithInvalidSourceType_Rejects: an invalid source_type must
// be rejected on memory_add in BOTH warn and strict modes (the enum check runs
// before the strict/warn gate) and nothing may be stored.
func TestProvenance_AddWithInvalidSourceType_Rejects(t *testing.T) {
	for _, mode := range []string{"warn", "strict"} {
		t.Run(mode, func(t *testing.T) {
			srv, store := newTestServerWithMode(mode)
			res, err := callTool(srv, "memory_add", map[string]any{
				"content":     "invalid provenance value",
				"source_type": "bogus",
			})
			if err != nil {
				t.Fatalf("callTool: %v", err)
			}
			if !res.IsError {
				t.Fatalf("expected error for invalid source_type in %s mode, got: %s", mode, extractText(res))
			}
			if store.count() != 0 {
				t.Errorf("expected 0 memories stored on invalid source_type in %s mode, got %d", mode, store.count())
			}
		})
	}
}

// TestProvenance_UpdateWithInvalidSourceType_Rejects: an invalid source_type on
// memory_update must be rejected in BOTH warn and strict modes.
func TestProvenance_UpdateWithInvalidSourceType_Rejects(t *testing.T) {
	for _, mode := range []string{"warn", "strict"} {
		t.Run(mode, func(t *testing.T) {
			srv, store := newTestServerWithMode(mode)
			injectMemory(t, srv, store, "The build server is ci-01", memory.TypeEvent, nil, 0)
			res, err := callTool(srv, "memory_update", map[string]any{
				"old_content":          "The build server is ci-01",
				"new_content":          "The build server is ci-02",
				"similarity_threshold": float64(0.92),
				"source_type":          "bogus",
			})
			if err != nil {
				t.Fatalf("callTool: %v", err)
			}
			if !res.IsError {
				t.Fatalf("expected error for invalid source_type in %s mode, got: %s", mode, extractText(res))
			}
		})
	}
}

// TestProvenance_RestPostWithInvalidSourceType_Rejects: REST POST /memories with
// an invalid source_type returns 400 even in strict mode (the enum check wins
// over the strict "missing" gate, which would otherwise return 422).
func TestProvenance_RestPostWithInvalidSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"content":  "invalid provenance",
		"metadata": map[string]any{"source_type": "bogus"},
	})
	resp, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 for invalid source_type, got %d", resp.StatusCode)
	}
}

// TestProvenance_RestPutWithInvalidSourceType_Rejects: REST PUT /memories/{id}
// with an invalid source_type returns 400 even in strict mode.
func TestProvenance_RestPutWithInvalidSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	seed, err := callTool(srv, "memory_add", map[string]any{
		"content":     "seed for invalid put",
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	id := parseAddMemory(t, seed).ID

	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"content":  "replacement, invalid provenance",
		"metadata": map[string]any{"source_type": "bogus"},
	})
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/memories/"+id, bytes.NewReader(body))
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("PUT: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 for invalid source_type, got %d", resp.StatusCode)
	}
}

// TestProvenance_Strict_AddWithoutSourceType_NoDefaulting: in strict mode a
// missing source_type must NOT be defaulted-and-stored — the store stays empty.
func TestProvenance_Strict_AddWithoutSourceType_NoDefaulting(t *testing.T) {
	srv, store := newTestServerWithMode("strict")
	res, err := callTool(srv, "memory_add", map[string]any{"content": "no provenance, strict"})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected error, got: %s", extractText(res))
	}
	if store.count() != 0 {
		t.Errorf("strict mode must not store a defaulted memory, got count=%d", store.count())
	}
}

// TestProvenance_Warn_UpdateWithoutSourceType_DefaultsReflection: in warn mode a
// memory_update without source_type succeeds AND the stored new memory carries
// metadata.source_type == "unknown". Asserted against the store (not the
// response) under an explicit warn-mode server.
func TestProvenance_Warn_UpdateWithoutSourceType_DefaultsReflection(t *testing.T) {
	srv, store := newTestServerWithMode("warn")
	injectMemory(t, srv, store, "The cache TTL is 60 seconds", memory.TypeEvent, nil, 0)

	res, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "The cache TTL is 60 seconds",
		"new_content":          "The cache TTL is 120 seconds",
		"similarity_threshold": float64(0.92),
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("warn-mode update without source_type should succeed, got: %s", extractText(res))
	}

	found := false
	for _, m := range store.all() {
		if m.Content == "The cache TTL is 120 seconds" {
			found = true
			if got, _ := m.Metadata["source_type"].(string); got != "unknown" {
				t.Errorf("expected stored source_type=unknown, got %v", m.Metadata["source_type"])
			}
		}
	}
	if !found {
		t.Fatalf("updated memory was not stored")
	}
}

// TestProvenance_SearchFilterBySourceType: with three memories of distinct
// source types, memory_search filtered to source_type=[web_search] returns only
// the web_search memory.
func TestProvenance_SearchFilterBySourceType(t *testing.T) {
	srv, store := newTestServerWithMode("warn")
	// Inject directly with distinct source_type metadata to bypass the MCP add
	// dedup path (the mock embedder treats dissimilar English sentences as near
	// duplicates, see integration_test.go).
	ctx := context.Background()
	seed := func(content, st string) {
		mem := memory.New(content, memory.WithType(memory.TypeEvent))
		mem.Metadata = map[string]any{"source_type": st}
		vec, err := srv.embedder.Embed(ctx, content)
		if err != nil {
			t.Fatalf("embed: %v", err)
		}
		if err := store.Insert(ctx, mem, vec); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	seed("The satellite telemetry shows a nominal orbit", "tool_output")
	seed("The user prefers dark mode in the editor", "user_input")
	seed("Wikipedia notes the Rhine flows through Basel", "web_search")

	res, err := callTool(srv, "memory_search", map[string]any{
		"query":       "telemetry editor Rhine Basel orbit",
		"limit":       float64(10),
		"source_type": []interface{}{"web_search"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res.IsError {
		t.Fatalf("search returned error: %s", extractText(res))
	}
	var hits []struct {
		Content  string         `json:"content"`
		Metadata map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(extractText(res)), &hits); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("expected exactly 1 web_search hit, got %d: %s", len(hits), extractText(res))
	}
	if got, _ := hits[0].Metadata["source_type"].(string); got != "web_search" {
		t.Errorf("expected hit source_type=web_search, got %v", hits[0].Metadata["source_type"])
	}
}

// TestProvenance_SearchFilterInvalidSourceType_Rejects: a source_type filter set
// containing an invalid value (even mixed with a valid one) is rejected — every
// element of the set is validated.
func TestProvenance_SearchFilterInvalidSourceType_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("warn")
	res, err := callTool(srv, "memory_search", map[string]any{
		"query":       "anything",
		"source_type": []interface{}{"web_search", "bogus"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected error for invalid source_type in filter set, got: %s", extractText(res))
	}
}

// TestProvenance_SearchResult_SourceTypeTopLevel: source_type must appear as a
// TOP-LEVEL field in MCP search results (in addition to inside metadata).
func TestProvenance_SearchResult_SourceTypeTopLevel(t *testing.T) {
	srv, store := newTestServerWithMode("warn")
	ctx := context.Background()
	mem := memory.New("Wikipedia notes the Rhine flows through Basel", memory.WithType(memory.TypeEvent))
	mem.Metadata = map[string]any{"source_type": "web_search"}
	vec, err := srv.embedder.Embed(ctx, mem.Content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if err := store.Insert(ctx, mem, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}

	res, err := callTool(srv, "memory_search", map[string]any{
		"query": "Rhine Basel",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if res.IsError {
		t.Fatalf("search returned error: %s", extractText(res))
	}

	raw := extractText(res)
	// Re-marshal to confirm the top-level JSON field is present.
	var hits []map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &hits); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least 1 hit, got 0: %s", raw)
	}
	st, ok := hits[0]["source_type"]
	if !ok {
		t.Fatalf("expected top-level source_type field, got: %s", raw)
	}
	var stv string
	if err := json.Unmarshal(st, &stv); err != nil {
		t.Fatalf("source_type not a string: %v", err)
	}
	if stv != "web_search" {
		t.Errorf("expected top-level source_type=web_search, got %q", stv)
	}
	if !strings.Contains(raw, `"source_type":"web_search"`) {
		t.Errorf("expected raw JSON to contain top-level source_type:web_search, got: %s", raw)
	}
}

// TestProvenance_RestGetByID_SourceTypeTopLevel: REST GET /memories/{id} must
// return source_type as a TOP-LEVEL field while metadata still contains it.
func TestProvenance_RestGetByID_SourceTypeTopLevel(t *testing.T) {
	srv, _ := newTestServerWithMode("warn")
	seed, err := callTool(srv, "memory_add", map[string]any{
		"content":     "Wikipedia notes the Rhine flows through Basel",
		"source_type": "web_search",
	})
	if err != nil {
		t.Fatalf("seed: %v", err)
	}
	id := parseAddMemory(t, seed).ID

	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/memories/" + id)
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var obj map[string]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&obj); err != nil {
		t.Fatalf("decode: %v", err)
	}
	var stv string
	if raw, ok := obj["source_type"]; !ok {
		t.Fatalf("expected top-level source_type field, got: %v", obj)
	} else if err := json.Unmarshal(raw, &stv); err != nil || stv != "web_search" {
		t.Fatalf("expected top-level source_type=web_search, got %q (err=%v)", stv, err)
	}
	// Additive invariant: metadata must still contain source_type.
	var md map[string]any
	if err := json.Unmarshal(obj["metadata"], &md); err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if got, _ := md["source_type"].(string); got != "web_search" {
		t.Errorf("expected metadata.source_type=web_search, got %v", md["source_type"])
	}
}

// TestProvenance_RestSearch_SourceTypeTopLevel: REST POST /memories/search must
// return source_type as a TOP-LEVEL field in a result while metadata retains it.
func TestProvenance_RestSearch_SourceTypeTopLevel(t *testing.T) {
	srv, _ := newTestServerWithMode("warn")
	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "Wikipedia notes the Rhine flows through Basel",
		"source_type": "web_search",
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{"query": "Rhine Basel", "limit": 10})
	resp, err := http.Post(ts.URL+"/memories/search", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var hits []map[string]json.RawMessage
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least 1 hit, got 0")
	}
	var stv string
	if raw, ok := hits[0]["source_type"]; !ok {
		t.Fatalf("expected top-level source_type field, got: %v", hits[0])
	} else if err := json.Unmarshal(raw, &stv); err != nil || stv != "web_search" {
		t.Fatalf("expected top-level source_type=web_search, got %q (err=%v)", stv, err)
	}
	var md map[string]any
	if err := json.Unmarshal(hits[0]["metadata"], &md); err != nil {
		t.Fatalf("metadata: %v", err)
	}
	if got, _ := md["source_type"].(string); got != "web_search" {
		t.Errorf("expected metadata.source_type=web_search, got %v", md["source_type"])
	}
}
