package server

import (
	"encoding/json"
	"net/http"
	"testing"
)

// seedCrossSearchMem creates a memory in engram_user (caller "user") carrying
// the given source_type in metadata. Fails the test on any non-201 status.
func seedCrossSearchMem(t *testing.T, ts, content, sourceType string) {
	t.Helper()
	body := `{"content":"` + content + `","type":"event","importance":5,"metadata":{"source_type":"` + sourceType + `"}}`
	resp := doJSON(t, ts, "POST", "/collections/engram_user/memories", "user", body)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("seed %s: want 201, got %d", sourceType, resp.StatusCode)
	}
}

func crossSearch(t *testing.T, ts, body string) *http.Response {
	t.Helper()
	return doJSON(t, ts, "POST", "/memories/cross-search", "user", body)
}

// (a) source_type filter returns only matching memories.
// (d) response includes the source_type field in results.
func TestCrossSearch_SourceTypeFilter(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	seedCrossSearchMem(t, ts.URL, "Paris is the capital of France", "web_search")
	seedCrossSearchMem(t, ts.URL, "Paris is a lovely place to visit", "user_input")

	body := `{"query":"Paris","collections":["engram_user"],"limit":10,"source_type":["web_search"]}`
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
	ts := buildHTTPTestServer(t, "")
	seedCrossSearchMem(t, ts.URL, "Paris is the capital of France", "web_search")
	seedCrossSearchMem(t, ts.URL, "Paris is a lovely place to visit", "user_input")

	body := `{"query":"Paris","collections":["engram_user"],"limit":10}`
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
