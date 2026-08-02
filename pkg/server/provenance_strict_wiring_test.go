package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/reflection"
)

// FIX 1: strict mode must reach the reflection engine as block-mode filtering.
// Previously the MCP/REST reflect handlers wired only the legacy
// RequireProvenance/AllowedProvenances fields, which resolveProvenanceFilter()
// always maps to ProvenanceModeDefault — so ENGRAM_PROVENANCE_MODE=strict
// silently downgraded to default. reflectionConfig() now threads the mode.
func TestReflectionConfig_StrictYieldsBlockFilter(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	srv.cfg.RequireProvenance = true
	srv.cfg.AllowedProvenances = []string{"reflection", "user_input"}

	rc := srv.reflectionConfig()
	if !rc.ProvenanceFilter.Enabled {
		t.Fatal("expected provenance filter enabled in strict mode")
	}
	if rc.ProvenanceFilter.Mode != reflection.ProvenanceModeBlock {
		t.Fatalf("expected block mode, got %q", rc.ProvenanceFilter.Mode)
	}
	// Block mode must actually produce the exclusion filters (OpIn + OpIsNull).
	filters := reflection.BuildEvidenceFilters(rc.ProvenanceFilter)
	if len(filters) != 2 {
		t.Fatalf("expected 2 block-mode evidence filters, got %d: %+v", len(filters), filters)
	}
}

// Sanity: warn mode must NOT downgrade to block.
func TestReflectionConfig_WarnYieldsWarnFilter(t *testing.T) {
	srv, _ := newTestServerWithMode("warn")
	srv.cfg.RequireProvenance = true
	srv.cfg.AllowedProvenances = []string{"reflection"}
	if m := srv.reflectionConfig().ProvenanceFilter.Mode; m != reflection.ProvenanceModeWarn {
		t.Fatalf("expected warn mode, got %q", m)
	}
}

// FIX 2: explicit "unknown" (the sentinel, excluded from the allow-list) must be
// rejected in strict mode on memory_add — not persisted then silently filtered.
func TestProvenance_Strict_AddExplicitUnknown_Rejects(t *testing.T) {
	srv, store := newTestServerWithMode("strict")
	srv.cfg.AllowedProvenances = []string{"reflection", "user_input"}
	res, err := callTool(srv, "memory_add", map[string]any{
		"content":     "explicit unknown bypass attempt",
		"source_type": "unknown",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected rejection for explicit unknown, got: %s", extractText(res))
	}
	if store.count() != 0 {
		t.Errorf("strict mode must not store explicit unknown, count=%d", store.count())
	}
}

// FIX 2: a valid source_type NOT in the allow-list must also be rejected.
func TestProvenance_Strict_AddNotInAllowList_Rejects(t *testing.T) {
	srv, store := newTestServerWithMode("strict")
	srv.cfg.AllowedProvenances = []string{"reflection"}
	res, err := callTool(srv, "memory_add", map[string]any{
		"content":     "valid enum but not allowed",
		"source_type": "web_search",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected rejection for value outside allow-list, got: %s", extractText(res))
	}
	if store.count() != 0 {
		t.Errorf("strict mode must not store non-allowed source_type, count=%d", store.count())
	}
}

// FIX 2: a source_type IN the allow-list is still accepted in strict mode.
func TestProvenance_Strict_AddInAllowList_Succeeds(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	srv.cfg.AllowedProvenances = []string{"reflection", "user_input"}
	res, err := callTool(srv, "memory_add", map[string]any{
		"content":     "allowed provenance",
		"source_type": "user_input",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if res.IsError {
		t.Fatalf("expected success for allowed source_type, got: %s", extractText(res))
	}
}

// FIX 2: explicit unknown on memory_update must be rejected in strict mode too.
func TestProvenance_Strict_UpdateExplicitUnknown_Rejects(t *testing.T) {
	srv, store := newTestServerWithMode("strict")
	injectMemory(t, srv, store, "The build server is ci-01", memory.TypeEvent, nil, 0)
	res, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "The build server is ci-01",
		"new_content":          "The build server is ci-02",
		"similarity_threshold": float64(0.92),
		"source_type":          "unknown",
	})
	if err != nil {
		t.Fatalf("callTool: %v", err)
	}
	if !res.IsError {
		t.Fatalf("expected rejection for explicit unknown on update, got: %s", extractText(res))
	}
}

// FIX 2: REST POST /memories with explicit unknown must return 422 in strict mode.
func TestProvenance_Strict_RestPostExplicitUnknown_Rejects(t *testing.T) {
	srv, _ := newTestServerWithMode("strict")
	srv.cfg.AllowedProvenances = []string{"reflection", "user_input"}
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	body, _ := json.Marshal(map[string]any{
		"content":  "explicit unknown via rest",
		"metadata": map[string]any{"source_type": "unknown"},
	})
	resp, err := http.Post(ts.URL+"/memories", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("want 422 for explicit unknown, got %d", resp.StatusCode)
	}
	var obj map[string]string
	_ = json.NewDecoder(resp.Body).Decode(&obj)
	if !strings.Contains(obj["error"], "strict") {
		t.Errorf("expected strict-mode error message, got %q", obj["error"])
	}
}
