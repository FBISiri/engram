// pigo_isolation_test.go — read-isolation for the pigo caller type.
//
// Contract:
//   - a pigo caller's search is FORCED to its own collection (engram_pigo);
//     memories in other collections are never returned (read isolation).
//   - a pigo search response DOES carry resolved_collection=engram_pigo
//     (genuinely collection-scoped), whereas legacy fan-out callers do not.
package server

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
)

func TestPigoSearch_ReadsOnlyOwnCollection(t *testing.T) {
	collection.DefaultRegistry.Init() // idempotent
	ts := buildHTTPTestServer(t, "")

	// Seed one memory as a legacy user (→ engram_user) and one as pigo
	// (→ engram_pigo). The pigo search must only ever see the pigo one.
	if r := doJSON(t, ts.URL, "POST", "/memories", "user",
		`{"content":"user-scoped secret alpha","type":"event","importance":5}`); r.StatusCode != http.StatusCreated {
		t.Fatalf("seed user: want 201, got %d", r.StatusCode)
	} else {
		_ = r.Body.Close()
	}
	if r := doJSON(t, ts.URL, "POST", "/memories", "pigo",
		`{"content":"pigo-scoped secret alpha","type":"event","importance":5}`); r.StatusCode != http.StatusCreated {
		t.Fatalf("seed pigo: want 201, got %d", r.StatusCode)
	} else {
		_ = r.Body.Close()
	}

	resp := doJSON(t, ts.URL, "POST", "/memories/search", "pigo", `{"query":"secret alpha","limit":10}`)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("want 200, got %d body=%s", resp.StatusCode, string(raw))
	}
	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected the pigo-scoped hit")
	}
	for _, h := range hits {
		// (c) read isolation: only engram_pigo memories are visible.
		if got, _ := h["collection"].(string); got != collection.CollectionPigo {
			t.Fatalf("pigo search leaked non-pigo memory: collection=%q content=%v", got, h["content"])
		}
		// (d) genuinely scoped → resolved_collection reported.
		if got, _ := h["resolved_collection"].(string); got != collection.CollectionPigo {
			t.Fatalf("pigo search: want resolved_collection=%s, got %q", collection.CollectionPigo, got)
		}
	}
}
