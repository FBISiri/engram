// legacy_search_test.go — W20 Day2 Phase 3 tests for the
// /memories/search compatibility layer (BMO Q3 decision, 2026-05-06).
//
// Behavior contract:
//   - explicit `collection` field in body wins over X-Caller-Type
//   - missing `collection` falls back to ctx-resolution (X-Caller-Type)
//   - unknown collection name → 400
//   - response carries `resolved_collection` annotation per hit (back-compat:
//     legacy callers parsing the array shape still work, just gain a field)
//   - never 30x — old callers must keep working unchanged
package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

func TestLegacySearch_DefaultsToUserViaCtx(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	// Seed at least one memory so /search returns something to inspect.
	createBody := `{"content":"phase3 legacy test seed","type":"event","importance":5}`
	createResp := doJSON(t, ts.URL, "POST", "/memories", "user", createBody)
	createResp.Body.Close()
	if createResp.StatusCode != http.StatusCreated {
		t.Fatalf("seed: want 201, got %d", createResp.StatusCode)
	}

	body := `{"query":"phase3 legacy","limit":5}`
	resp := doJSON(t, ts.URL, "POST", "/memories/search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("want 200, got %d body=%s", resp.StatusCode, string(raw))
	}

	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least one hit")
	}
	// Default ctx-resolution → engram_user.
	if got, _ := hits[0]["resolved_collection"].(string); got != "engram_user" {
		t.Fatalf("default ctx resolution: want engram_user, got %q", got)
	}
}

func TestLegacySearch_ExplicitCollectionOverridesCtx(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	// Seed something so search returns a hit.
	createResp := doJSON(t, ts.URL, "POST", "/memories", "user",
		`{"content":"phase3 explicit override seed","type":"event","importance":5}`)
	createResp.Body.Close()

	// Caller header says user, body explicit says engram_reflection.
	// BMO Q3: explicit wins. (No 403 here — legacy /memories/search is
	// not a per-collection endpoint, so namespace enforcement does not
	// apply; only routing/observability does. Phase 4 will tighten this
	// at the Store layer.)
	body := `{"query":"phase3 explicit","collection":"engram_reflection","limit":3}`
	resp := doJSON(t, ts.URL, "POST", "/memories/search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("want 200, got %d body=%s", resp.StatusCode, string(raw))
	}

	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least one hit")
	}
	if got, _ := hits[0]["resolved_collection"].(string); got != "engram_reflection" {
		t.Fatalf("explicit override: want engram_reflection, got %q", got)
	}
}

func TestLegacySearch_UnknownCollection400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	body := `{"query":"x","collection":"engram_does_not_exist"}`
	resp := doJSON(t, ts.URL, "POST", "/memories/search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", resp.StatusCode)
	}
	raw, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(raw), "unknown collection") {
		t.Fatalf("want 'unknown collection' in body, got %s", string(raw))
	}
}

func TestLegacySearch_NeverRedirects(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	// Disable redirect-following on the client to be safe; the existing
	// httptest.NewServer + DefaultClient already returns 30x as-is for
	// non-GET, but assert explicitly: a legacy search call returns 200,
	// not a 301/308 to /collections/{name}/memories/search.
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	req, err := http.NewRequest("POST", ts.URL+"/memories/search",
		strings.NewReader(`{"query":"phase3 noredirect"}`))
	if err != nil {
		t.Fatalf("build req: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Caller-Type", "agent-self")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 && resp.StatusCode < 400 {
		t.Fatalf("legacy search must not 30x (BMO Q3); got %d", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
}

func TestLegacySearch_AgentSelfCtxResolution(t *testing.T) {
	ts := buildHTTPTestServer(t, "")

	createResp := doJSON(t, ts.URL, "POST", "/memories", "agent-self",
		`{"content":"phase3 agent-self seed","type":"event","importance":5}`)
	createResp.Body.Close()

	body := `{"query":"phase3 agent-self","limit":3}`
	resp := doJSON(t, ts.URL, "POST", "/memories/search", "agent-self", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200, got %d", resp.StatusCode)
	}
	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(hits) == 0 {
		t.Fatalf("expected at least one hit")
	}
	if got, _ := hits[0]["resolved_collection"].(string); got != "engram_agent_self" {
		t.Fatalf("agent-self ctx resolution: want engram_agent_self, got %q", got)
	}
}
