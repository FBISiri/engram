package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// helper: do a request with a body and X-Caller-Type header, return resp.
func doJSON(t *testing.T, ts string, method, path, callerType, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequest(method, ts+path, strings.NewReader(body))
	if err != nil {
		t.Fatalf("build req: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if callerType != "" {
		req.Header.Set("X-Caller-Type", callerType)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, path, err)
	}
	return resp
}

// ─────────────────────────────────────────────────────────────
// Per-collection CRUD: namespace enforcement
// ─────────────────────────────────────────────────────────────

func TestCollectionCreate_MismatchForbidden(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	// caller is "user" (default), but URL says engram_reflection → 403.
	body := `{"content":"hi","type":"event","importance":5}`
	resp := doJSON(t, ts.URL, "POST", "/collections/engram_reflection/memories", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusForbidden {
		t.Fatalf("want 403 on cross-collection write, got %d", resp.StatusCode)
	}
}

func TestCollectionCreate_MatchOK(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"content":"hello from user","type":"event","importance":5}`
	resp := doJSON(t, ts.URL, "POST", "/collections/engram_user/memories", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("want 201, got %d", resp.StatusCode)
	}
}

func TestCollectionCreate_UnknownCollection(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"content":"x","type":"event"}`
	resp := doJSON(t, ts.URL, "POST", "/collections/engram_does_not_exist/memories", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("want 404 on unregistered collection, got %d", resp.StatusCode)
	}
}

func TestCollectionCreate_ReflectionCallerOK(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"content":"reflection insight","type":"insight","importance":6}`
	resp := doJSON(t, ts.URL, "POST", "/collections/engram_reflection/memories", "reflection", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("want 201 for reflection→engram_reflection, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// Cross-collection search: strict mode
// ─────────────────────────────────────────────────────────────

func TestCrossSearch_MissingCollections_400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"query":"hello","limit":5}`
	resp := doJSON(t, ts.URL, "POST", "/memories/cross-search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 when collections missing (strict mode), got %d", resp.StatusCode)
	}
}

func TestCrossSearch_EmptyCollections_400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"query":"hello","collections":[],"limit":5}`
	resp := doJSON(t, ts.URL, "POST", "/memories/cross-search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 when collections=[], got %d", resp.StatusCode)
	}
}

func TestCrossSearch_UnknownCollection_400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"query":"hello","collections":["engram_user","engram_phantom"],"limit":5}`
	resp := doJSON(t, ts.URL, "POST", "/memories/cross-search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 on unknown collection in list, got %d", resp.StatusCode)
	}
}

func TestCrossSearch_ValidCollections_200(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"query":"anything","collections":["engram_user","engram_reflection"],"limit":5}`
	resp := doJSON(t, ts.URL, "POST", "/memories/cross-search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("want 200 with strict-mode valid request, got %d", resp.StatusCode)
	}
	var hits []map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Each hit should carry the `collection` annotation.
	for _, h := range hits {
		if _, ok := h["collection"]; !ok {
			t.Fatalf("hit missing collection annotation: %v", h)
		}
	}
}

func TestCrossSearch_MissingQuery_400(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"collections":["engram_user"]}`
	resp := doJSON(t, ts.URL, "POST", "/memories/cross-search", "user", body)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400 when query missing, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// Compatibility: legacy /memories routes still work (no 30x).
// Day1 review pinned: compat layer must forward, not redirect.
// ─────────────────────────────────────────────────────────────

func TestLegacyMemoriesPOST_NoRedirect(t *testing.T) {
	ts := buildHTTPTestServer(t, "")
	body := `{"content":"legacy path","type":"event","importance":5}`
	req, _ := http.NewRequest("POST", ts.URL+"/memories", bytes.NewReader([]byte(body)))
	req.Header.Set("Content-Type", "application/json")
	// Disable auto-follow so we can detect a stray redirect explicitly.
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /memories: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 == 3 {
		t.Fatalf("legacy /memories must not redirect (Day1 lock-in), got %d", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("want 201 from legacy /memories, got %d", resp.StatusCode)
	}
}
