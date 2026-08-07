// mcp_http_test.go — /mcp streamable-HTTP transport + isolation contract.
//
// Verifies: principal-key auth (401 without/with bad key), the initialize +
// tools/list handshake, and that the pigo principal is HARD-isolated to
// engram_pigo across add/search/delete/reflection tool calls. Uses httptest +
// the mock store/embedder — no real qdrant required.
package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

const testPigoKey = "testpigokey"

// buildMCPTestServer returns an httptest.Server exposing /mcp with the pigo
// principal key configured, plus the backing mock store for inspection.
func buildMCPTestServer(t *testing.T) (*httptest.Server, *mockStore) {
	t.Helper()
	collection.DefaultRegistry.Init() // idempotent
	srv, store := newTestServer()
	h := NewHTTPServer(srv, 0, "")
	h.SetPrincipalKeys(map[string]string{"pigo": testPigoKey})
	ts := httptest.NewServer(h.Handler())
	t.Cleanup(ts.Close)
	return ts, store
}

// newMCPClient dials /mcp with the given bearer token and completes initialize.
func newMCPClient(t *testing.T, url, token string) *mcpclient.Client {
	t.Helper()
	c, err := mcpclient.NewStreamableHttpClient(url+"/mcp",
		transport.WithHTTPHeaders(map[string]string{"Authorization": "Bearer " + token}))
	if err != nil {
		t.Fatalf("new client: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	ctx := context.Background()
	if err := c.Start(ctx); err != nil {
		t.Fatalf("start: %v", err)
	}
	var initReq mcp.InitializeRequest
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{Name: "test", Version: "0"}
	if _, err := c.Initialize(ctx, initReq); err != nil {
		t.Fatalf("initialize: %v", err)
	}
	return c
}

func callMCP(t *testing.T, c *mcpclient.Client, name string, args map[string]any) *mcp.CallToolResult {
	t.Helper()
	var req mcp.CallToolRequest
	req.Params.Name = name
	req.Params.Arguments = args
	res, err := c.CallTool(context.Background(), req)
	if err != nil {
		t.Fatalf("call %s: %v", name, err)
	}
	return res
}

// ─────────────────────────────────────────────────────────────
// Auth
// ─────────────────────────────────────────────────────────────

func TestMCPHTTP_NoKey_401(t *testing.T) {
	ts, _ := buildMCPTestServer(t)
	resp, err := http.Post(ts.URL+"/mcp", "application/json", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no key: want 401, got %d", resp.StatusCode)
	}
}

func TestMCPHTTP_BadKey_401(t *testing.T) {
	ts, _ := buildMCPTestServer(t)
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	req.Header.Set("Authorization", "Bearer wrongkey")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad key: want 401, got %d", resp.StatusCode)
	}
}

func TestMCPHTTP_FailClosed_NoPrincipalKeys(t *testing.T) {
	collection.DefaultRegistry.Init()
	srv, _ := newTestServer()
	h := NewHTTPServer(srv, 0, "") // no principal keys configured
	ts := httptest.NewServer(h.Handler())
	t.Cleanup(ts.Close)
	req, _ := http.NewRequest(http.MethodPost, ts.URL+"/mcp",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	req.Header.Set("Authorization", "Bearer "+testPigoKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("fail-closed: want 401, got %d", resp.StatusCode)
	}
}

// ─────────────────────────────────────────────────────────────
// Handshake
// ─────────────────────────────────────────────────────────────

func TestMCPHTTP_Handshake(t *testing.T) {
	ts, _ := buildMCPTestServer(t)
	c := newMCPClient(t, ts.URL, testPigoKey)
	tools, err := c.ListTools(context.Background(), mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("tools/list: %v", err)
	}
	if len(tools.Tools) == 0 {
		t.Fatalf("expected tools from tools/list")
	}
	var haveSearch bool
	for _, tl := range tools.Tools {
		if tl.Name == "memory_search" {
			haveSearch = true
		}
	}
	if !haveSearch {
		t.Fatalf("memory_search not advertised")
	}
}

// ─────────────────────────────────────────────────────────────
// Isolation
// ─────────────────────────────────────────────────────────────

func TestMCPHTTP_Add_LandsInPigoCollection(t *testing.T) {
	ts, store := buildMCPTestServer(t)
	c := newMCPClient(t, ts.URL, testPigoKey)
	res := callMCP(t, c, "memory_add", map[string]any{"content": "pigo add via mcp", "type": "event", "source_type": "tool_output"})
	if res.IsError {
		t.Fatalf("add errored: %s", extractText(res))
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	if len(store.memories) != 1 {
		t.Fatalf("want 1 memory, got %d", len(store.memories))
	}
	for _, sp := range store.memories {
		if sp.mem.Collection != collection.CollectionPigo {
			t.Fatalf("memory landed in %q, want %s", sp.mem.Collection, collection.CollectionPigo)
		}
	}
}

func TestMCPHTTP_Search_PigoScopedIgnoresCollectionsArg(t *testing.T) {
	ts, store := buildMCPTestServer(t)
	// Seed one user memory + one pigo memory directly in the store.
	seed(store, "engram_user", collection.CollectionUser, "shared secret alpha")
	seed(store, "engram_pigo", collection.CollectionPigo, "shared secret alpha")

	c := newMCPClient(t, ts.URL, testPigoKey)
	// Even asking for engram_user must never widen scope.
	res := callMCP(t, c, "memory_search", map[string]any{
		"query":       "shared secret alpha",
		"limit":       float64(10),
		"collections": []any{collection.CollectionUser},
	})
	if res.IsError {
		t.Fatalf("search errored: %s", extractText(res))
	}
	var hits []map[string]any
	if err := json.Unmarshal([]byte(extractText(res)), &hits); err != nil {
		t.Fatalf("decode hits: %v (%s)", err, extractText(res))
	}
	if len(hits) == 0 {
		t.Fatalf("expected the pigo hit")
	}
	for _, h := range hits {
		if got, _ := h["source_collection"].(string); got != collection.CollectionPigo {
			t.Fatalf("pigo search leaked %q", got)
		}
	}
}

func TestMCPHTTP_Delete_CannotTouchOtherCollection(t *testing.T) {
	ts, store := buildMCPTestServer(t)
	seed(store, "u1", collection.CollectionUser, "delete target beta")
	seed(store, "p1", collection.CollectionPigo, "delete target beta")

	c := newMCPClient(t, ts.URL, testPigoKey)
	res := callMCP(t, c, "memory_delete", map[string]any{
		"query":                "delete target beta",
		"similarity_threshold": float64(0.5),
		"limit":                float64(1),
	})
	if res.IsError {
		t.Fatalf("delete errored: %s", extractText(res))
	}
	// The pigo memory IS deleted; the user memory SURVIVES (invisible to pigo).
	store.mu.Lock()
	_, userOK := store.memories["u1"]
	_, pigoOK := store.memories["p1"]
	store.mu.Unlock()
	if !userOK {
		t.Fatalf("pigo delete reached a non-pigo memory (u1 gone)")
	}
	if pigoOK {
		t.Fatalf("pigo delete failed to remove its own memory (p1 survived)")
	}
}

func TestMCPHTTP_Update_CannotTouchOtherCollection(t *testing.T) {
	ts, store := buildMCPTestServer(t)
	seed(store, "u1", collection.CollectionUser, "update target delta")

	c := newMCPClient(t, ts.URL, testPigoKey)
	res := callMCP(t, c, "memory_update", map[string]any{
		"old_content":          "update target delta",
		"new_content":          "replacement for delta",
		"similarity_threshold": float64(0.85),
	})
	if res.IsError {
		t.Fatalf("update errored: %s", extractText(res))
	}
	// The user memory must survive — invisible to the pigo caller, never deleted.
	store.mu.Lock()
	_, ok := store.memories["u1"]
	store.mu.Unlock()
	if !ok {
		t.Fatalf("pigo update reached a non-pigo memory")
	}
}

func TestMCPHTTP_GlobalActions_RejectedForPigo(t *testing.T) {
	ts, _ := buildMCPTestServer(t)
	c := newMCPClient(t, ts.URL, testPigoKey)
	cases := []struct {
		tool string
		args map[string]any
	}{
		{"reflection_run", map[string]any{"dry_run": true}},
		{"reflection_check", map[string]any{}},
		{"memory_apply_config", map[string]any{"config": "{}"}},
		{"memory_reset", map[string]any{"action": "snapshot"}},
	}
	for _, tc := range cases {
		res := callMCP(t, c, tc.tool, tc.args)
		if !res.IsError {
			t.Fatalf("%s should be rejected for pigo, got: %s", tc.tool, extractText(res))
		}
	}
}

func TestMCPHTTP_ReflectionRun_RejectedForPigo(t *testing.T) {
	ts, _ := buildMCPTestServer(t)
	c := newMCPClient(t, ts.URL, testPigoKey)
	res := callMCP(t, c, "reflection_run", map[string]any{"dry_run": true})
	if !res.IsError {
		t.Fatalf("reflection_run should be rejected for pigo, got: %s", extractText(res))
	}
}

// ─────────────────────────────────────────────────────────────
// stdio / ctx-less sanity: non-isolated default caller still fans out.
// ─────────────────────────────────────────────────────────────

func TestMCPHTTP_CtxlessSearchStillFansOut(t *testing.T) {
	collection.DefaultRegistry.Init()
	srv, store := newTestServer()
	seed(store, "u1", collection.CollectionUser, "fanout gamma")
	seed(store, "p1", collection.CollectionPigo, "fanout gamma")

	// Direct handler call with a background context (as stdio would) => "user"
	// caller => not isolated => no forced collection scope => both visible.
	res, err := callTool(srv, "memory_search", map[string]any{"query": "fanout gamma", "limit": float64(10)})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	var hits []map[string]any
	if err := json.Unmarshal([]byte(extractText(res)), &hits); err != nil {
		t.Fatalf("decode: %v", err)
	}
	seen := map[string]bool{}
	for _, h := range hits {
		if c, _ := h["source_collection"].(string); c != "" {
			seen[c] = true
		}
	}
	if !seen[collection.CollectionUser] || !seen[collection.CollectionPigo] {
		t.Fatalf("ctx-less search did not fan out across collections: %v", seen)
	}
}

// seed inserts a memory into the mock store with a fixed id + collection.
func seed(store *mockStore, id, col, content string) {
	emb := newMockEmbedder()
	vec, _ := emb.Embed(context.Background(), content)
	store.mu.Lock()
	defer store.mu.Unlock()
	store.memories[id] = storedPoint{
		mem: memory.Memory{
			ID:         id,
			Type:       memory.TypeEvent,
			Content:    content,
			Collection: col,
			Importance: 5,
			CreatedAt:  1,
			UpdatedAt:  1,
		},
		vector: vec,
	}
}
