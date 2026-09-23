package server

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
)

// seedMem inserts a memory (with an embedded vector) directly into the store.
func seedMem(t *testing.T, srv *Server, store *mockStore, m memory.Memory) {
	t.Helper()
	vec, err := srv.embedder.Embed(context.Background(), m.Content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if err := store.Insert(context.Background(), &m, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}
}

// idsFromArray parses a JSON array response body and returns the result[].id values.
func idsFromArray(t *testing.T, body string) []string {
	t.Helper()
	var arr []map[string]any
	if err := json.Unmarshal([]byte(body), &arr); err != nil {
		t.Fatalf("unmarshal array %q: %v", body, err)
	}
	ids := make([]string, 0, len(arr))
	for _, o := range arr {
		id, _ := o["id"].(string)
		ids = append(ids, id)
	}
	return ids
}

func contains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func callHandler(t *testing.T, h func(context.Context, mcp.CallToolRequest) (*mcp.CallToolResult, error), args map[string]any) string {
	t.Helper()
	req := mcp.CallToolRequest{Params: mcp.CallToolParams{Arguments: args}}
	res, err := h(context.Background(), req)
	if err != nil {
		t.Fatalf("handler err: %v", err)
	}
	if res.IsError {
		t.Fatalf("handler returned error result: %s", extractText(res))
	}
	return extractText(res)
}

func seedPair(t *testing.T, srv *Server, store *mockStore) {
	t.Helper()
	seedMem(t, srv, store, memory.Memory{
		ID: "A", Type: memory.TypeEvent, Content: "superseded content about frank",
		Source: "agent", Importance: 5, Tags: []string{"x"}, CreatedAt: 100,
		Collection: "engram_user", SupersededBy: "B",
	})
	seedMem(t, srv, store, memory.Memory{
		ID: "N", Type: memory.TypeEvent, Content: "normal content about frank",
		Source: "agent", Importance: 5, Tags: []string{"x"}, CreatedAt: 200,
		Collection: "engram_user",
	})
}

// V1 + V3
func TestSearchSupersededVisibility(t *testing.T) {
	srv, store := newTestServer()
	seedPair(t, srv, store)

	body := callHandler(t, srv.handleSearch, map[string]any{"query": "frank", "limit": float64(10)})
	if ids := idsFromArray(t, body); contains(ids, "A") {
		t.Errorf("V1: default search must NOT contain superseded A; ids=%v", ids)
	}

	body = callHandler(t, srv.handleSearch, map[string]any{"query": "frank", "limit": float64(10), "include_superseded": true})
	if ids := idsFromArray(t, body); !contains(ids, "A") {
		t.Errorf("V3: include_superseded search MUST contain A; ids=%v", ids)
	}
}

// V4 + normal-id
func TestGetByID(t *testing.T) {
	srv, store := newTestServer()
	seedPair(t, srv, store)

	body := callHandler(t, srv.handleGetByID, map[string]any{"id": "A"})
	var obj map[string]any
	if err := json.Unmarshal([]byte(body), &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if obj["id"] != "A" {
		t.Errorf("V4: want id A, got %v", obj["id"])
	}
	if obj["superseded_by"] != "B" {
		t.Errorf("V4: want superseded_by B, got %v", obj["superseded_by"])
	}

	// normal (non-superseded) id still works, superseded_by omitted
	body = callHandler(t, srv.handleGetByID, map[string]any{"id": "N"})
	obj = nil
	if err := json.Unmarshal([]byte(body), &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if obj["id"] != "N" {
		t.Errorf("want id N, got %v", obj["id"])
	}
	if _, ok := obj["superseded_by"]; ok {
		t.Errorf("normal memory must omit superseded_by, got %v", obj["superseded_by"])
	}

	// missing id => null
	body = callHandler(t, srv.handleGetByID, map[string]any{"id": "ZZZ"})
	if body != "null" {
		t.Errorf("missing id want null, got %q", body)
	}
}

// V5
func TestListSupersededTool(t *testing.T) {
	srv, store := newTestServer()
	seedPair(t, srv, store)

	body := callHandler(t, srv.handleListSuperseded, map[string]any{"limit": float64(50)})
	var arr []map[string]any
	if err := json.Unmarshal([]byte(body), &arr); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	ids := idsFromArray(t, body)
	if !contains(ids, "A") {
		t.Errorf("V5: list_superseded must contain A; ids=%v", ids)
	}
	if contains(ids, "N") {
		t.Errorf("V5: list_superseded must NOT contain normal N; ids=%v", ids)
	}
	for _, o := range arr {
		if s, _ := o["superseded_by"].(string); s == "" {
			t.Errorf("V5: every item must carry non-empty superseded_by; got %v", o)
		}
	}
}

// V6
func TestListSupersededVisibility(t *testing.T) {
	srv, store := newTestServer()
	seedPair(t, srv, store)

	body := callHandler(t, srv.handleList, map[string]any{"limit": float64(50)})
	if ids := idsFromArray(t, body); contains(ids, "A") {
		t.Errorf("V6: default list must NOT contain superseded A; ids=%v", ids)
	}

	body = callHandler(t, srv.handleList, map[string]any{"limit": float64(50), "include_superseded": true})
	if ids := idsFromArray(t, body); !contains(ids, "A") {
		t.Errorf("V6: include_superseded list MUST contain A; ids=%v", ids)
	}
}
