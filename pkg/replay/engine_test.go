package replay

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/FBISiri/engram/pkg/trajectory"
)

func TestEngine_GuardRejectsNonEvalCollection(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	for _, bad := range []string{"engram_user", "siri", "bmo", "production", ""} {
		eng := &Engine{BaseURL: srv.URL, Collection: bad, Client: srv.Client()}
		_, err := eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{})
		if err == nil {
			t.Errorf("collection %q: expected guard error, got nil", bad)
		}
		if _, ok := err.(*GuardViolation); !ok {
			t.Errorf("collection %q: expected *GuardViolation, got %T", bad, err)
		}
	}
	if called {
		t.Error("guard was bypassed: HTTP endpoint was hit for a non-eval collection")
	}
}

func TestEngine_SearchRequestShape(t *testing.T) {
	var gotPath, gotAuth string
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/collections" {
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{}`))
			return
		}
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &gotBody)
		resp := []trajectory.ResultItem{{ID: "a", Content: "c", Score: 0.9}}
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	eng := &Engine{BaseURL: srv.URL, APIKey: "secret", Collection: "engram_eval_replay", Client: srv.Client()}
	// Record 15 items so the engine requests a limit above the topK floor.
	rec := make([]trajectory.ResultItem, 15)
	for i := range rec {
		rec[i] = trajectory.ResultItem{ID: string(rune('a' + i))}
	}
	results, err := eng.Run(context.Background(), []ReplayCase{{Query: "hello", RecordedResults: rec}}, RunOptions{TopK: 5})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if gotPath != "/memories/search" {
		t.Errorf("path = %q, want /memories/search", gotPath)
	}
	if gotAuth != "Bearer secret" {
		t.Errorf("auth = %q, want Bearer secret", gotAuth)
	}
	if gotBody["query"] != "hello" {
		t.Errorf("body query = %v, want hello", gotBody["query"])
	}
	if gotBody["collection"] != "engram_eval_replay" {
		t.Errorf("body collection = %v", gotBody["collection"])
	}
	// limit must be max(len(recorded)=15, topK floor=5) = 15, not capped at 5.
	if gotBody["limit"].(float64) != 15 {
		t.Errorf("body limit = %v, want 15 (not capped by topK)", gotBody["limit"])
	}
	if len(results) != 1 || len(results[0].LiveResults) != 1 || results[0].LiveResults[0].ID != "a" {
		t.Errorf("results parsed wrong: %+v", results)
	}
}

func TestEngine_RestoresConfigOnExit(t *testing.T) {
	var applied []map[string]any // arguments of each memory_apply_config call
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/collections":
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{}`))
		case "/mcp":
			data, _ := io.ReadAll(r.Body)
			var req struct {
				Params struct {
					Name      string         `json:"name"`
					Arguments map[string]any `json:"arguments"`
				} `json:"params"`
			}
			_ = json.Unmarshal(data, &req)
			if req.Params.Name == "memory_apply_config" {
				applied = append(applied, req.Params.Arguments)
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{}"}]}}`))
		default:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
		}
	}))
	defer srv.Close()

	eng := &Engine{BaseURL: srv.URL, Collection: "engram_eval_replay", Client: srv.Client()}
	override := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.25}}
	baseline := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.5}}
	_, err := eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{
		ConfigOverride: &override,
		BaselineConfig: &baseline,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Expect exactly two apply_config calls: override first, baseline restore last.
	if len(applied) != 2 {
		t.Fatalf("apply_config calls = %d, want 2 (override + restore)", len(applied))
	}
	if !hasRecency(applied[0]["config"], 0.25) {
		t.Errorf("first apply = %v, want override 0.25", applied[0])
	}
	if !hasRecency(applied[1]["config"], 0.5) {
		t.Errorf("last apply = %v, want baseline restore 0.5", applied[1])
	}
}

func TestEngine_RestoresStateOnExit(t *testing.T) {
	var resets []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/collections":
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{}`))
		case "/mcp":
			data, _ := io.ReadAll(r.Body)
			var req struct {
				Params struct {
					Name      string         `json:"name"`
					Arguments map[string]any `json:"arguments"`
				} `json:"params"`
			}
			_ = json.Unmarshal(data, &req)
			if req.Params.Name == "memory_reset" {
				resets = append(resets, req.Params.Arguments)
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{\"snapshot_id\":\"fresh-123\"}"}]}}`))
		default:
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
		}
	}))
	defer srv.Close()

	eng := &Engine{BaseURL: srv.URL, Collection: "engram_eval_replay", Client: srv.Client()}
	_, err := eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{
		SnapshotID: "prod-baseline",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// Sequence: snapshot (capture), restore prod-baseline, restore fresh-123 (defer).
	if len(resets) != 3 {
		t.Fatalf("memory_reset calls = %d, want 3, got %v", len(resets), resets)
	}
	if resets[0]["action"] != "snapshot" {
		t.Errorf("first reset = %v, want action=snapshot", resets[0])
	}
	if resets[1]["snapshot_id"] != "prod-baseline" {
		t.Errorf("second reset = %v, want restore prod-baseline", resets[1])
	}
	if resets[2]["snapshot_id"] != "fresh-123" {
		t.Errorf("last reset = %v, want restore fresh-123 (captured original)", resets[2])
	}
}

// hasRecency reports whether the marshalled config string carries the wanted
// recency weight value.
func hasRecency(config any, want float64) bool {
	s, _ := config.(string)
	var cfg MemoryConfig
	if json.Unmarshal([]byte(s), &cfg) != nil {
		return false
	}
	return cfg.RetrieveConfig.RecencyWeight == want
}

func TestEngine_ConfigWithoutBaseline_FailsClosed(t *testing.T) {
	mutated := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/collections" {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{}`))
			return
		}
		if r.URL.Path == "/mcp" {
			mutated = true // any apply_config/reset means we mutated the server
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[]`))
	}))
	defer srv.Close()

	eng := &Engine{BaseURL: srv.URL, Collection: "engram_eval_replay", Client: srv.Client()}
	override := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.25}}

	// nil baseline → refuse before mutating.
	_, err := eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{ConfigOverride: &override})
	if err == nil {
		t.Fatal("expected fail-closed error with nil baseline, got nil")
	}
	// zero baseline → also refuse (would be a no-op restore).
	zero := MemoryConfig{}
	_, err = eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{ConfigOverride: &override, BaselineConfig: &zero})
	if err == nil {
		t.Fatal("expected fail-closed error with zero baseline, got nil")
	}
	if mutated {
		t.Error("server was mutated before the fail-closed check — must refuse pre-mutation")
	}
}

func TestEngine_ToolIsError_ReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/collections" {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{}`))
			return
		}
		// tools/call returns HTTP 200 but result.isError=true (NewToolResultError).
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"isError":true,"content":[{"type":"text","text":"boom"}]}}`))
	}))
	defer srv.Close()

	eng := &Engine{BaseURL: srv.URL, Collection: "engram_eval_replay", Client: srv.Client()}
	override := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.25}}
	baseline := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.5}}
	_, err := eng.Run(context.Background(), []ReplayCase{{Query: "q"}}, RunOptions{
		ConfigOverride: &override,
		BaselineConfig: &baseline,
	})
	if err == nil {
		t.Fatal("expected error when tool result.isError=true, got nil")
	}
}
