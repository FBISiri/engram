package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/qdrant"
)

// grpcURL is the Qdrant gRPC address used by the regression test, overridable
// via ENGRAM_TEST_QDRANT_URL. restURL is the REST address used only to inspect
// the RAW stored payload (to prove no dotted literal key leaks through).
const (
	defaultGRPCURL = "127.0.0.1:6334"
	restBaseURL    = "http://127.0.0.1:6333"
)

// newProvMergeQdrantStore constructs a throwaway-collection qdrant.Store and
// skips the test when Qdrant is unreachable so CI without Qdrant does not fail.
func newProvMergeQdrantStore(t *testing.T) *qdrant.Store {
	t.Helper()
	url := os.Getenv("ENGRAM_TEST_QDRANT_URL")
	if url == "" {
		url = defaultGRPCURL
	}
	conn, err := net.DialTimeout("tcp", url, 500*time.Millisecond)
	if err != nil {
		t.Skipf("qdrant not available at %s: %v", url, err)
	}
	_ = conn.Close()

	coll := fmt.Sprintf("engram_test_provmerge_%d", time.Now().UnixNano())
	store, err := qdrant.New(qdrant.Config{URL: url, CollectionName: coll, Dimension: 8})
	if err != nil {
		t.Skipf("qdrant connect failed: %v", err)
	}
	if err := store.EnsureCollection(context.Background()); err != nil {
		t.Skipf("qdrant ensure collection failed: %v", err)
	}
	t.Cleanup(func() {
		_ = store.DropCollection(context.Background())
	})
	return store
}

// rawPayload fetches the raw stored payload for a point via the REST API, so we
// can assert no top-level dotted literal key ("metadata.source_type" /
// "metadata.provenance_history") remains after the merge.
func rawPayload(t *testing.T, coll, id string) map[string]any {
	t.Helper()
	url := fmt.Sprintf("%s/collections/%s/points/%s", restBaseURL, coll, id)
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("rest get: %v", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("rest get status %d: %s", resp.StatusCode, body)
	}
	var parsed struct {
		Result struct {
			Payload map[string]any `json:"payload"`
		} `json:"result"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("rest decode: %v", err)
	}
	return parsed.Result.Payload
}

func assertNoDottedKeys(t *testing.T, coll, id string) {
	t.Helper()
	payload := rawPayload(t, coll, id)
	for _, k := range []string{"metadata.source_type", "metadata.provenance_history"} {
		if _, ok := payload[k]; ok {
			t.Errorf("raw payload retains dotted top-level key %q: %v", k, payload)
		}
	}
}

// insertWithSourceType inserts a fresh memory carrying metadata.source_type and
// returns its read-back form (as the store round-trips it).
func insertWithSourceType(t *testing.T, store *qdrant.Store, sourceType string) memory.Memory {
	t.Helper()
	mem := memory.New("Engram uses Qdrant as its vector store",
		memory.WithType(memory.TypeIdentity),
		memory.WithSource("agent"),
		memory.WithMetadata(map[string]any{"source_type": sourceType}),
	)
	vec := make([]float32, 8)
	for i := range vec {
		vec[i] = 0.1
	}
	if err := store.Insert(context.Background(), mem, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}
	got, err := store.SearchByIDs(context.Background(), []string{mem.ID})
	if err != nil || len(got) != 1 {
		t.Fatalf("read-back after insert: err=%v n=%d", err, len(got))
	}
	return got[0]
}

// TestProvenanceMerge_RealQdrant is the regression test for the dotted-key bug:
// a mock store cannot catch it because the bug is in how Qdrant persists dotted
// literal payload keys. Runs against real Qdrant with a throwaway collection.
func TestProvenanceMerge_RealQdrant(t *testing.T) {
	store := newProvMergeQdrantStore(t)
	coll := store.CollectionName()
	srv := &Server{store: store}
	ctx := context.Background()

	t.Run("basic_merge_visibility", func(t *testing.T) {
		existing := insertWithSourceType(t, store, "tool_output")
		sm := &memory.ScoredMemory{Memory: existing, Score: 0.97}
		primary, merged, err := srv.provenanceMerge(ctx, sm, "reflection")
		if err != nil {
			t.Fatalf("provenanceMerge: %v", err)
		}
		if !merged {
			t.Fatalf("expected merged=true")
		}
		// tool_output (rank 2) beats reflection (rank 6): primary stays tool_output.
		if primary != "tool_output" {
			t.Errorf("primary = %q, want tool_output", primary)
		}

		got, err := store.SearchByIDs(ctx, []string{existing.ID})
		if err != nil || len(got) != 1 {
			t.Fatalf("read-back: err=%v n=%d", err, len(got))
		}
		md := got[0].Metadata
		if st, _ := md["source_type"].(string); st != "tool_output" {
			t.Errorf("nested source_type = %q, want tool_output", st)
		}
		ph, ok := md["provenance_history"].([]any)
		if !ok || len(ph) < 1 {
			t.Fatalf("provenance_history not visible in nested metadata: %T %v", md["provenance_history"], md["provenance_history"])
		}
		assertNoDottedKeys(t, coll, existing.ID)
	})

	t.Run("trust_promotion", func(t *testing.T) {
		existing := insertWithSourceType(t, store, "reflection")
		sm := &memory.ScoredMemory{Memory: existing, Score: 0.98}
		// user_input (rank 1) beats reflection (rank 6): primary promotes.
		primary, merged, err := srv.provenanceMerge(ctx, sm, "user_input")
		if err != nil {
			t.Fatalf("provenanceMerge: %v", err)
		}
		if !merged {
			t.Fatalf("expected merged=true")
		}
		if primary != "user_input" {
			t.Errorf("primary = %q, want user_input", primary)
		}

		got, err := store.SearchByIDs(ctx, []string{existing.ID})
		if err != nil || len(got) != 1 {
			t.Fatalf("read-back: err=%v n=%d", err, len(got))
		}
		md := got[0].Metadata
		if st, _ := md["source_type"].(string); st != "user_input" {
			t.Errorf("nested source_type = %q, want promoted user_input", st)
		}
		if !hasSourceInHistory(md, "user_input") {
			t.Errorf("provenance_history missing user_input entry: %v", md["provenance_history"])
		}
		assertNoDottedKeys(t, coll, existing.ID)
	})

	t.Run("idempotency", func(t *testing.T) {
		existing := insertWithSourceType(t, store, "tool_output")
		sm := &memory.ScoredMemory{Memory: existing, Score: 0.97}
		if _, merged, err := srv.provenanceMerge(ctx, sm, "reflection"); err != nil || !merged {
			t.Fatalf("first merge: merged=%v err=%v", merged, err)
		}

		got, err := store.SearchByIDs(ctx, []string{existing.ID})
		if err != nil || len(got) != 1 {
			t.Fatalf("read-back: err=%v n=%d", err, len(got))
		}
		if n := historyLen(got[0].Metadata); n != 1 {
			t.Fatalf("after first merge provenance_history len = %d, want 1", n)
		}

		// Merge the SAME source again off the fresh read-back: must be a no-op.
		sm2 := &memory.ScoredMemory{Memory: got[0], Score: 0.97}
		if _, merged, err := srv.provenanceMerge(ctx, sm2, "reflection"); err != nil {
			t.Fatalf("second merge err: %v", err)
		} else if merged {
			t.Errorf("second merge returned merged=true, want false (idempotent)")
		}

		got2, err := store.SearchByIDs(ctx, []string{existing.ID})
		if err != nil || len(got2) != 1 {
			t.Fatalf("read-back 2: err=%v n=%d", err, len(got2))
		}
		if n := historyLen(got2[0].Metadata); n != 1 {
			t.Errorf("after idempotent second merge provenance_history len = %d, want 1", n)
		}
		assertNoDottedKeys(t, coll, existing.ID)
	})
}

func historyLen(md map[string]any) int {
	ph, _ := md["provenance_history"].([]any)
	return len(ph)
}

func hasSourceInHistory(md map[string]any, st string) bool {
	ph, _ := md["provenance_history"].([]any)
	for _, e := range ph {
		m, ok := e.(map[string]any)
		if !ok {
			continue
		}
		if s, _ := m["source_type"].(string); s == st {
			return true
		}
	}
	return false
}
