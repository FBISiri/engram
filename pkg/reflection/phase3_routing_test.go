// phase3_routing_test.go — W20 Day2 Phase 3.
//
// Verifies that the reflection engine tags inserted insights with the
// metadata markers required for Phase 4 physical-isolation routing:
//   - caller_type        = "reflection"
//   - target_collection  = "engram_reflection"
//
// In-process callers (the engine itself) don't traverse HTTP middleware,
// so we cannot rely on X-Caller-Type. The metadata markers are the
// stable contract Phase 4's Store layer will read.
package reflection

import (
	"context"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

func TestPhase3_DialecticInsight_TaggedReflectionRouting(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := makeDialecticInsights(1, 0.85)
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1-a", "e1-b")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)
	if stats.Written != 1 {
		t.Fatalf("want Written=1, got %d", stats.Written)
	}
	if len(inserted) != 1 {
		t.Fatalf("want 1 insert, got %d", len(inserted))
	}

	mem := inserted[0]
	if got, _ := mem.Metadata["caller_type"].(string); got != "reflection" {
		t.Errorf("caller_type metadata: want 'reflection', got %q", got)
	}
	if got, _ := mem.Metadata["target_collection"].(string); got != "engram_reflection" {
		t.Errorf("target_collection metadata: want 'engram_reflection', got %q", got)
	}
}
