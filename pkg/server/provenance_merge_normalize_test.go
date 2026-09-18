package server

import (
	"context"
	"testing"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
)

// capturingStore wraps the existing mockStore and records the fields passed to
// the most recent Update call, so tests can assert the exact payload shape that
// provenanceMerge hands to the store (which must be qdrant-safe: []any of
// map[string]any, never a []memory.ProvenanceEntry struct slice).
type capturingStore struct {
	*mockStore
	lastUpdateFields map[string]any
}

func (c *capturingStore) Update(ctx context.Context, id string, fields map[string]any) error {
	c.lastUpdateFields = fields
	return c.mockStore.Update(ctx, id, fields)
}

// TestProvenanceMerge_UpdateFieldsNormalized drives provenanceMerge with an
// existing memory whose provenance_history is stored both in the in-process
// []memory.ProvenanceEntry form and the JSON round-tripped []any-of-map form.
// It asserts no panic, that Update receives metadata.provenance_history as a
// []any of map[string]any with keys source_type/merged_at/content_score, and
// that the promoted primary source_type is correct.
func TestProvenanceMerge_UpdateFieldsNormalized(t *testing.T) {
	cases := []struct {
		name string
		ph   any
	}{
		{
			name: "struct-slice",
			ph:   []memory.ProvenanceEntry{{SourceType: "reflection", MergedAt: 100, ContentScore: 0.9}},
		},
		{
			name: "any-roundtripped",
			ph: []any{map[string]any{
				"source_type":   "reflection",
				"merged_at":     float64(100),
				"content_score": 0.9,
			}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cap := &capturingStore{mockStore: newMockStore()}
			cfg := &config.Config{
				Weights:        memory.DefaultScoringWeights(),
				Decay:          memory.DefaultDecayConfig(),
				MMRLambda:      0.5,
				DedupThreshold: 0.92,
			}
			srv := NewServer(cap, newMockEmbedder(), cfg)

			existing := &memory.ScoredMemory{
				Memory: memory.Memory{
					ID:      "m1",
					Content: "Engram uses Qdrant",
					Metadata: map[string]any{
						"source_type":        "reflection",
						"provenance_history": tc.ph,
					},
				},
				Score: 0.95,
			}
			if err := cap.Insert(context.Background(), &existing.Memory, []float32{1}); err != nil {
				t.Fatalf("seed insert: %v", err)
			}

			primary, merged, err := srv.provenanceMerge(context.Background(), existing, "user_input")
			if err != nil {
				t.Fatalf("provenanceMerge error: %v", err)
			}
			if !merged {
				t.Fatalf("expected merged=true")
			}
			if primary != "user_input" {
				t.Errorf("expected primary user_input, got %q", primary)
			}

			raw, ok := cap.lastUpdateFields["metadata"]
			if !ok {
				t.Fatalf("Update did not receive metadata")
			}
			md, ok := raw.(map[string]any)
			if !ok {
				t.Fatalf("metadata is %T, want map[string]any", raw)
			}
			if md["source_type"] != "user_input" {
				t.Errorf("nested source_type = %v, want user_input", md["source_type"])
			}
			slice, ok := md["provenance_history"].([]any)
			if !ok {
				t.Fatalf("metadata.provenance_history is %T, want []any", md["provenance_history"])
			}
			if len(slice) == 0 {
				t.Fatalf("expected non-empty provenance_history")
			}
			for i, e := range slice {
				m, ok := e.(map[string]any)
				if !ok {
					t.Fatalf("element %d is %T, want map[string]any", i, e)
				}
				if _, ok := m["source_type"]; !ok {
					t.Errorf("element %d missing source_type", i)
				}
				if _, ok := m["merged_at"]; !ok {
					t.Errorf("element %d missing merged_at", i)
				}
				if _, ok := m["content_score"]; !ok {
					t.Errorf("element %d missing content_score", i)
				}
			}
			// last appended entry is the incoming user_input source
			last := slice[len(slice)-1].(map[string]any)
			if last["source_type"] != "user_input" {
				t.Errorf("last entry source_type = %v, want user_input", last["source_type"])
			}
		})
	}
}

// TestProvenanceMerge_PreservesSiblingMetadata (R2) asserts that a provenance
// merge preserves unrelated metadata sibling keys (the RMW clone must not drop
// them) while adding provenance_history and the promoted source_type.
func TestProvenanceMerge_PreservesSiblingMetadata(t *testing.T) {
	cap := &capturingStore{mockStore: newMockStore()}
	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
	}
	srv := NewServer(cap, newMockEmbedder(), cfg)

	existing := &memory.ScoredMemory{
		Memory: memory.Memory{
			ID:      "sib1",
			Content: "Engram uses Qdrant",
			Metadata: map[string]any{
				"source_type": "reflection",
				"city":        "Berlin",
			},
		},
		Score: 0.96,
	}
	if err := cap.Insert(context.Background(), &existing.Memory, []float32{1}); err != nil {
		t.Fatalf("seed insert: %v", err)
	}

	primary, merged, err := srv.provenanceMerge(context.Background(), existing, "user_input")
	if err != nil {
		t.Fatalf("provenanceMerge error: %v", err)
	}
	if !merged {
		t.Fatalf("expected merged=true")
	}
	if primary != "user_input" {
		t.Errorf("expected primary user_input, got %q", primary)
	}

	md, ok := cap.lastUpdateFields["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("Update metadata is %T, want map[string]any", cap.lastUpdateFields["metadata"])
	}
	if md["city"] != "Berlin" {
		t.Errorf("sibling key city dropped: got %v", md["city"])
	}
	if md["source_type"] != "user_input" {
		t.Errorf("source_type = %v, want user_input", md["source_type"])
	}
	ph, ok := md["provenance_history"].([]any)
	if !ok || len(ph) == 0 {
		t.Fatalf("provenance_history missing/empty: %T %v", md["provenance_history"], md["provenance_history"])
	}
	// Confirm the stored point retains the sibling too (read-back).
	stored := findStored(t, cap.mockStore, "Engram uses Qdrant")
	if stored.Metadata["city"] != "Berlin" {
		t.Errorf("stored sibling key city dropped: got %v", stored.Metadata["city"])
	}
}
