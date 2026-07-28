package reflection

import (
	"context"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// C1 (a): importance cap enforcement — an insight with Importance above the
// configured MaxInsightImportance is clamped before store.Insert.
func TestWriteDialecticInsights_ImportanceCap(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	cfg := e.cfg
	cfg.MaxInsightImportance = 8

	dialectics := []DialecticInsight{{
		Question:   "q1",
		Content:    "over-important insight",
		Tensions:   []string{},
		SourceIDs:  []string{"e1", "e2"},
		Confidence: 0.85,
		Importance: 10, // above cap
		Tags:       []string{"pattern"},
	}}
	evidenceList := []PerQuestionEvidence{{Question: "q1", Evidence: makeEvidence("e1", "e2")}}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, cfg)
	if stats.Written != 1 {
		t.Fatalf("expected Written=1, got %d", stats.Written)
	}
	if len(inserted) != 1 {
		t.Fatalf("expected 1 insert, got %d", len(inserted))
	}
	if inserted[0].Importance > 8 {
		t.Errorf("expected clamped importance <= 8, got %.0f", inserted[0].Importance)
	}
	if inserted[0].Importance != 8 {
		t.Errorf("expected importance clamped to exactly 8, got %.0f", inserted[0].Importance)
	}
}

// C1 (a'): default cap (8) is applied when MaxInsightImportance is unset (0).
func TestWriteDialecticInsights_ImportanceCapDefault(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	cfg := e.cfg
	cfg.MaxInsightImportance = 0 // exercise fallback default

	dialectics := []DialecticInsight{{
		Question:   "q1",
		Content:    "over-important insight",
		Tensions:   []string{},
		SourceIDs:  []string{"e1", "e2"},
		Confidence: 0.85,
		Importance: 10,
		Tags:       []string{"pattern"},
	}}
	evidenceList := []PerQuestionEvidence{{Question: "q1", Evidence: makeEvidence("e1", "e2")}}

	e.writeDialecticInsights(context.Background(), dialectics, evidenceList, cfg)
	if len(inserted) != 1 {
		t.Fatalf("expected 1 insert, got %d", len(inserted))
	}
	if inserted[0].Importance != 8 {
		t.Errorf("expected fallback default cap of 8, got %.0f", inserted[0].Importance)
	}
}

// C1 (b): every insight memory carries source_type=reflection metadata (EU AI
// Act provenance requirement).
func TestWriteDialecticInsights_SourceTypeMetadata(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := makeDialecticInsights(2, 0.85)
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1-a", "e1-b")},
		{Question: "q2", Evidence: makeEvidence("e2-a", "e2-b")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)
	if stats.Written != 2 {
		t.Fatalf("expected Written=2, got %d", stats.Written)
	}
	if len(inserted) != 2 {
		t.Fatalf("expected 2 inserts, got %d", len(inserted))
	}
	for i, m := range inserted {
		st, ok := m.Metadata["source_type"]
		if !ok {
			t.Errorf("insight %d: missing source_type metadata", i)
			continue
		}
		if st != "reflection" {
			t.Errorf("insight %d: expected source_type=reflection, got %v", i, st)
		}
	}
}

// applyMockFilters simulates a store applying an OpIn filter on
// metadata.source_type: memories whose source_type is not in the allowed set
// (including memories lacking the field) are excluded.
func applyMockFilters(mems []memory.ScoredMemory, filters []memory.Filter) []memory.ScoredMemory {
	for _, f := range filters {
		if f.Field != "metadata.source_type" || f.Op != memory.OpIn {
			continue
		}
		allowed, _ := f.Value.([]string)
		allowSet := make(map[string]struct{}, len(allowed))
		for _, a := range allowed {
			allowSet[a] = struct{}{}
		}
		var kept []memory.ScoredMemory
		for _, sm := range mems {
			st, ok := sm.Metadata["source_type"].(string)
			if !ok {
				continue // no source_type → excluded by OpIn
			}
			if _, in := allowSet[st]; in {
				kept = append(kept, sm)
			}
		}
		mems = kept
	}
	return mems
}

func scoredWithSourceType(id, sourceType string, createdAt float64) memory.ScoredMemory {
	m := memory.Memory{
		ID:         id,
		Content:    "content-" + id,
		Type:       memory.TypeEvent,
		Importance: 5,
		Confidence: 0.8,
		CreatedAt:  createdAt,
	}
	if sourceType != "" {
		m.Metadata = map[string]any{"source_type": sourceType}
	}
	return memory.ScoredMemory{Memory: m, Score: 0.9}
}

// C1 (c) + (d): RequireProvenance builds an OpIn filter on metadata.source_type
// and memories without a matching source_type (including those lacking the
// field entirely) are excluded.
func TestRetrieveEvidence_RequireProvenanceFilter(t *testing.T) {
	old := float64(time.Now().Add(-30 * 24 * time.Hour).Unix())
	mems := []memory.ScoredMemory{
		scoredWithSourceType("user1", "user_input", old),
		scoredWithSourceType("web1", "web_search", old),
		scoredWithSourceType("refl1", "reflection", old), // not allowed → excluded
		scoredWithSourceType("nometa", "", old),          // no source_type → excluded
	}

	var captured []memory.Filter
	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			captured = opts.Filters
			return applyMockFilters(mems, opts.Filters), nil
		},
	}

	cfg := Config{
		EvidencePerFocal:   10,
		RequireProvenance:  true,
		AllowedProvenances: []string{"user_input", "web_search"},
	}
	result, _, err := retrieveEvidence(context.Background(), "q", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// (c) the store received an OpIn filter on metadata.source_type.
	var found *memory.Filter
	for i := range captured {
		if captured[i].Field == "metadata.source_type" {
			found = &captured[i]
			break
		}
	}
	if found == nil {
		t.Fatal("expected a metadata.source_type filter to be passed to the store")
	}
	if found.Op != memory.OpIn {
		t.Errorf("expected OpIn, got %s", found.Op)
	}
	vals, ok := found.Value.([]string)
	if !ok || len(vals) != 2 || vals[0] != "user_input" || vals[1] != "web_search" {
		t.Errorf("unexpected filter value: %v", found.Value)
	}

	// (d) memories without a matching source_type are excluded.
	if len(result) != 2 {
		t.Fatalf("expected 2 allowed memories, got %d", len(result))
	}
	for _, m := range result {
		if m.ID == "refl1" || m.ID == "nometa" {
			t.Errorf("memory %s should have been excluded by provenance filter", m.ID)
		}
	}
}

// C1 (c'): when RequireProvenance is false, no source_type filter is applied.
func TestRetrieveEvidence_NoProvenanceFilterWhenDisabled(t *testing.T) {
	old := float64(time.Now().Add(-30 * 24 * time.Hour).Unix())
	mems := []memory.ScoredMemory{
		scoredWithSourceType("nometa", "", old),
	}

	var captured []memory.Filter
	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			captured = opts.Filters
			return applyMockFilters(mems, opts.Filters), nil
		},
	}

	cfg := Config{EvidencePerFocal: 10, RequireProvenance: false}
	result, _, err := retrieveEvidence(context.Background(), "q", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, f := range captured {
		if f.Field == "metadata.source_type" {
			t.Errorf("did not expect a source_type filter when RequireProvenance=false")
		}
	}
	if len(result) != 1 {
		t.Errorf("expected the no-metadata memory to pass through, got %d results", len(result))
	}
}
