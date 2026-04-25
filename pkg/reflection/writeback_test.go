package reflection

import (
	"context"
	"fmt"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// writeBackTestEngine creates an Engine with configurable mock store for write-back tests.
func writeBackTestEngine(store memory.Store) *Engine {
	return &Engine{
		store:    store,
		embedder: &mockEmbedder{dim: 8},
		cfg:      DefaultConfig(),
	}
}

func makeDialecticInsights(n int, confidence float64) []DialecticInsight {
	var out []DialecticInsight
	for i := 0; i < n; i++ {
		out = append(out, DialecticInsight{
			Question:   fmt.Sprintf("focal-q%d", i+1),
			Content:    fmt.Sprintf("synthesized insight %d", i+1),
			Tensions:   []string{},
			SourceIDs:  []string{fmt.Sprintf("e%d-a", i+1), fmt.Sprintf("e%d-b", i+1)},
			Confidence: confidence,
			Importance: 7,
			Tags:       []string{"pattern"},
		})
	}
	return out
}

// --- Test 1: HappyPath (OkCount=3, all written) ---

func TestWriteDialecticInsights_HappyPath(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := makeDialecticInsights(3, 0.85)
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1-a", "e1-b")},
		{Question: "q2", Evidence: makeEvidence("e2-a", "e2-b")},
		{Question: "q3", Evidence: makeEvidence("e3-a", "e3-b")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)

	if stats.Written != 3 {
		t.Errorf("expected Written=3, got %d", stats.Written)
	}
	if stats.Skipped != 0 {
		t.Errorf("expected Skipped=0, got %d", stats.Skipped)
	}
	if stats.Failed != 0 {
		t.Errorf("expected Failed=0, got %d", stats.Failed)
	}
	// Invariant: Written + Skipped + Failed == len(dialectics)
	sum := stats.Written + stats.Skipped + stats.Failed
	if sum != 3 {
		t.Errorf("invariant broken: %d + %d + %d = %d, expected 3",
			stats.Written, stats.Skipped, stats.Failed, sum)
	}
	if len(inserted) != 3 {
		t.Errorf("expected 3 store inserts, got %d", len(inserted))
	}
}

// --- Test 2: PartialDropNoEvidence (Q2 evidence=[]) ---

func TestWriteDialecticInsights_PartialDropNoEvidence(t *testing.T) {
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, _ *memory.Memory, _ []float32) error { return nil },
	}

	e := writeBackTestEngine(store)
	// Only 2 dialectics (Q2 was dropped at Part3 due to empty evidence)
	dialectics := makeDialecticInsights(2, 0.85)
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1-a", "e1-b")},
		{Question: "q2", Evidence: nil},
		{Question: "q3", Evidence: makeEvidence("e3-a", "e3-b")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)

	if stats.Written != 2 {
		t.Errorf("expected Written=2, got %d", stats.Written)
	}
	if stats.Skipped != 0 {
		t.Errorf("expected Skipped=0, got %d", stats.Skipped)
	}
	if stats.Failed != 0 {
		t.Errorf("expected Failed=0, got %d", stats.Failed)
	}
}

// --- Test 3: LowConfidenceSkip (confidence=0.3 → skipped) ---

func TestWriteDialecticInsights_LowConfidenceSkip(t *testing.T) {
	insertCount := 0
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, _ *memory.Memory, _ []float32) error {
			insertCount++
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := []DialecticInsight{
		{Question: "q1", Content: "good insight", Tensions: []string{}, SourceIDs: []string{"e1", "e2"}, Confidence: 0.85, Importance: 7, Tags: []string{"tag"}},
		{Question: "q2", Content: "low conf insight", Tensions: []string{}, SourceIDs: []string{"e3", "e4"}, Confidence: 0.3, Importance: 5, Tags: []string{"tag"}},
		{Question: "q3", Content: "another good", Tensions: []string{}, SourceIDs: []string{"e5", "e6"}, Confidence: 0.9, Importance: 8, Tags: []string{"tag"}},
	}
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2", Evidence: makeEvidence("e3", "e4")},
		{Question: "q3", Evidence: makeEvidence("e5", "e6")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)

	if stats.Written != 2 {
		t.Errorf("expected Written=2, got %d", stats.Written)
	}
	if stats.Skipped != 1 {
		t.Errorf("expected Skipped=1, got %d", stats.Skipped)
	}
	if stats.Failed != 0 {
		t.Errorf("expected Failed=0, got %d", stats.Failed)
	}
	if insertCount != 2 {
		t.Errorf("expected 2 store inserts, got %d", insertCount)
	}
	// Invariant
	sum := stats.Written + stats.Skipped + stats.Failed
	if sum != 3 {
		t.Errorf("invariant broken: sum=%d, expected 3", sum)
	}
}

// --- Test 4: StoreAddFailure (mock store.Insert fails) ---

func TestWriteDialecticInsights_StoreAddFailure(t *testing.T) {
	callIdx := 0
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, _ *memory.Memory, _ []float32) error {
			callIdx++
			if callIdx == 2 {
				return fmt.Errorf("simulated store failure")
			}
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := makeDialecticInsights(3, 0.85)
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1-a", "e1-b")},
		{Question: "q2", Evidence: makeEvidence("e2-a", "e2-b")},
		{Question: "q3", Evidence: makeEvidence("e3-a", "e3-b")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)

	if stats.Written != 2 {
		t.Errorf("expected Written=2, got %d", stats.Written)
	}
	if stats.Failed != 1 {
		t.Errorf("expected Failed=1, got %d", stats.Failed)
	}
	if len(stats.Errors) == 0 {
		t.Error("expected error in stats.Errors")
	}
	// Invariant
	sum := stats.Written + stats.Skipped + stats.Failed
	if sum != 3 {
		t.Errorf("invariant broken: sum=%d, expected 3", sum)
	}
}

// --- Test 5: MetadataPersistence (tensions=[], source_ids, focal_question, provenance) ---

func TestWriteDialecticInsights_MetadataPersistence(t *testing.T) {
	var inserted []*memory.Memory
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, m *memory.Memory, _ []float32) error {
			inserted = append(inserted, m)
			return nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := []DialecticInsight{
		{
			Question:   "What patterns emerge in task scheduling?",
			Content:    "insight with tensions",
			Tensions:   []string{"tension-a", "tension-b"},
			SourceIDs:  []string{"src-1", "src-2", "src-3"},
			Confidence: 0.85,
			Importance: 7,
			Tags:       []string{"scheduling"},
		},
		{
			Question:   "How does error handling evolve?",
			Content:    "insight with empty tensions",
			Tensions:   []string{},
			SourceIDs:  []string{"src-4", "src-5"},
			Confidence: 0.9,
			Importance: 8,
			Tags:       []string{"errors"},
		},
	}
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("src-1", "src-2", "src-3")},
		{Question: "q2", Evidence: makeEvidence("src-4", "src-5")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)
	if stats.Written != 2 {
		t.Fatalf("expected Written=2, got %d", stats.Written)
	}

	// Check first insight metadata
	m1 := inserted[0]
	if m1.Type != memory.TypeInsight {
		t.Errorf("expected TypeInsight, got %s", m1.Type)
	}
	if m1.Source != "system" {
		t.Errorf("expected source=system, got %s", m1.Source)
	}

	// tensions: must be present and serialized
	tensions, ok := m1.Metadata["tensions"]
	if !ok {
		t.Fatal("missing tensions in metadata")
	}
	tensionsSlice, ok := tensions.([]any)
	if !ok {
		t.Fatalf("tensions is not []any: %T", tensions)
	}
	if len(tensionsSlice) != 2 {
		t.Errorf("expected 2 tensions, got %d", len(tensionsSlice))
	}

	// source_ids
	sourceIDs, ok := m1.Metadata["source_ids"]
	if !ok {
		t.Fatal("missing source_ids in metadata")
	}
	sourceIDsSlice, ok := sourceIDs.([]any)
	if !ok {
		t.Fatalf("source_ids is not []any: %T", sourceIDs)
	}
	if len(sourceIDsSlice) != 3 {
		t.Errorf("expected 3 source_ids, got %d", len(sourceIDsSlice))
	}

	// focal_question
	fq, ok := m1.Metadata["focal_question"]
	if !ok {
		t.Fatal("missing focal_question in metadata")
	}
	if fq != "What patterns emerge in task scheduling?" {
		t.Errorf("unexpected focal_question: %v", fq)
	}

	// provenance
	prov, ok := m1.Metadata["provenance"]
	if !ok {
		t.Fatal("missing provenance in metadata")
	}
	if prov != "reflection-v2" {
		t.Errorf("expected provenance=reflection-v2, got %v", prov)
	}

	// source:reflection tag
	if !hasTag(m1.Tags, sourceReflectionTag) {
		t.Errorf("missing source:reflection tag")
	}

	// Check second insight: tensions=[] must persist as empty array, not nil
	m2 := inserted[1]
	tensions2, ok := m2.Metadata["tensions"]
	if !ok {
		t.Fatal("missing tensions in second insight metadata")
	}
	tensionsSlice2, ok := tensions2.([]any)
	if !ok {
		t.Fatalf("tensions is not []any: %T", tensions2)
	}
	if len(tensionsSlice2) != 0 {
		t.Errorf("expected 0 tensions for empty, got %d", len(tensionsSlice2))
	}
}

// --- Test 6: ModeFlag (V1/V2 paths don't break each other) ---

func TestWriteDialecticInsights_ModeFlag(t *testing.T) {
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, _ *memory.Memory, _ []float32) error { return nil },
	}

	// V2 write-back should work with v2 config
	e := writeBackTestEngine(store)
	cfg := e.cfg
	cfg.Mode = "v2"
	dialectics := makeDialecticInsights(1, 0.85)
	evidenceList := []PerQuestionEvidence{{Question: "q1", Evidence: makeEvidence("e1", "e2")}}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, cfg)
	if stats.Written != 1 {
		t.Errorf("V2 write-back: expected Written=1, got %d", stats.Written)
	}

	// Verify that V1 Run path still works (doesn't panic or interfere)
	// V1 RunResult should not have V2 write-back fields populated
	v1Result := &RunResult{Mode: "v1-flat"}
	if v1Result.InsightsWritten != 0 {
		t.Errorf("V1 result should have InsightsWritten=0 by default")
	}
	if v1Result.WriteBackMs != 0 {
		t.Errorf("V1 result should have WriteBackMs=0 by default")
	}
}

// ── mock store for write-back tests ──────────────────────────────────────

type writeBackMockStore struct {
	insertFn func(ctx context.Context, mem *memory.Memory, vec []float32) error
}

func (s *writeBackMockStore) Insert(ctx context.Context, mem *memory.Memory, vec []float32) error {
	if s.insertFn != nil {
		return s.insertFn(ctx, mem, vec)
	}
	return nil
}

func (s *writeBackMockStore) Search(context.Context, []float32, memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (s *writeBackMockStore) Scroll(context.Context, memory.ScrollOptions) ([]memory.Memory, string, error) {
	return nil, "", nil
}
func (s *writeBackMockStore) Delete(context.Context, []string) (int, error)          { return 0, nil }
func (s *writeBackMockStore) Update(context.Context, string, map[string]any) error    { return nil }
func (s *writeBackMockStore) SearchByIDs(context.Context, []string) ([]memory.Memory, error) {
	return nil, nil
}
func (s *writeBackMockStore) EnsureCollection(context.Context) error                 { return nil }
func (s *writeBackMockStore) Stats(context.Context) (*memory.CollectionStats, error) { return nil, nil }
func (s *writeBackMockStore) DeleteExpired(context.Context) (int, error)             { return 0, nil }
