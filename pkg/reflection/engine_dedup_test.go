package reflection

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── mock store for pre-write dedup tests ─────────────────────────────────────

// dedupMockStore satisfies memory.Store. It lets tests inject a Search result
// (to drive the pre-write dedup branch) and counts Insert calls so a dedup
// skip vs. a real write can be distinguished.
type dedupMockStore struct {
	searchFn    func(ctx context.Context, vec []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error)
	inserts     int
	searchCalls int
	memories    []memory.Memory // returned from Scroll (unreflected source memories)
}

func (s *dedupMockStore) Insert(_ context.Context, _ *memory.Memory, _ []float32) error {
	s.inserts++
	return nil
}

func (s *dedupMockStore) Search(ctx context.Context, vec []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	s.searchCalls++
	if s.searchFn != nil {
		return s.searchFn(ctx, vec, opts)
	}
	return nil, nil
}

func (s *dedupMockStore) Scroll(_ context.Context, _ memory.ScrollOptions) ([]memory.Memory, string, error) {
	return s.memories, "", nil
}

func (s *dedupMockStore) Update(_ context.Context, _ string, _ map[string]any) error { return nil }
func (s *dedupMockStore) Delete(_ context.Context, _ []string) (int, error)          { return 0, nil }

func (s *dedupMockStore) SearchByIDs(_ context.Context, ids []string) ([]memory.Memory, error) {
	// Echo back source memories so mark-reflected does not fail.
	return s.memories, nil
}
func (s *dedupMockStore) EnsureCollection(_ context.Context) error { return nil }
func (s *dedupMockStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{PointCount: uint64(len(s.memories)), Status: "green"}, nil
}
func (s *dedupMockStore) DeleteExpired(_ context.Context) (int, error) { return 0, nil }

// oneInsightLLM returns a single high-confidence insight block that the parser
// accepts (confidence >= 0.6 so it is NOT diverted to an Obsidian draft).
const oneInsightLLM = `---
INSIGHT: Siri should batch similar tasks to reduce coordination overhead.
IMPORTANCE: 7
CONFIDENCE: 0.9
TAGS: efficiency, batching
---`

// newDedupSources builds N unreflected source memories so the V1 batch trigger
// fires and validateEvidenceGrounding (>=2) passes.
func newDedupSources(n int) []memory.Memory {
	mems := make([]memory.Memory, n)
	for i := range mems {
		mems[i] = memory.Memory{
			ID:         memory.New("src").ID,
			Type:       memory.TypeEvent,
			Content:    "test source memory",
			Importance: 5,
			CreatedAt:  float64(time.Now().Unix()),
			Metadata:   map[string]any{},
		}
	}
	return mems
}

// withMockLLM swaps callLLMFunc for the duration of a test.
func withMockLLM(t *testing.T, resp string, err error) {
	t.Helper()
	orig := callLLMFunc
	t.Cleanup(func() { callLLMFunc = orig })
	callLLMFunc = func(_ context.Context, _ string) (string, error) { return resp, err }
}

// v1Config returns a config whose trigger gates pass on a first run.
func v1Config() Config {
	return Config{Threshold: 10, MaxInputSize: 20, MinIntervalH: 2.0, Mode: "v1"}
}

// ── V1 batch Run() dedup tests ───────────────────────────────────────────────

func TestV1Run_DedupSkip(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	t.Setenv("HOME", t.TempDir())

	store := &dedupMockStore{
		memories: newDedupSources(3),
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return []memory.ScoredMemory{{Score: 0.85}}, nil
		},
	}
	eng := NewEngine(store, &mockEmbedder{dim: 8}, v1Config())

	result, err := eng.Run(context.Background())
	if err != nil {
		t.Fatalf("Run returned hard error: %v", err)
	}
	if store.inserts != 0 {
		t.Errorf("expected 0 inserts on dedup skip, got %d", store.inserts)
	}
	if result.InsightsDedupSkipped != 1 {
		t.Errorf("expected InsightsDedupSkipped=1, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 0 {
		t.Errorf("expected InsightsCreated=0 on dedup skip, got %d", result.InsightsCreated)
	}
}

func TestV1Run_DedupProceedNoMatch(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	t.Setenv("HOME", t.TempDir())

	store := &dedupMockStore{
		memories: newDedupSources(3),
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return nil, nil // no similar insight → proceed with insert
		},
	}
	eng := NewEngine(store, &mockEmbedder{dim: 8}, v1Config())

	result, err := eng.Run(context.Background())
	if err != nil {
		t.Fatalf("Run returned hard error: %v", err)
	}
	if store.searchCalls == 0 {
		t.Error("expected dedup Search to be called")
	}
	if store.inserts != 1 {
		t.Errorf("expected 1 insert on no-match, got %d", store.inserts)
	}
	if result.InsightsDedupSkipped != 0 {
		t.Errorf("expected InsightsDedupSkipped=0, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 1 {
		t.Errorf("expected InsightsCreated=1, got %d", result.InsightsCreated)
	}
}

func TestV1Run_DedupFailOpen(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	t.Setenv("HOME", t.TempDir())

	store := &dedupMockStore{
		memories: newDedupSources(3),
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return nil, errors.New("search backend down")
		},
	}
	eng := NewEngine(store, &mockEmbedder{dim: 8}, v1Config())

	result, err := eng.Run(context.Background())
	if err != nil {
		t.Fatalf("Run returned hard error: %v", err)
	}
	if store.inserts != 1 {
		t.Errorf("expected fail-open insert on search error, got %d inserts", store.inserts)
	}
	if result.InsightsDedupSkipped != 0 {
		t.Errorf("expected InsightsDedupSkipped=0 on fail-open, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 1 {
		t.Errorf("expected InsightsCreated=1 on fail-open, got %d", result.InsightsCreated)
	}
}

// ── RunSingleEvent() dedup tests ─────────────────────────────────────────────

func runSingleEvent(t *testing.T, store *dedupMockStore) *RunResult {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	eng := NewEngine(store, &mockEmbedder{dim: 8}, DefaultConfig())
	result, err := eng.RunSingleEvent(context.Background(), SingleEventInput{
		Cause:       TriggerTaskFailure,
		Summary:     "a task failed and warrants an immediate insight",
		EvidenceIDs: []string{"e1", "e2"},
	})
	if err != nil {
		t.Fatalf("RunSingleEvent returned hard error: %v", err)
	}
	return result
}

func TestRunSingleEvent_DedupSkip(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	store := &dedupMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return []memory.ScoredMemory{{Score: 0.85}}, nil
		},
	}
	result := runSingleEvent(t, store)
	if store.inserts != 0 {
		t.Errorf("expected 0 inserts on dedup skip, got %d", store.inserts)
	}
	if result.InsightsDedupSkipped != 1 {
		t.Errorf("expected InsightsDedupSkipped=1, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 0 {
		t.Errorf("expected InsightsCreated=0 on dedup skip, got %d", result.InsightsCreated)
	}
}

func TestRunSingleEvent_DedupProceedNoMatch(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	store := &dedupMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return nil, nil
		},
	}
	result := runSingleEvent(t, store)
	if store.searchCalls == 0 {
		t.Error("expected dedup Search to be called")
	}
	if store.inserts != 1 {
		t.Errorf("expected 1 insert on no-match, got %d", store.inserts)
	}
	if result.InsightsDedupSkipped != 0 {
		t.Errorf("expected InsightsDedupSkipped=0, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 1 {
		t.Errorf("expected InsightsCreated=1, got %d", result.InsightsCreated)
	}
}

func TestRunSingleEvent_DedupFailOpen(t *testing.T) {
	withMockLLM(t, oneInsightLLM, nil)
	store := &dedupMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return nil, errors.New("search backend down")
		},
	}
	result := runSingleEvent(t, store)
	if store.inserts != 1 {
		t.Errorf("expected fail-open insert on search error, got %d inserts", store.inserts)
	}
	if result.InsightsDedupSkipped != 0 {
		t.Errorf("expected InsightsDedupSkipped=0 on fail-open, got %d", result.InsightsDedupSkipped)
	}
	if result.InsightsCreated != 1 {
		t.Errorf("expected InsightsCreated=1 on fail-open, got %d", result.InsightsCreated)
	}
}
