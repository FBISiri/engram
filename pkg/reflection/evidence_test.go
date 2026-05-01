package reflection

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── mock store for evidence tests ──────────────────────────────────────────

type evidenceMockStore struct {
	mu       sync.Mutex
	searchFn func(ctx context.Context, vec []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error)
}

func (s *evidenceMockStore) Search(ctx context.Context, vec []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	s.mu.Lock()
	fn := s.searchFn
	s.mu.Unlock()
	return fn(ctx, vec, opts)
}

func (s *evidenceMockStore) Insert(context.Context, *memory.Memory, []float32) error { return nil }
func (s *evidenceMockStore) Scroll(context.Context, memory.ScrollOptions) ([]memory.Memory, string, error) {
	return nil, "", nil
}
func (s *evidenceMockStore) Delete(context.Context, []string) (int, error)             { return 0, nil }
func (s *evidenceMockStore) Update(context.Context, string, map[string]any) error       { return nil }
func (s *evidenceMockStore) SearchByIDs(context.Context, []string) ([]memory.Memory, error) {
	return nil, nil
}
func (s *evidenceMockStore) EnsureCollection(context.Context) error                    { return nil }
func (s *evidenceMockStore) Stats(context.Context) (*memory.CollectionStats, error)    { return nil, nil }
func (s *evidenceMockStore) DeleteExpired(context.Context) (int, error)                { return 0, nil }

// ── mock embedder ──────────────────────────────────────────────────────────

type mockEmbedder struct {
	dim int
}

func (e *mockEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return make([]float32, e.dim), nil
}
func (e *mockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, e.dim)
	}
	return out, nil
}
func (e *mockEmbedder) Dimension() int { return e.dim }

// ── helpers ────────────────────────────────────────────────────────────────

func makeScoredMemory(id string, score, confidence, createdAt float64, tags []string) memory.ScoredMemory {
	m := memory.Memory{
		ID:         id,
		Content:    "content-" + id,
		Type:       memory.TypeEvent,
		Importance: 5,
		Confidence: confidence,
		CreatedAt:  createdAt,
		Tags:       tags,
	}
	return memory.ScoredMemory{Memory: m, Score: score}
}

// ── Test 1: Normal path — single question, top_k truncation ────────────────

func TestRetrieveEvidence_NormalPath(t *testing.T) {
	now := float64(time.Now().Unix())
	var mems []memory.ScoredMemory
	for i := 0; i < 15; i++ {
		mems = append(mems, makeScoredMemory(
			fmt.Sprintf("m%02d", i),
			1.0-float64(i)*0.01, // descending score
			0.8,                 // above threshold
			now-3600,            // 1h ago
			nil,
		))
	}

	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			limit := opts.Limit
			if limit > len(mems) {
				limit = len(mems)
			}
			return mems[:limit], nil
		},
	}

	cfg := Config{EvidencePerFocal: 10}
	result, _, err := retrieveEvidence(context.Background(), "test question", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 10 {
		t.Fatalf("expected 10 results, got %d", len(result))
	}
	// Verify score ordering preserved (first ID should be m00).
	if result[0].ID != "m00" {
		t.Errorf("expected first result m00, got %s", result[0].ID)
	}
}

// ── Test 2: Empty result — store returns nothing ───────────────────────────

func TestRetrieveEvidence_EmptyResult(t *testing.T) {
	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return nil, nil
		},
	}

	cfg := Config{EvidencePerFocal: 10}
	result, _, err := retrieveEvidence(context.Background(), "test", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 0 {
		t.Fatalf("expected 0 results, got %d", len(result))
	}
}

// ── Test 3: Concurrent + 1 question timeout ────────────────────────────────

func TestRetrieveAllEvidence_ConcurrentTimeout(t *testing.T) {
	now := float64(time.Now().Unix())
	normalMems := []memory.ScoredMemory{
		makeScoredMemory("ok1", 0.9, 0.8, now-3600, nil),
		makeScoredMemory("ok2", 0.8, 0.7, now-3600, nil),
	}

	store := &evidenceMockStore{
		searchFn: func(ctx context.Context, vec []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			// Use vector length as a hack: mockEmbedder returns zero vectors,
			// so we rely on ctx to detect the slow question (set via timeout).
			// Instead, we'll tag slow question via a shared map keyed by goroutine.
			// Simpler: check if this is the "slow" call by sleeping and checking ctx.
			// We need a way to identify which question this is for.
			// The mock embedder always returns the same vector, so we can't distinguish.
			// Use a counter but accept that ordering is non-deterministic — instead
			// make the slow path sleep-based so any 1 timeout is enough.
			select {
			case <-time.After(5 * time.Second):
				return normalMems, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	// Override: only the second question should be slow.
	// We need per-question control. Use a different approach: embed different
	// vectors and match on them. But mockEmbedder returns zeros.
	// Simplest fix: use a channel to make exactly one goroutine slow.
	slowCh := make(chan struct{}, 1)
	slowCh <- struct{}{} // exactly one goroutine will get this

	store.searchFn = func(ctx context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
		select {
		case <-slowCh:
			// This goroutine is the slow one.
			select {
			case <-time.After(5 * time.Second):
				return normalMems, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		default:
			// Fast path.
			return normalMems, nil
		}
	}

	cfg := Config{
		EvidencePerFocal:      10,
		EvidenceSearchTimeout: 200 * time.Millisecond, // short for test
	}
	result := &RunResult{}
	questions := []string{"q1", "q2-slow", "q3"}

	perQ := retrieveAllEvidence(context.Background(), questions, store, &mockEmbedder{dim: 8}, cfg, result)

	// Exactly 1 question should have timed out, 2 should have evidence.
	emptyCount := 0
	fullCount := 0
	for i, pq := range perQ {
		switch len(pq.Evidence) {
		case 0:
			emptyCount++
		case 2:
			fullCount++
		default:
			t.Errorf("q%d: unexpected evidence count %d", i+1, len(pq.Evidence))
		}
	}
	if emptyCount != 1 {
		t.Errorf("expected exactly 1 timed-out question, got %d", emptyCount)
	}
	if fullCount != 2 {
		t.Errorf("expected exactly 2 successful questions, got %d", fullCount)
	}

	if len(result.Errors) == 0 {
		t.Error("expected timeout error in result.Errors")
	}
	if result.DroppedNoEvidence != 1 {
		t.Errorf("expected DroppedNoEvidence=1, got %d", result.DroppedNoEvidence)
	}
}

// ── Test 4: Confidence filtering ───────────────────────────────────────────

func TestRetrieveEvidence_ConfidenceFilter(t *testing.T) {
	now := float64(time.Now().Unix())
	var mems []memory.ScoredMemory
	for i := 0; i < 20; i++ {
		conf := 0.8
		if i >= 10 {
			conf = 0.5 // below threshold
		}
		mems = append(mems, makeScoredMemory(
			fmt.Sprintf("c%02d", i), 0.9, conf, now-3600, nil,
		))
	}

	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			// Return all memories; confidence filtering is now post-search.
			if opts.Limit > 0 && len(mems) > opts.Limit {
				return mems[:opts.Limit], nil
			}
			return mems, nil
		},
	}

	cfg := Config{EvidencePerFocal: 20}
	result, _, err := retrieveEvidence(context.Background(), "test", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 10 {
		t.Fatalf("expected 10 (confidence>=0.6), got %d", len(result))
	}
	for _, m := range result {
		if m.Confidence < 0.6 {
			t.Errorf("memory %s has confidence %.2f < 0.6", m.ID, m.Confidence)
		}
	}
}

// ── Test 4b: Confidence=0 (absent) treated as 1.0 ─────────────────────────

func TestRetrieveEvidence_ZeroConfidenceTreatedAsOne(t *testing.T) {
	now := float64(time.Now().Unix())
	mems := []memory.ScoredMemory{
		makeScoredMemory("no-conf", 0.9, 0, now-3600, nil),
		makeScoredMemory("high-conf", 0.8, 0.8, now-3600, nil),
		makeScoredMemory("low-conf", 0.7, 0.3, now-3600, nil),
	}

	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return mems, nil
		},
	}

	cfg := Config{EvidencePerFocal: 10}
	result, _, err := retrieveEvidence(context.Background(), "test", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 (conf=0 treated as 1.0, conf=0.3 excluded), got %d", len(result))
	}
	if result[0].ID != "no-conf" {
		t.Errorf("expected no-conf first, got %s", result[0].ID)
	}
}

// ── Test 5: Reflection-origin age filtering ────────────────────────────────

func TestRetrieveEvidence_ReflectionOriginAgeFilter(t *testing.T) {
	now := float64(time.Now().Unix())
	mems := []memory.ScoredMemory{
		// 2 normal memories (old, no reflection tag).
		makeScoredMemory("old1", 0.9, 0.8, now-30*86400, nil),
		makeScoredMemory("old2", 0.8, 0.8, now-20*86400, nil),
		// 3 reflection-origin memories created within 7 days → should be excluded.
		makeScoredMemory("ref-new1", 0.7, 0.8, now-3600, []string{sourceReflectionTag}),
		makeScoredMemory("ref-new2", 0.6, 0.8, now-86400, []string{sourceReflectionTag}),
		makeScoredMemory("ref-new3", 0.5, 0.8, now-5*86400, []string{sourceReflectionTag}),
	}

	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			return mems, nil
		},
	}

	cfg := Config{EvidencePerFocal: 10}
	result, _, err := retrieveEvidence(context.Background(), "test", store, &mockEmbedder{dim: 8}, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 (exclude 3 recent reflection-origin), got %d", len(result))
	}
	for _, m := range result {
		if hasTag(m.Tags, sourceReflectionTag) {
			t.Errorf("memory %s is reflection-origin and should have been filtered", m.ID)
		}
	}
}

// ── Optional: Cross-question dedup stats ───────────────────────────────────

func TestRetrieveAllEvidence_CrossQuestionDedup(t *testing.T) {
	now := float64(time.Now().Unix())

	// Q1 returns [A,B,C], Q2 returns [B,C,D], Q3 returns [C,D,E].
	sets := [][]memory.ScoredMemory{
		{
			makeScoredMemory("A", 0.9, 0.8, now-3600, nil),
			makeScoredMemory("B", 0.8, 0.8, now-3600, nil),
			makeScoredMemory("C", 0.7, 0.8, now-3600, nil),
		},
		{
			makeScoredMemory("B", 0.9, 0.8, now-3600, nil),
			makeScoredMemory("C", 0.8, 0.8, now-3600, nil),
			makeScoredMemory("D", 0.7, 0.8, now-3600, nil),
		},
		{
			makeScoredMemory("C", 0.9, 0.8, now-3600, nil),
			makeScoredMemory("D", 0.8, 0.8, now-3600, nil),
			makeScoredMemory("E", 0.7, 0.8, now-3600, nil),
		},
	}

	callIdx := 0
	var mu sync.Mutex
	store := &evidenceMockStore{
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			mu.Lock()
			i := callIdx
			callIdx++
			mu.Unlock()
			return sets[i], nil
		},
	}

	cfg := Config{EvidencePerFocal: 10}
	result := &RunResult{}
	perQ := retrieveAllEvidence(context.Background(), []string{"q1", "q2", "q3"}, store, &mockEmbedder{dim: 8}, cfg, result)

	// Each question should still see its own 3 evidence items.
	for i, pq := range perQ {
		if len(pq.Evidence) != 3 {
			t.Errorf("q%d: expected 3 evidence, got %d", i+1, len(pq.Evidence))
		}
	}

	// Union = {A,B,C,D,E} = 5.
	if result.EvidenceCount != 5 {
		t.Errorf("expected EvidenceCount=5, got %d", result.EvidenceCount)
	}
	// Overlap = 9 - 5 = 4.
	if result.EvidenceOverlap != 4 {
		t.Errorf("expected EvidenceOverlap=4, got %d", result.EvidenceOverlap)
	}
}

// ── Test 7: Default timeout when EvidenceSearchTimeout=0 ─────────────────

func TestRetrieveAllEvidence_DefaultTimeout(t *testing.T) {
	now := float64(time.Now().Unix())
	normalMems := []memory.ScoredMemory{
		makeScoredMemory("dt1", 0.9, 0.8, now-3600, nil),
	}

	store := &evidenceMockStore{
		searchFn: func(ctx context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			// Sleep longer than 3s default timeout but less than 10s max.
			select {
			case <-time.After(5 * time.Second):
				return normalMems, nil
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		},
	}

	cfg := Config{
		EvidencePerFocal:      10,
		EvidenceSearchTimeout: 0, // should fall back to 3s default
	}
	result := &RunResult{}
	perQ := retrieveAllEvidence(context.Background(), []string{"q1"}, store, &mockEmbedder{dim: 8}, cfg, result)

	if len(perQ[0].Evidence) != 0 {
		t.Error("expected timeout with default 3s, but got evidence")
	}
	if len(result.Errors) == 0 {
		t.Error("expected timeout error in result.Errors")
	}
}
