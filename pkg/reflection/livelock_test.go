package reflection

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── R1: Stage-5 gate (shouldMarkSources) ─────────────────────────────────────

func TestShouldMarkSources(t *testing.T) {
	cases := []struct {
		name  string
		stats writeBackStats
		want  bool
	}{
		{"dedup only → mark (livelock fix)", writeBackStats{DedupSkipped: 3}, true},
		{"written → mark", writeBackStats{Written: 1}, true},
		{"drafts → mark", writeBackStats{Drafts: 2}, true},
		{"all failed → do not mark", writeBackStats{Failed: 3}, false},
		{"generic skips only → do not mark", writeBackStats{Skipped: 2}, false},
		{"nothing → do not mark", writeBackStats{}, false},
		{"mixed with dedup → mark", writeBackStats{Written: 1, DedupSkipped: 1, Skipped: 1}, true},
	}
	for _, c := range cases {
		if got := shouldMarkSources(c.stats); got != c.want {
			t.Errorf("%s: shouldMarkSources(%+v)=%v, want %v", c.name, c.stats, got, c.want)
		}
	}
}

// ── R4c: counter split — mixed batch (1 written, 1 dedup, 1 non-dedup skip) ───

func TestWriteDialecticInsights_CounterSplit(t *testing.T) {
	searchCalls := 0
	insertCount := 0
	store := &writeBackMockStore{
		insertFn: func(_ context.Context, _ *memory.Memory, _ []float32) error {
			insertCount++
			return nil
		},
		searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
			searchCalls++
			// First high-conf insight: no match → written.
			// Second high-conf insight: match → dedup skip.
			if searchCalls == 2 {
				return []memory.ScoredMemory{{Score: 0.85}}, nil
			}
			return nil, nil
		},
	}

	e := writeBackTestEngine(store)
	dialectics := []DialecticInsight{
		{Question: "q1", Content: "written insight", Tensions: []string{}, SourceIDs: []string{"e1", "e2"}, Confidence: 0.85, Importance: 7, Tags: []string{"tag"}},
		{Question: "q2", Content: "dedup insight", Tensions: []string{}, SourceIDs: []string{"e3", "e4"}, Confidence: 0.85, Importance: 7, Tags: []string{"tag"}},
		{Question: "q3", Content: "low conf insight", Tensions: []string{}, SourceIDs: []string{"e5", "e6"}, Confidence: 0.3, Importance: 5, Tags: []string{"tag"}},
	}
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2", Evidence: makeEvidence("e3", "e4")},
		{Question: "q3", Evidence: makeEvidence("e5", "e6")},
	}

	stats := e.writeDialecticInsights(context.Background(), dialectics, evidenceList, e.cfg)

	if stats.Written != 1 {
		t.Errorf("expected Written=1, got %d", stats.Written)
	}
	if stats.DedupSkipped != 1 {
		t.Errorf("expected DedupSkipped=1, got %d", stats.DedupSkipped)
	}
	if stats.Skipped != 1 {
		t.Errorf("expected Skipped=1 (non-dedup low-conf), got %d", stats.Skipped)
	}
	if insertCount != 1 {
		t.Errorf("expected 1 insert, got %d", insertCount)
	}
}

// ── R4a: dedup hit → sources marked (end-to-end via RunV2) ────────────────────

func TestRunV2_DedupHitMarksSources(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	withV2MockLLM(t)

	batch := newDedupSources(3)
	store := &dedupMockStore{
		memories: batch,
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			if isDedupSearch(opts) {
				// Write-back dedup: an existing near-duplicate insight exists.
				return []memory.ScoredMemory{{Score: 0.85}}, nil
			}
			return v2Evidence(), nil
		},
	}
	eng := NewEngine(store, &mockEmbedder{dim: 8}, v2Config())

	result, err := eng.RunV2(context.Background())
	if err != nil {
		t.Fatalf("RunV2 hard error: %v", err)
	}
	if !result.Triggered {
		t.Fatalf("expected run to trigger; skip_reason=%q errors=%v", result.SkipReason, result.Errors)
	}
	if result.InsightsCreated != 0 {
		t.Errorf("expected InsightsCreated=0, got %d", result.InsightsCreated)
	}
	if result.InsightsDedupSkipped == 0 {
		t.Errorf("expected InsightsDedupSkipped>0, got 0 (errors=%v)", result.Errors)
	}
	if result.InsightsSkipped != 0 {
		t.Errorf("expected InsightsSkipped=0, got %d", result.InsightsSkipped)
	}
	if result.SourcesMarked != len(batch) {
		t.Errorf("expected SourcesMarked=%d (livelock fix), got %d", len(batch), result.SourcesMarked)
	}
	if strings.Contains(joinErrs(result.Errors), "sources not marked") {
		t.Errorf("errors must NOT contain 'sources not marked': %v", result.Errors)
	}
}

// ── R4b: genuine failure → sources NOT marked (embed failure) ─────────────────

func TestRunV2_GenuineFailureDoesNotMarkSources(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	withV2MockLLM(t)

	batch := newDedupSources(3)
	store := &dedupMockStore{
		memories: batch,
		searchFn: func(_ context.Context, _ []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
			if isDedupSearch(opts) {
				return nil, nil
			}
			return v2Evidence(), nil
		},
	}
	// Embedder fails on the synthesized insight content (marker "synth-insight"),
	// so evidence retrieval still works but every write-back embed fails → Failed.
	eng := NewEngine(store, &failOnInsightEmbedder{dim: 8}, v2Config())

	result, err := eng.RunV2(context.Background())
	if err != nil {
		t.Fatalf("RunV2 hard error: %v", err)
	}
	if !result.Triggered {
		t.Fatalf("expected run to trigger; skip_reason=%q errors=%v", result.SkipReason, result.Errors)
	}
	if result.SourcesMarked != 0 {
		t.Errorf("expected SourcesMarked=0 on genuine failure, got %d", result.SourcesMarked)
	}
	if !strings.Contains(joinErrs(result.Errors), "sources not marked") {
		t.Errorf("expected errors to contain 'sources not marked', got %v", result.Errors)
	}
}

// ── shared V2 test fixtures ───────────────────────────────────────────────────

func joinErrs(errs []string) string { return strings.Join(errs, " | ") }

// isDedupSearch reports whether opts targets the reflection collection, i.e. it
// is the pre-write dedup Search (vs. evidence retrieval).
func isDedupSearch(opts memory.SearchOptions) bool {
	for _, f := range opts.Filters {
		if f.Field == "collection" {
			return true
		}
	}
	return false
}

// v2Evidence returns two evidence memories with stable IDs the canned dialectic
// LLM response references in its source_ids.
func v2Evidence() []memory.ScoredMemory {
	now := float64(time.Now().Unix())
	return []memory.ScoredMemory{
		makeScoredMemory("ev1", 0.99, 0.8, now, nil),
		makeScoredMemory("ev2", 0.98, 0.8, now, nil),
	}
}

// v2Config returns a config whose trigger gates pass and selects the V2 path.
func v2Config() Config {
	c := DefaultConfig()
	c.Mode = "v2"
	c.Threshold = 10
	c.MaxInputSize = 20
	c.MinIntervalH = 2.0
	c.FocalInputSize = 50
	c.FocalQuestions = 1
	return c
}

// withV2MockLLM drives focal-question generation and dialectic synthesis with
// canned output that both parsers accept.
func withV2MockLLM(t *testing.T) {
	t.Helper()
	orig := callLLMFunc
	t.Cleanup(func() { callLLMFunc = orig })
	callLLMFunc = func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "JSON array") {
			return `["What cross-domain patterns emerge?"]`, nil
		}
		return `{"content":"synth-insight: a dialectic synthesis","tensions":[],"source_ids":["ev1","ev2"],"confidence":0.9,"importance":7,"tags":["pattern"]}`, nil
	}
}

// failOnInsightEmbedder embeds evidence/question text normally but errors on the
// synthesized insight content, forcing every write-back into stats.Failed.
type failOnInsightEmbedder struct{ dim int }

func (m *failOnInsightEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if strings.Contains(text, "synth-insight") {
		return nil, fmt.Errorf("simulated embed failure")
	}
	return make([]float32, m.dim), nil
}

func (m *failOnInsightEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = make([]float32, m.dim)
	}
	return out, nil
}
func (m *failOnInsightEmbedder) Dimension() int { return m.dim }
