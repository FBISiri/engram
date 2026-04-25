package reflection

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

func makeEvidence(ids ...string) []memory.Memory {
	var mems []memory.Memory
	for _, id := range ids {
		mems = append(mems, memory.Memory{
			ID:      id,
			Content: "evidence-" + id,
			Type:    memory.TypeEvent,
		})
	}
	return mems
}

// mockHaikuServer overrides callHaiku for dialectic tests.
// We use a package-level variable + init pattern to inject mock responses.
// Since callHaiku is a package function, we wrap it via the Engine's LLM path.
// For tests, we override at the test level using a custom approach.

// dialecticTestEngine creates an Engine with mock store/embedder for dialectic tests.
func dialecticTestEngine() *Engine {
	return &Engine{
		store:    &evidenceMockStore{searchFn: func(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) { return nil, nil }},
		embedder: &mockEmbedder{dim: 8},
		cfg:      DefaultConfig(),
	}
}

// --- Test 1: Normal path (OkCount=3) ---

func TestGenerateDialecticInsights_NormalPath(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	var mu sync.Mutex
	callCount := 0
	callHaikuFunc = func(_ context.Context, prompt string) (string, error) {
		mu.Lock()
		callCount++
		mu.Unlock()
		resp := dialecticLLMResponse{
			Content:    "Synthesized insight from evidence",
			Tensions:   []string{"tension-a", "tension-b"},
			SourceIDs:  []string{"e1", "e2"},
			Confidence: 0.85,
			Importance: 7,
			Tags:       []string{"pattern", "behavior"},
		}
		b, _ := json.Marshal(resp)
		return string(b), nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2", "e3")},
		{Question: "q2", Evidence: makeEvidence("e1", "e2", "e4")},
		{Question: "q3", Evidence: makeEvidence("e1", "e2", "e5")},
	}

	insights, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.OkCount != 3 {
		t.Errorf("expected OkCount=3, got %d", stats.OkCount)
	}
	if stats.LLMCalls != 3 {
		t.Errorf("expected LLMCalls=3, got %d", stats.LLMCalls)
	}
	if stats.DroppedNoEvidence != 0 {
		t.Errorf("expected DroppedNoEvidence=0, got %d", stats.DroppedNoEvidence)
	}
	if len(insights) != 3 {
		t.Errorf("expected 3 insights, got %d", len(insights))
	}
	// Invariant: OkCount + FailedCount + DroppedNoEvidence + DroppedLowConf == 3
	sum := stats.OkCount + stats.FailedCount + stats.DroppedNoEvidence + stats.DroppedLowConf
	if sum != 3 {
		t.Errorf("invariant broken: %d + %d + %d + %d = %d, expected 3",
			stats.OkCount, stats.FailedCount, stats.DroppedNoEvidence, stats.DroppedLowConf, sum)
	}
	if callCount != 3 {
		t.Errorf("expected 3 LLM calls, got %d", callCount)
	}
}

// --- Test 2: Empty evidence skip (no LLM call) ---

func TestGenerateDialecticInsights_EmptyEvidenceSkip(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	var mu sync.Mutex
	callCount := 0
	callHaikuFunc = func(_ context.Context, _ string) (string, error) {
		mu.Lock()
		callCount++
		mu.Unlock()
		resp := dialecticLLMResponse{
			Content:    "insight",
			Tensions:   []string{},
			SourceIDs:  []string{"e1", "e2"},
			Confidence: 0.8,
			Importance: 5,
			Tags:       []string{"tag"},
		}
		b, _ := json.Marshal(resp)
		return string(b), nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2", Evidence: nil},
		{Question: "q3", Evidence: makeEvidence("e1", "e2")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.DroppedNoEvidence != 1 {
		t.Errorf("expected DroppedNoEvidence=1, got %d", stats.DroppedNoEvidence)
	}
	if stats.OkCount != 2 {
		t.Errorf("expected OkCount=2, got %d", stats.OkCount)
	}
	// LLMCalls == 3 - DroppedNoEvidence
	if stats.LLMCalls != 2 {
		t.Errorf("expected LLMCalls=2, got %d", stats.LLMCalls)
	}
	if callCount != 2 {
		t.Errorf("expected 2 actual LLM calls, got %d", callCount)
	}
	// Invariant
	sum := stats.OkCount + stats.FailedCount + stats.DroppedNoEvidence + stats.DroppedLowConf
	if sum != 3 {
		t.Errorf("invariant broken: sum=%d, expected 3", sum)
	}
}

// --- Test 3: JSON corruption → single-question degrade ---

func TestGenerateDialecticInsights_JSONCorruption(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	var mu sync.Mutex
	callIdx := 0
	callHaikuFunc = func(_ context.Context, _ string) (string, error) {
		mu.Lock()
		idx := callIdx
		callIdx++
		mu.Unlock()
		if idx == 1 {
			return `{broken json!!!`, nil
		}
		resp := dialecticLLMResponse{
			Content:    "valid insight",
			Tensions:   []string{},
			SourceIDs:  []string{"e1", "e2"},
			Confidence: 0.8,
			Importance: 6,
			Tags:       []string{"tag"},
		}
		b, _ := json.Marshal(resp)
		return string(b), nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2", Evidence: makeEvidence("e1", "e2")},
		{Question: "q3", Evidence: makeEvidence("e1", "e2")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.OkCount != 2 {
		t.Errorf("expected OkCount=2, got %d", stats.OkCount)
	}
	if stats.FailedCount != 1 {
		t.Errorf("expected FailedCount=1, got %d", stats.FailedCount)
	}
	if len(stats.Errors) == 0 {
		t.Error("expected parse error in stats.Errors")
	}
}

// --- Test 4: LLM timeout ---

func TestGenerateDialecticInsights_LLMTimeout(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	callHaikuFunc = func(ctx context.Context, _ string) (string, error) {
		select {
		case <-time.After(5 * time.Second):
			resp := dialecticLLMResponse{
				Content:    "late",
				Tensions:   []string{},
				SourceIDs:  []string{"e1", "e2"},
				Confidence: 0.8,
				Importance: 5,
				Tags:       []string{},
			}
			b, _ := json.Marshal(resp)
			return string(b), nil
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	e := dialecticTestEngine()
	cfg := e.cfg
	cfg.DialecticTimeout = 200 * time.Millisecond

	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2", Evidence: makeEvidence("e1", "e2")},
		{Question: "q3", Evidence: makeEvidence("e1", "e2")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.FailedCount != 3 {
		t.Errorf("expected FailedCount=3, got %d", stats.FailedCount)
	}
	if stats.OkCount != 0 {
		t.Errorf("expected OkCount=0, got %d", stats.OkCount)
	}
}

// --- Test 5: source_id out of bounds → degrade (prompt injection defense) ---

func TestGenerateDialecticInsights_SourceIDOutOfBounds(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	callHaikuFunc = func(_ context.Context, _ string) (string, error) {
		resp := dialecticLLMResponse{
			Content:    "injected insight",
			Tensions:   []string{},
			SourceIDs:  []string{"e1", "INJECTED_ID"},
			Confidence: 0.9,
			Importance: 8,
			Tags:       []string{},
		}
		b, _ := json.Marshal(resp)
		return string(b), nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2", "e3")},
		{Question: "q2", Evidence: makeEvidence("e1", "e2", "e3")},
		{Question: "q3", Evidence: makeEvidence("e1", "e2", "e3")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.FailedCount != 3 {
		t.Errorf("expected FailedCount=3 (all degraded due to injected ID), got %d", stats.FailedCount)
	}
	if stats.OkCount != 0 {
		t.Errorf("expected OkCount=0, got %d", stats.OkCount)
	}
	for _, e := range stats.Errors {
		if !contains(e, "prompt injection") {
			t.Errorf("expected prompt injection error, got: %s", e)
		}
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && containsStr(s, substr)
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// --- Test 6: Cross-question concurrent failure isolation ---

func TestGenerateDialecticInsights_ConcurrentFailureIsolation(t *testing.T) {
	origCallHaiku := callHaikuFunc
	defer func() { callHaikuFunc = origCallHaiku }()

	var mu sync.Mutex
	callIdx := 0
	callHaikuFunc = func(ctx context.Context, _ string) (string, error) {
		mu.Lock()
		idx := callIdx
		callIdx++
		mu.Unlock()

		switch idx {
		case 0:
			// Q1: timeout
			select {
			case <-time.After(5 * time.Second):
				return "", fmt.Errorf("should not reach")
			case <-ctx.Done():
				return "", ctx.Err()
			}
		case 1:
			// Q2: JSON corruption
			return `not valid json at all`, nil
		case 2:
			// Q3: success
			resp := dialecticLLMResponse{
				Content:    "valid insight for q3",
				Tensions:   []string{"tension"},
				SourceIDs:  []string{"e1", "e2"},
				Confidence: 0.8,
				Importance: 7,
				Tags:       []string{"ok"},
			}
			b, _ := json.Marshal(resp)
			return string(b), nil
		}
		return "", fmt.Errorf("unexpected call")
	}

	e := dialecticTestEngine()
	cfg := e.cfg
	cfg.DialecticTimeout = 200 * time.Millisecond

	evidenceList := []PerQuestionEvidence{
		{Question: "q1-timeout", Evidence: makeEvidence("e1", "e2")},
		{Question: "q2-corrupt", Evidence: makeEvidence("e1", "e2")},
		{Question: "q3-ok", Evidence: makeEvidence("e1", "e2")},
	}

	insights, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.OkCount != 1 {
		t.Errorf("expected OkCount=1, got %d", stats.OkCount)
	}
	if stats.FailedCount != 2 {
		t.Errorf("expected FailedCount=2, got %d", stats.FailedCount)
	}
	if len(insights) != 1 {
		t.Errorf("expected 1 insight, got %d", len(insights))
	}
	// Invariant
	sum := stats.OkCount + stats.FailedCount + stats.DroppedNoEvidence + stats.DroppedLowConf
	if sum != 3 {
		t.Errorf("invariant broken: sum=%d, expected 3", sum)
	}
}
