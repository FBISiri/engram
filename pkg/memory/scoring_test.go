package memory

import (
	"math"
	"testing"
)

func TestCosineSim_Identical(t *testing.T) {
	a := []float32{1.0, 0.0, 0.0}
	sim := cosineSim(a, a)
	if math.Abs(sim-1.0) > 1e-6 {
		t.Errorf("cosineSim(identical) = %f, want 1.0", sim)
	}
}

func TestCosineSim_Orthogonal(t *testing.T) {
	a := []float32{1.0, 0.0}
	b := []float32{0.0, 1.0}
	sim := cosineSim(a, b)
	if math.Abs(sim) > 1e-6 {
		t.Errorf("cosineSim(orthogonal) = %f, want 0.0", sim)
	}
}

func TestCosineSim_Opposite(t *testing.T) {
	a := []float32{1.0, 0.0}
	b := []float32{-1.0, 0.0}
	sim := cosineSim(a, b)
	if math.Abs(sim-(-1.0)) > 1e-6 {
		t.Errorf("cosineSim(opposite) = %f, want -1.0", sim)
	}
}

func TestCosineSim_DifferentLengths(t *testing.T) {
	a := []float32{1.0, 0.0}
	b := []float32{1.0, 0.0, 0.0}
	sim := cosineSim(a, b)
	if sim != 0 {
		t.Errorf("cosineSim(different lengths) = %f, want 0", sim)
	}
}

func TestCosineSim_Empty(t *testing.T) {
	sim := cosineSim(nil, nil)
	if sim != 0 {
		t.Errorf("cosineSim(nil, nil) = %f, want 0", sim)
	}
}

func TestCosineSim_ZeroVector(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	sim := cosineSim(a, b)
	if sim != 0 {
		t.Errorf("cosineSim(zero, non-zero) = %f, want 0", sim)
	}
}

func TestMMR_FewerThanTopK(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "a"}, Score: 0.9},
		{Memory: Memory{Content: "b"}, Score: 0.8},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
	}
	result := MMR(candidates, vectors, 5, 0.5)
	if len(result) != 2 {
		t.Errorf("expected 2 results, got %d", len(result))
	}
}

func TestMMR_SelectsTopFirst(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "best"}, Score: 0.95},
		{Memory: Memory{Content: "good"}, Score: 0.85},
		{Memory: Memory{Content: "ok"}, Score: 0.75},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0.9, 0.1, 0},
		{0, 0, 1},
	}
	result := MMR(candidates, vectors, 2, 0.5)
	if len(result) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result))
	}
	if result[0].Content != "best" {
		t.Errorf("first result = %q, want 'best'", result[0].Content)
	}
}

func TestMMR_DiversityPreference(t *testing.T) {
	// With two similar items and one diverse item, MMR should prefer diversity.
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "A"}, Score: 0.90},
		{Memory: Memory{Content: "A-clone"}, Score: 0.89}, // very similar to A
		{Memory: Memory{Content: "B-diverse"}, Score: 0.88},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0.99, 0.01, 0}, // nearly identical to A
		{0, 1, 0},       // orthogonal to A
	}
	result := MMR(candidates, vectors, 2, 0.5)
	if len(result) != 2 {
		t.Fatalf("expected 2 results, got %d", len(result))
	}
	// First should be A (highest score), second should be B-diverse (diverse)
	if result[0].Content != "A" {
		t.Errorf("first = %q, want 'A'", result[0].Content)
	}
	if result[1].Content != "B-diverse" {
		t.Errorf("second = %q, want 'B-diverse' (diversity win over A-clone)", result[1].Content)
	}
}

func TestMMR_PureRelevance(t *testing.T) {
	// Lambda=1.0 means pure relevance, no diversity penalty.
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "first"}, Score: 0.9},
		{Memory: Memory{Content: "second"}, Score: 0.8},
		{Memory: Memory{Content: "third"}, Score: 0.7},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0.99, 0.01, 0},
		{0.98, 0.02, 0},
	}
	result := MMR(candidates, vectors, 2, 1.0)
	if result[0].Content != "first" {
		t.Errorf("first = %q, want 'first'", result[0].Content)
	}
	if result[1].Content != "second" {
		t.Errorf("second = %q, want 'second'", result[1].Content)
	}
}

func TestMMR_PureDiversity(t *testing.T) {
	// Lambda=0.0 means pure diversity (maximally different from selected).
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "A"}, Score: 0.9},
		{Memory: Memory{Content: "A-clone"}, Score: 0.89},
		{Memory: Memory{Content: "diverse"}, Score: 0.1},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0.99, 0.01, 0},
		{0, 0, 1}, // maximally diverse from A
	}
	result := MMR(candidates, vectors, 2, 0.0)
	// First is always the highest score, second should be the most diverse.
	if result[1].Content != "diverse" {
		t.Errorf("second = %q, want 'diverse' (pure diversity mode)", result[1].Content)
	}
}

func TestMMR_SingleCandidate(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "only"}, Score: 0.5},
	}
	vectors := [][]float32{{1, 0}}
	result := MMR(candidates, vectors, 1, 0.5)
	if len(result) != 1 || result[0].Content != "only" {
		t.Errorf("unexpected result: %v", result)
	}
}

func TestMMR_EqualScores(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "a"}, Score: 0.5},
		{Memory: Memory{Content: "b"}, Score: 0.5},
		{Memory: Memory{Content: "c"}, Score: 0.5},
	}
	vectors := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}
	result := MMR(candidates, vectors, 2, 0.5)
	if len(result) != 2 {
		t.Fatalf("expected 2, got %d", len(result))
	}
}
