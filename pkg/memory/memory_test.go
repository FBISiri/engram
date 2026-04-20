package memory

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

func TestDefaultScoringWeights(t *testing.T) {
	w := DefaultScoringWeights()
	if w.Relevance != 1.0 {
		t.Errorf("Relevance = %f, want 1.0", w.Relevance)
	}
	if w.Recency != 0.5 {
		t.Errorf("Recency = %f, want 0.5", w.Recency)
	}
	if w.Importance != 0.3 {
		t.Errorf("Importance = %f, want 0.3", w.Importance)
	}
}

func TestDefaultDecayConfig(t *testing.T) {
	dc := DefaultDecayConfig()
	if dc.Identity != 1.0 {
		t.Errorf("Identity decay = %f, want 1.0", dc.Identity)
	}
	if dc.Event != 0.99 {
		t.Errorf("Event decay = %f, want 0.99", dc.Event)
	}
	if dc.Insight != 0.9998 {
		t.Errorf("Insight decay = %f, want 0.9998", dc.Insight)
	}
	if dc.Directive != 1.0 {
		t.Errorf("Directive decay = %f, want 1.0", dc.Directive)
	}
}

func TestDecayConfig_DecayFactor(t *testing.T) {
	dc := DefaultDecayConfig()
	tests := []struct {
		typ  MemoryType
		want float64
	}{
		{TypeIdentity, 1.0},
		{TypeEvent, 0.99},
		{TypeInsight, 0.9998},
		{TypeDirective, 1.0},
		{MemoryType("unknown"), 0.99}, // default fallback
	}
	for _, tt := range tests {
		got := dc.DecayFactor(tt.typ)
		if got != tt.want {
			t.Errorf("DecayFactor(%s) = %f, want %f", tt.typ, got, tt.want)
		}
	}
}

func TestScore_PureRelevance(t *testing.T) {
	// Identity type has no decay, so recency=1.0 always.
	m := New("test", WithType(TypeIdentity), WithImportance(5))
	m.CreatedAt = float64(time.Now().Unix()) // just created

	weights := ScoringWeights{Relevance: 1.0, Recency: 0.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, 0.85, weights, decay)
	if math.Abs(score-0.85) > 1e-6 {
		t.Errorf("score = %f, want 0.85", score)
	}
}

func TestScore_PureImportance(t *testing.T) {
	m := New("test", WithType(TypeIdentity), WithImportance(8))
	m.CreatedAt = float64(time.Now().Unix())

	weights := ScoringWeights{Relevance: 0.0, Recency: 0.0, Importance: 1.0}
	decay := DefaultDecayConfig()

	score := Score(m, 0.5, weights, decay)
	expected := 8.0 / 10.0
	if math.Abs(score-expected) > 1e-6 {
		t.Errorf("score = %f, want %f", score, expected)
	}
}

func TestScore_RecencyDecay_Identity(t *testing.T) {
	// Identity type: no decay, recency always 1.0.
	m := New("test", WithType(TypeIdentity), WithImportance(5))
	m.CreatedAt = float64(time.Now().Add(-24 * time.Hour).Unix()) // 24h ago

	weights := ScoringWeights{Relevance: 0.0, Recency: 1.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, 0.0, weights, decay)
	// decay=1.0, so 1.0^24 = 1.0
	if math.Abs(score-1.0) > 1e-6 {
		t.Errorf("score = %f, want 1.0 (identity never decays)", score)
	}
}

func TestScore_RecencyDecay_Event(t *testing.T) {
	// Event type: decay=0.99, after 69h should be ~0.5.
	m := New("test", WithType(TypeEvent), WithImportance(5))
	m.CreatedAt = float64(time.Now().Add(-69 * time.Hour).Unix())

	weights := ScoringWeights{Relevance: 0.0, Recency: 1.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, 0.0, weights, decay)
	// 0.99^69 ≈ 0.5005
	if score < 0.45 || score > 0.55 {
		t.Errorf("score = %f, want ~0.5 for event after 69h", score)
	}
}

func TestScore_RecencyDecay_Insight(t *testing.T) {
	// Insight type: decay=0.9998, very slow. After 24h, should still be high.
	m := New("test", WithType(TypeInsight), WithImportance(5))
	m.CreatedAt = float64(time.Now().Add(-24 * time.Hour).Unix())

	weights := ScoringWeights{Relevance: 0.0, Recency: 1.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, 0.0, weights, decay)
	// 0.9998^24 ≈ 0.9952
	if score < 0.99 {
		t.Errorf("score = %f, want > 0.99 for insight after 24h", score)
	}
}

func TestScore_Combined(t *testing.T) {
	// Full combined scoring with default weights.
	m := New("test", WithType(TypeIdentity), WithImportance(10))
	m.CreatedAt = float64(time.Now().Unix())

	weights := DefaultScoringWeights()
	decay := DefaultDecayConfig()

	score := Score(m, 1.0, weights, decay)
	// 1.0 * 1.0 + 0.5 * 1.0 + 0.3 * 1.0 = 1.8
	if math.Abs(score-1.8) > 0.01 {
		t.Errorf("score = %f, want ~1.8", score)
	}
}

func TestScore_ClampsNegativeSimilarity(t *testing.T) {
	m := New("test", WithType(TypeIdentity), WithImportance(5))
	m.CreatedAt = float64(time.Now().Unix())

	weights := ScoringWeights{Relevance: 1.0, Recency: 0.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, -0.5, weights, decay)
	if score != 0.0 {
		t.Errorf("score = %f, want 0.0 (clamped negative)", score)
	}
}

func TestScore_ClampsHighSimilarity(t *testing.T) {
	m := New("test", WithType(TypeIdentity), WithImportance(5))
	m.CreatedAt = float64(time.Now().Unix())

	weights := ScoringWeights{Relevance: 1.0, Recency: 0.0, Importance: 0.0}
	decay := DefaultDecayConfig()

	score := Score(m, 1.5, weights, decay)
	if math.Abs(score-1.0) > 1e-6 {
		t.Errorf("score = %f, want 1.0 (clamped high)", score)
	}
}

func TestClamp(t *testing.T) {
	tests := []struct {
		v, lo, hi, want float64
	}{
		{0.5, 0, 1, 0.5},
		{-0.1, 0, 1, 0},
		{1.5, 0, 1, 1},
		{0, 0, 1, 0},
		{1, 0, 1, 1},
	}
	for _, tt := range tests {
		got := clamp(tt.v, tt.lo, tt.hi)
		if got != tt.want {
			t.Errorf("clamp(%f, %f, %f) = %f, want %f", tt.v, tt.lo, tt.hi, got, tt.want)
		}
	}
}

// ── Dedup Tests ──

func TestIsDuplicate_Empty(t *testing.T) {
	result := IsDuplicate(nil, 0.92)
	if result != nil {
		t.Errorf("expected nil for empty candidates")
	}
}

func TestIsDuplicate_AboveThreshold(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "dup"}, Score: 0.95},
	}
	result := IsDuplicate(candidates, 0.92)
	if result == nil {
		t.Fatal("expected duplicate")
	}
	if result.Content != "dup" {
		t.Errorf("Content = %q, want 'dup'", result.Content)
	}
}

func TestIsDuplicate_BelowThreshold(t *testing.T) {
	candidates := []ScoredMemory{
		{Memory: Memory{Content: "unique"}, Score: 0.80},
	}
	result := IsDuplicate(candidates, 0.92)
	if result != nil {
		t.Errorf("expected nil for below-threshold score")
	}
}

func TestDefaultDedupThreshold(t *testing.T) {
	if DefaultDedupThreshold != 0.92 {
		t.Errorf("DefaultDedupThreshold = %f, want 0.92", DefaultDedupThreshold)
	}
}

// ── Memory Validation Tests ──

func TestMemory_Validate_Valid(t *testing.T) {
	m := New("test content", WithType(TypeEvent), WithImportance(5))
	if err := m.Validate(); err != nil {
		t.Errorf("unexpected validation error: %v", err)
	}
}

func TestMemory_Validate_EmptyContent(t *testing.T) {
	m := New("")
	if err := m.Validate(); err == nil {
		t.Error("expected error for empty content")
	}
}

func TestMemory_Validate_InvalidType(t *testing.T) {
	m := New("content")
	m.Type = MemoryType("invalid")
	if err := m.Validate(); err == nil {
		t.Error("expected error for invalid type")
	}
}

func TestMemory_Validate_ImportanceRange(t *testing.T) {
	m := New("content")
	m.Importance = 0
	if err := m.Validate(); err == nil {
		t.Error("expected error for importance < 1")
	}
	m.Importance = 11
	if err := m.Validate(); err == nil {
		t.Error("expected error for importance > 10")
	}
}

func TestNew_Defaults(t *testing.T) {
	m := New("hello")
	if m.Content != "hello" {
		t.Errorf("Content = %q, want 'hello'", m.Content)
	}
	if m.Type != TypeEvent {
		t.Errorf("Type = %s, want event", m.Type)
	}
	if m.Source != "agent" {
		t.Errorf("Source = %q, want 'agent'", m.Source)
	}
	if m.Importance != 5.0 {
		t.Errorf("Importance = %f, want 5.0", m.Importance)
	}
	if m.ID == "" {
		t.Error("ID should be generated")
	}
	if m.CreatedAt == 0 {
		t.Error("CreatedAt should be set")
	}
}

func TestNew_WithOptions(t *testing.T) {
	m := New("content",
		WithType(TypeIdentity),
		WithSource("user"),
		WithImportance(9),
		WithTags("work", "project"),
		WithMetadata(map[string]any{"key": "value"}),
	)
	if m.Type != TypeIdentity {
		t.Errorf("Type = %s, want identity", m.Type)
	}
	if m.Source != "user" {
		t.Errorf("Source = %q, want 'user'", m.Source)
	}
	if m.Importance != 9 {
		t.Errorf("Importance = %f, want 9", m.Importance)
	}
	if len(m.Tags) != 2 || m.Tags[0] != "work" || m.Tags[1] != "project" {
		t.Errorf("Tags = %v, want [work, project]", m.Tags)
	}
	if m.Metadata["key"] != "value" {
		t.Errorf("Metadata[key] = %v, want 'value'", m.Metadata["key"])
	}
}

func TestValidTypes(t *testing.T) {
	expected := []MemoryType{TypeIdentity, TypeEvent, TypeInsight, TypeDirective}
	for _, typ := range expected {
		if !ValidTypes[typ] {
			t.Errorf("ValidTypes[%s] = false, want true", typ)
		}
	}
	if ValidTypes[MemoryType("bogus")] {
		t.Error("ValidTypes['bogus'] = true, want false")
	}
}

// ── ReflectedAt Roundtrip Tests (W17 T1) ──

// TestMemoryReflectedAtRoundtrip verifies that the ReflectedAt field survives
// JSON marshal/unmarshal cycles and respects the `omitempty` tag when unset.
// This is the W17 T1 Part 1 test: Memory struct gains top-level reflected_at.
func TestMemoryReflectedAtRoundtrip(t *testing.T) {
	t.Run("SetValue_SurvivesRoundtrip", func(t *testing.T) {
		now := float64(time.Now().Unix())
		m := New("test content", WithType(TypeEvent), WithImportance(5))
		m.ReflectedAt = now

		data, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}

		// JSON should contain the reflected_at field.
		if !strings.Contains(string(data), `"reflected_at"`) {
			t.Errorf("marshaled JSON should contain reflected_at, got: %s", data)
		}

		var got Memory
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if got.ReflectedAt != now {
			t.Errorf("ReflectedAt roundtrip: got %f, want %f", got.ReflectedAt, now)
		}
	})

	t.Run("Zero_OmittedFromJSON", func(t *testing.T) {
		m := New("test content", WithType(TypeEvent), WithImportance(5))
		// ReflectedAt defaults to 0.

		data, err := json.Marshal(m)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}

		// omitempty means zero value must not appear in JSON.
		if strings.Contains(string(data), `"reflected_at"`) {
			t.Errorf("marshaled JSON should NOT contain reflected_at when zero, got: %s", data)
		}

		var got Memory
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if got.ReflectedAt != 0 {
			t.Errorf("ReflectedAt should be 0 after roundtrip of unset field, got %f", got.ReflectedAt)
		}
	})

	t.Run("UnmarshalExplicitValue", func(t *testing.T) {
		// Simulate reading a Memory from an external source (e.g., Qdrant payload
		// decoded into JSON) where reflected_at is explicitly set.
		raw := `{
			"id": "abc-123",
			"type": "insight",
			"content": "reflected memory",
			"source": "agent",
			"importance": 7,
			"tags": ["a", "b"],
			"created_at": 1700000000,
			"updated_at": 1700000100,
			"access_count": 3,
			"reflected_at": 1700000200
		}`
		var m Memory
		if err := json.Unmarshal([]byte(raw), &m); err != nil {
			t.Fatalf("json.Unmarshal: %v", err)
		}
		if m.ReflectedAt != 1700000200 {
			t.Errorf("ReflectedAt = %f, want 1700000200", m.ReflectedAt)
		}
		if m.Type != TypeInsight {
			t.Errorf("Type = %s, want insight", m.Type)
		}
	})
}
