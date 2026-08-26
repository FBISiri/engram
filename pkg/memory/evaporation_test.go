package memory

import (
	"math"
	"testing"
	"time"
)

func daysAgo(days float64) float64 {
	return float64(time.Now().Unix()) - days*86400.0
}

func enabledEvapConfig() EvaporationConfig {
	c := DefaultEvaporationConfig()
	c.Enabled = true
	return c
}

func TestEffectiveImportance_NoDecay(t *testing.T) {
	// Disabled config → effective == base.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(100)
	if got := EffectiveImportance(m, DefaultEvaporationConfig()); got != 6 {
		t.Errorf("disabled: got %f, want 6", got)
	}

	// Identity (half-life 0) → no decay even when enabled.
	id := New("x", WithType(TypeIdentity), WithImportance(6))
	id.CreatedAt = daysAgo(1000)
	if got := EffectiveImportance(id, enabledEvapConfig()); got != 6 {
		t.Errorf("identity: got %f, want 6", got)
	}
}

func TestEffectiveImportance_HalfLife(t *testing.T) {
	// event, imp=6, age=30d (== half-life), access=0 → ≈ 3.0.
	m := New("x", WithType(TypeEvent), WithImportance(6))
	m.CreatedAt = daysAgo(30)
	got := EffectiveImportance(m, enabledEvapConfig())
	if math.Abs(got-3.0) > 0.05 {
		t.Errorf("half-life: got %f, want ≈3.0", got)
	}
}

func TestEffectiveImportance_AccessBoost(t *testing.T) {
	// access_count=20, α=0.15 → boost ≈ 1.456. Verify via age=0 (decay=1).
	m := New("x", WithType(TypeEvent), WithImportance(1))
	m.CreatedAt = daysAgo(0)
	m.AccessCount = 20
	got := EffectiveImportance(m, enabledEvapConfig())
	wantBoost := 1.0 + 0.15*math.Log(1.0+20.0)
	if math.Abs(got-wantBoost) > 1e-6 {
		t.Errorf("access boost: got %f, want %f", got, wantBoost)
	}
	if math.Abs(wantBoost-1.456) > 0.01 {
		t.Errorf("boost sanity: got %f, want ≈1.456", wantBoost)
	}
}

func TestEffectiveImportance_BelowThreshold(t *testing.T) {
	// event, imp=3, age=60d, access=0 → effective < 1.0.
	m := New("x", WithType(TypeEvent), WithImportance(3))
	m.CreatedAt = daysAgo(60)
	got := EffectiveImportance(m, enabledEvapConfig())
	if got >= 1.0 {
		t.Errorf("below threshold: got %f, want < 1.0", got)
	}
}
