package memory

import (
	"testing"
	"time"
)

func TestImportanceBand(t *testing.T) {
	tests := []struct {
		importance float64
		want       int
	}{
		{1, 0}, {3, 0}, {4.9, 0}, // low
		{5, 1}, {6, 1}, {7.9, 1}, // mid
		{8, 2}, {9, 2}, {10, 2}, // high
	}
	for _, tt := range tests {
		got := importanceBand(tt.importance)
		if got != tt.want {
			t.Errorf("importanceBand(%v) = %d, want %d", tt.importance, got, tt.want)
		}
	}
}

func TestComputeValidUntil_ExplicitOverride(t *testing.T) {
	cfg := DefaultTTLConfig()
	explicit := float64(time.Now().Add(24 * time.Hour).Unix())
	got := ComputeValidUntil(cfg, TypeEvent, 3, nil, explicit)
	if got != explicit {
		t.Errorf("expected explicit value %v, got %v", explicit, got)
	}
}

func TestComputeValidUntil_IdentityAlwaysPermanent(t *testing.T) {
	cfg := DefaultTTLConfig()
	for _, imp := range []float64{1, 5, 10} {
		got := ComputeValidUntil(cfg, TypeIdentity, imp, nil, 0)
		if got != 0 {
			t.Errorf("identity with importance=%v should be permanent (0), got %v", imp, got)
		}
	}
}

func TestComputeValidUntil_EventTTL(t *testing.T) {
	cfg := DefaultTTLConfig()
	now := time.Now()

	// A-MAC MVP: all event bands → uniform 90 days
	for _, imp := range []float64{3, 6, 9} {
		got := ComputeValidUntil(cfg, TypeEvent, imp, nil, 0)
		expected := float64(now.Add(90 * 24 * time.Hour).Unix())
		if abs(got-expected) > 2 {
			t.Errorf("event importance=%v: got %v, want ~%v (90d)", imp, got, expected)
		}
	}
}

// TestEventUniformTTL verifies the A-MAC MVP uniform 90-day event TTL and its
// overrides, and that other types are unaffected (spec §5.2).
func TestEventUniformTTL(t *testing.T) {
	cfg := DefaultTTLConfig()
	now := time.Now()
	ninetyDays := float64(now.Add(90 * 24 * time.Hour).Unix())

	// Each importance band → 90d.
	for _, imp := range []float64{2, 6, 9} {
		got := ComputeValidUntil(cfg, TypeEvent, imp, nil, 0)
		if abs(got-ninetyDays) > 2 {
			t.Errorf("event importance=%v: got %v, want ~%v (90d)", imp, got, ninetyDays)
		}
	}

	// Explicit valid_until overrides 90d.
	explicit := float64(now.Add(24 * time.Hour).Unix())
	if got := ComputeValidUntil(cfg, TypeEvent, 2, nil, explicit); got != explicit {
		t.Errorf("explicit override: got %v, want %v", got, explicit)
	}

	// permanent tag → 0 (no expiry).
	if got := ComputeValidUntil(cfg, TypeEvent, 2, []string{"permanent"}, 0); got != 0 {
		t.Errorf("permanent-tagged event: got %v, want 0", got)
	}

	// Other types unaffected: identity permanent, directive mid permanent,
	// insight low → 30d.
	if got := ComputeValidUntil(cfg, TypeIdentity, 3, nil, 0); got != 0 {
		t.Errorf("identity should stay permanent, got %v", got)
	}
	if got := ComputeValidUntil(cfg, TypeDirective, 6, nil, 0); got != 0 {
		t.Errorf("mid directive should stay permanent, got %v", got)
	}
	insight30d := float64(now.Add(30 * 24 * time.Hour).Unix())
	if got := ComputeValidUntil(cfg, TypeInsight, 3, nil, 0); abs(got-insight30d) > 2 {
		t.Errorf("low insight should stay 30d: got %v, want ~%v", got, insight30d)
	}
}

func TestComputeValidUntil_InsightTTL(t *testing.T) {
	cfg := DefaultTTLConfig()
	now := time.Now()

	// Low → 30d
	got := ComputeValidUntil(cfg, TypeInsight, 3, nil, 0)
	expected := float64(now.Add(30 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("low-importance insight: got %v, want ~%v", got, expected)
	}

	// Mid → 90d
	got = ComputeValidUntil(cfg, TypeInsight, 6, nil, 0)
	expected = float64(now.Add(90 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("mid-importance insight: got %v, want ~%v", got, expected)
	}

	// High → permanent
	got = ComputeValidUntil(cfg, TypeInsight, 9, nil, 0)
	if got != 0 {
		t.Errorf("high-importance insight should be permanent, got %v", got)
	}
}

func TestComputeValidUntil_DirectiveTTL(t *testing.T) {
	cfg := DefaultTTLConfig()
	now := time.Now()

	// Low → 90d
	got := ComputeValidUntil(cfg, TypeDirective, 3, nil, 0)
	expected := float64(now.Add(90 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("low-importance directive: got %v, want ~%v", got, expected)
	}

	// Mid → permanent
	got = ComputeValidUntil(cfg, TypeDirective, 6, nil, 0)
	if got != 0 {
		t.Errorf("mid-importance directive should be permanent, got %v", got)
	}

	// High → permanent
	got = ComputeValidUntil(cfg, TypeDirective, 9, nil, 0)
	if got != 0 {
		t.Errorf("high-importance directive should be permanent, got %v", got)
	}
}

func TestComputeValidUntil_PermanentTag(t *testing.T) {
	cfg := DefaultTTLConfig()
	// Even a low-importance event gets permanent with "permanent" tag
	got := ComputeValidUntil(cfg, TypeEvent, 2, []string{"permanent"}, 0)
	if got != 0 {
		t.Errorf("permanent-tagged memory should be permanent, got %v", got)
	}
}

func TestComputeValidUntil_TimeSensitiveTag(t *testing.T) {
	cfg := DefaultTTLConfig()
	now := time.Now()

	// Mid-importance insight normally → 90d, but with time-sensitive → capped at 7d
	got := ComputeValidUntil(cfg, TypeInsight, 6, []string{"time-sensitive"}, 0)
	expected := float64(now.Add(7 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("time-sensitive insight: got %v, want ~%v (7d)", got, expected)
	}

	// Event now has uniform 90d TTL; location tag caps it to 7d.
	got = ComputeValidUntil(cfg, TypeEvent, 2, []string{"location"}, 0)
	expected = float64(now.Add(7 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("location-tagged event capped to 7d: got %v, want ~%v", got, expected)
	}
}

func TestComputeValidUntil_PermanentOverridesTimeSensitive(t *testing.T) {
	cfg := DefaultTTLConfig()
	// If both permanent and time-sensitive, permanent wins
	got := ComputeValidUntil(cfg, TypeEvent, 2, []string{"permanent", "time-sensitive"}, 0)
	if got != 0 {
		t.Errorf("permanent should override time-sensitive, got %v", got)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
