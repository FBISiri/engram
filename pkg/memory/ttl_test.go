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
		{1, 0}, {3, 0}, {4.9, 0},  // low
		{5, 1}, {6, 1}, {7.9, 1},  // mid
		{8, 2}, {9, 2}, {10, 2},   // high
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

	// Low importance event → 3 days
	got := ComputeValidUntil(cfg, TypeEvent, 3, nil, 0)
	expected := float64(now.Add(3 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("low-importance event: got %v, want ~%v", got, expected)
	}

	// Mid importance event → 7 days
	got = ComputeValidUntil(cfg, TypeEvent, 6, nil, 0)
	expected = float64(now.Add(7 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("mid-importance event: got %v, want ~%v", got, expected)
	}

	// High importance event → 30 days
	got = ComputeValidUntil(cfg, TypeEvent, 9, nil, 0)
	expected = float64(now.Add(30 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("high-importance event: got %v, want ~%v", got, expected)
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

	// Low-importance event → 3d, time-sensitive doesn't shorten it further
	got = ComputeValidUntil(cfg, TypeEvent, 2, []string{"location"}, 0)
	expected = float64(now.Add(3 * 24 * time.Hour).Unix())
	if abs(got-expected) > 2 {
		t.Errorf("location-tagged event with 3d TTL: got %v, want ~%v", got, expected)
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
