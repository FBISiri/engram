package qdrant

import (
	"testing"
	"time"
)

// TestDecideExpiryDwell exercises the pure per-point dwell decision: a point
// that just became delete-eligible must be STAMPED (not deleted), a point still
// inside the window must WAIT, and only a point past the window may be DELETED.
// A zero/negative window disables the grace (legacy immediate delete).
func TestDecideExpiryDwell(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	dwell := 72 * time.Hour

	cases := []struct {
		name       string
		eligibleAt float64
		validUntil float64
		dwell      time.Duration
		want       expiryDwellDecision
	}{
		{
			name:       "never stamped -> stamp and skip",
			eligibleAt: 0,
			validUntil: float64(now.Add(-time.Hour).Unix()),
			dwell:      dwell,
			want:       dwellStamp,
		},
		{
			name:       "just marked, inside window -> wait",
			eligibleAt: float64(now.Add(-1 * time.Hour).Unix()),
			validUntil: float64(now.Add(-2 * time.Hour).Unix()),
			dwell:      dwell,
			want:       dwellWait,
		},
		{
			name:       "one second before window closes -> wait",
			eligibleAt: float64(now.Add(-dwell + time.Second).Unix()),
			validUntil: float64(now.Add(-dwell).Unix()),
			dwell:      dwell,
			want:       dwellWait,
		},
		{
			name:       "exactly at window -> delete",
			eligibleAt: float64(now.Add(-dwell).Unix()),
			validUntil: float64(now.Add(-dwell - time.Hour).Unix()),
			dwell:      dwell,
			want:       dwellDelete,
		},
		{
			name:       "well past window -> delete",
			eligibleAt: float64(now.Add(-2 * dwell).Unix()),
			validUntil: float64(now.Add(-3 * dwell).Unix()),
			dwell:      dwell,
			want:       dwellDelete,
		},
		{
			name:       "stale stamp predating current valid_until (revived) -> re-stamp",
			eligibleAt: float64(now.Add(-2 * dwell).Unix()),   // old stamp from a previous episode
			validUntil: float64(now.Add(-time.Minute).Unix()), // re-expired just now, after that stamp
			dwell:      dwell,
			want:       dwellStamp,
		},
		{
			name:       "dwell disabled (zero) -> immediate delete even if just stamped",
			eligibleAt: float64(now.Unix()),
			validUntil: float64(now.Add(-time.Hour).Unix()),
			dwell:      0,
			want:       dwellDelete,
		},
		{
			name:       "dwell disabled (negative) -> immediate delete, unstamped",
			eligibleAt: 0,
			validUntil: float64(now.Add(-time.Hour).Unix()),
			dwell:      -1,
			want:       dwellDelete,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := decideExpiryDwell(tc.eligibleAt, tc.validUntil, tc.dwell, now)
			if got != tc.want {
				t.Errorf("decideExpiryDwell(eligibleAt=%v, validUntil=%v, dwell=%v) = %d, want %d",
					tc.eligibleAt, tc.validUntil, tc.dwell, got, tc.want)
			}
		})
	}
}

// TestPlanExpiryDwellMarkThenDelete proves the mark-then-delete-later behaviour
// end-to-end over the pure planner: a batch of just-expired points is only
// STAMPED (never deleted) on the first pass, and the SAME point becomes
// deletable only after the dwell window has elapsed since its stamp.
func TestPlanExpiryDwellMarkThenDelete(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	dwell := 72 * time.Hour

	// First pass: three freshly-expired points, none stamped yet.
	validUntil := float64(now.Add(-time.Hour).Unix()) // expired an hour ago
	firstPass := []expiredPoint{
		{id: "a", eligibleAt: 0, validUntil: validUntil},
		{id: "b", eligibleAt: 0, validUntil: validUntil},
		{id: "c", eligibleAt: 0, validUntil: validUntil},
	}
	plan := planExpiryDwell(firstPass, dwell, now)
	if len(plan.toDelete) != 0 {
		t.Fatalf("first pass deleted %v; a just-marked object must NOT be deleted inside the dwell window", plan.toDelete)
	}
	if got := len(plan.toStamp); got != 3 {
		t.Fatalf("first pass stamped %d points, want 3", got)
	}

	// Second pass (still inside window): the points now carry the stamp written
	// on the first pass. They must still WAIT (neither stamped again nor deleted).
	stampedAt := float64(now.Unix())
	insideWindow := now.Add(1 * time.Hour)
	secondPass := []expiredPoint{
		{id: "a", eligibleAt: stampedAt, validUntil: validUntil},
		{id: "b", eligibleAt: stampedAt, validUntil: validUntil},
		{id: "c", eligibleAt: stampedAt, validUntil: validUntil},
	}
	plan = planExpiryDwell(secondPass, dwell, insideWindow)
	if len(plan.toDelete) != 0 || len(plan.toStamp) != 0 {
		t.Fatalf("inside window: toStamp=%v toDelete=%v; want both empty (wait)", plan.toStamp, plan.toDelete)
	}

	// Third pass (after the window): the same stamped points are now eligible.
	afterWindow := now.Add(dwell + time.Minute)
	plan = planExpiryDwell(secondPass, dwell, afterWindow)
	if len(plan.toStamp) != 0 {
		t.Fatalf("after window: stamped %v; want none", plan.toStamp)
	}
	if len(plan.toDelete) != 3 {
		t.Fatalf("after window: deleted %v; want all 3 eligible", plan.toDelete)
	}
}

// TestResolveExpiryDwellWindow verifies env parsing + default for the dwell
// window (the decision-age grace), including the explicit-disable path.
func TestResolveExpiryDwellWindow(t *testing.T) {
	t.Setenv("ENGRAM_EXPIRY_DWELL_HOURS", "")
	if got := resolveExpiryDwellWindow(); got != defaultExpiryDwellWindow {
		t.Errorf("unset -> %v, want default %v", got, defaultExpiryDwellWindow)
	}

	t.Setenv("ENGRAM_EXPIRY_DWELL_HOURS", "24")
	if got := resolveExpiryDwellWindow(); got != 24*time.Hour {
		t.Errorf("24 -> %v, want 24h", got)
	}

	t.Setenv("ENGRAM_EXPIRY_DWELL_HOURS", "0")
	if got := resolveExpiryDwellWindow(); got != 0 {
		t.Errorf("0 -> %v, want 0 (disabled)", got)
	}

	t.Setenv("ENGRAM_EXPIRY_DWELL_HOURS", "not-a-number")
	if got := resolveExpiryDwellWindow(); got != defaultExpiryDwellWindow {
		t.Errorf("garbage -> %v, want default %v", got, defaultExpiryDwellWindow)
	}
}
