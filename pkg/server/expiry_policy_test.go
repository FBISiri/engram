package server

import (
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// evapDeprecated builds an evaporation-deprecated memory created `ageDays` ago
// and deprecated `deprecatedDaysAgo` ago.
func evapDeprecated(id string, ty memory.MemoryType, importance, ageDays, deprecatedDaysAgo float64, tags ...string) *memory.Memory {
	now := float64(time.Now().Unix())
	return &memory.Memory{
		ID:              id,
		Type:            ty,
		Importance:      importance,
		Tags:            tags,
		CreatedAt:       now - ageDays*86400.0,
		LifecycleStatus: memory.LifecycleDeprecated,
		Metadata: map[string]any{
			"deprecated_reason": "evaporation",
			"deprecated_at":     now - deprecatedDaysAgo*86400.0,
		},
	}
}

// TestIsExpiryCandidate_G1ImportanceGuard is the G1 regression (spec §6.4): an
// importance=9 evaporation-deprecated memory is NOT an expiry candidate — the
// evaporation flag must not bypass the importance>=8 (P2) protection.
func TestIsExpiryCandidate_G1ImportanceGuard(t *testing.T) {
	cfg := memory.DefaultEvaporationConfig()
	now := time.Now()
	// Deprecated 60 days ago (well past the 30-day observation window).
	m := evapDeprecated("hi", memory.TypeEvent, 9, 100, 60)
	if isExpiryCandidate(m, cfg, now) {
		t.Error("importance=9 evaporation-deprecated memory must NOT be an expiry candidate (G1)")
	}
}

// TestIsExpiryCandidate_ProtectedTagGuard: an evaporation-deprecated memory
// carrying a protected tag (P5) is not a candidate even past the window.
func TestIsExpiryCandidate_ProtectedTagGuard(t *testing.T) {
	cfg := memory.DefaultEvaporationConfig()
	now := time.Now()
	m := evapDeprecated("tagged", memory.TypeEvent, 3, 100, 60, "frank-feedback")
	if isExpiryCandidate(m, cfg, now) {
		t.Error("protected-tag evaporation-deprecated memory must NOT be an expiry candidate")
	}
}

// TestIsExpiryCandidate_ObservationWindow proves spec §4.4/§6.5: an
// evaporation-deprecated memory is a candidate only after the observation
// window has elapsed.
func TestIsExpiryCandidate_ObservationWindow(t *testing.T) {
	cfg := memory.DefaultEvaporationConfig() // ObservationDays = 30
	now := time.Now()

	// Deprecated 1 day ago → still inside the window → not a candidate.
	recent := evapDeprecated("recent", memory.TypeEvent, 3, 100, 1)
	if isExpiryCandidate(recent, cfg, now) {
		t.Error("deprecated 1 day ago should NOT be a candidate (inside observation window)")
	}

	// Deprecated 31 days ago → past the window, not exempt → candidate.
	old := evapDeprecated("old", memory.TypeEvent, 3, 100, 31)
	if !isExpiryCandidate(old, cfg, now) {
		t.Error("deprecated 31 days ago SHOULD be a candidate (window elapsed, not exempt)")
	}
}

// TestIsExpiryCandidate_MissingDeprecatedAt: without a parseable deprecated_at
// the window cannot be verified, so the memory is not a candidate.
func TestIsExpiryCandidate_MissingDeprecatedAt(t *testing.T) {
	cfg := memory.DefaultEvaporationConfig()
	now := time.Now()
	m := &memory.Memory{
		ID:              "nodep",
		Type:            memory.TypeEvent,
		Importance:      3,
		CreatedAt:       float64(now.Unix()) - 100*86400.0,
		LifecycleStatus: memory.LifecycleDeprecated,
		Metadata:        map[string]any{"deprecated_reason": "evaporation"},
	}
	if isExpiryCandidate(m, cfg, now) {
		t.Error("evaporation-deprecated memory with no deprecated_at must NOT be a candidate")
	}
}
