package reflection

import (
	"strings"
	"testing"
)

// TestFocalFallbackQuestions_Preset verifies that with no persisted file the
// fallback returns the generic preset set, tagged "preset", never empty.
func TestFocalFallbackQuestions_Preset(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())

	q, src := focalFallbackQuestions(0)
	if src != "preset" {
		t.Errorf("source = %q, want preset", src)
	}
	if len(q) == 0 {
		t.Fatal("preset fallback must never be empty")
	}
	if len(q) != len(presetFocalQuestions) {
		t.Errorf("uncapped len = %d, want %d", len(q), len(presetFocalQuestions))
	}
}

// TestFocalFallbackQuestions_Persisted verifies that a persisted set is
// preferred over the preset and tagged "persisted".
func TestFocalFallbackQuestions_Persisted(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())

	saved := []string{"saved q1", "saved q2", "saved q3"}
	if err := persistFocalQuestions(saved); err != nil {
		t.Fatalf("persistFocalQuestions: %v", err)
	}

	q, src := focalFallbackQuestions(0)
	if src != "persisted" {
		t.Errorf("source = %q, want persisted", src)
	}
	if len(q) != len(saved) || q[0] != "saved q1" {
		t.Errorf("questions = %v, want %v", q, saved)
	}
}

// TestFocalFallbackQuestions_CapAndNonEmpty verifies the n cap and the
// never-empty invariant across both sources.
func TestFocalFallbackQuestions_CapAndNonEmpty(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())

	// Preset capped to 2.
	q, src := focalFallbackQuestions(2)
	if src != "preset" || len(q) != 2 {
		t.Errorf("preset cap: src=%q len=%d, want preset/2", src, len(q))
	}

	// Persisted capped to 1.
	if err := persistFocalQuestions([]string{"a", "b", "c", "d"}); err != nil {
		t.Fatalf("persistFocalQuestions: %v", err)
	}
	q, src = focalFallbackQuestions(1)
	if src != "persisted" || len(q) != 1 || q[0] != "a" {
		t.Errorf("persisted cap: src=%q q=%v, want persisted/[a]", src, q)
	}

	// n larger than available does not pad or truncate.
	q, _ = focalFallbackQuestions(100)
	if len(q) != 4 {
		t.Errorf("n>len: got %d, want 4", len(q))
	}
	if len(q) == 0 {
		t.Fatal("must never be empty")
	}
}

// TestPersistLoadFocalRoundTrip verifies persist -> load round-trips.
func TestPersistLoadFocalRoundTrip(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())

	// No file yet -> load errors (missing).
	if _, err := loadPersistedFocalQuestions(); err == nil {
		t.Error("expected error loading nonexistent focal file")
	}

	want := []string{"one", "two"}
	if err := persistFocalQuestions(want); err != nil {
		t.Fatalf("persist: %v", err)
	}
	got, err := loadPersistedFocalQuestions()
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(got) != 2 || got[0] != "one" || got[1] != "two" {
		t.Errorf("round-trip = %v, want %v", got, want)
	}
}

// TestFormatLastRateLimit_None verifies the "none" path when no 429 observed.
// (LastRateLimit is process-global; we only assert the nil branch is reachable
// via a substring check that tolerates a prior 429 in the same process.)
func TestFormatLastRateLimit_None(t *testing.T) {
	s := formatLastRateLimit()
	if s == "" {
		t.Error("formatLastRateLimit must never return empty")
	}
	if s != "none" && !strings.Contains(s, "status=") {
		t.Errorf("unexpected format: %q", s)
	}
}
