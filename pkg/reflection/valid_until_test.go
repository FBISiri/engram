package reflection

import (
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// TestInsightMemory_ValidUntil_SetBy30dTTL is a lock-in test for R-S2.
// Reflection-origin insights MUST carry a 30-day TTL (valid_until > 0).
// Regression: pre-1b3e482 insights were stored without valid_until because
// the WithValidUntil option was not yet wired into the construction path.
func TestInsightMemory_ValidUntil_SetBy30dTTL(t *testing.T) {
	reflectionValidUntil := float64(time.Now().Add(30 * 24 * time.Hour).Unix())

	tags := ensureSourceReflectionTag([]string{"pattern"})
	sourceIDsAny := []any{"src-1", "src-2"}

	insightMem := memory.New("user prefers concise answers",
		memory.WithType(memory.TypeInsight),
		memory.WithSource("system"),
		memory.WithImportance(7.0),
		memory.WithTags(tags...),
		memory.WithConfidence(0.85),
		memory.WithValidUntil(reflectionValidUntil),
		memory.WithMetadata(map[string]any{
			"reflection_source_ids": sourceIDsAny,
			"reflection_count":      2,
			"trigger_importance":    12.5,
		}),
	)

	if insightMem.ValidUntil <= 0 {
		t.Fatalf("ValidUntil must be > 0 for reflection-origin insight, got %f", insightMem.ValidUntil)
	}

	nowUnix := float64(time.Now().Unix())
	thirtyDays := float64(30 * 24 * 3600)
	diff := insightMem.ValidUntil - nowUnix

	if diff < thirtyDays-60 || diff > thirtyDays+60 {
		t.Errorf("ValidUntil should be ~30 days from now (±60s); got delta=%f seconds", diff)
	}
}

// TestInsightMemory_ValidUntil_NotOverwrittenBySubsequentOptions verifies
// that WithMetadata (applied after WithValidUntil) does not reset ValidUntil.
func TestInsightMemory_ValidUntil_NotOverwrittenBySubsequentOptions(t *testing.T) {
	ttl := float64(time.Now().Add(30 * 24 * time.Hour).Unix())

	mem := memory.New("test",
		memory.WithValidUntil(ttl),
		memory.WithMetadata(map[string]any{"key": "value"}),
	)

	if mem.ValidUntil != ttl {
		t.Errorf("ValidUntil changed after WithMetadata: got %f, want %f", mem.ValidUntil, ttl)
	}
}
