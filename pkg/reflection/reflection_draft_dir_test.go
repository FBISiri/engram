package reflection

import (
	"path/filepath"
	"testing"
)

// TestReflectionDraftDir_StateDirOverride verifies ENGRAM_STATE_DIR is honored so
// the draft dir resolves under the override rather than the fallback literal.
func TestReflectionDraftDir_StateDirOverride(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", tmp)
	got := reflectionDraftDir()
	want := filepath.Join(tmp, "Reflection", "drafts")
	if got != want {
		t.Fatalf("reflectionDraftDir() = %q, want %q", got, want)
	}
}
