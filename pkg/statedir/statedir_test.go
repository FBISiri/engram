package statedir

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDirEngramStateDirOverride(t *testing.T) {
	base := t.TempDir()
	want := filepath.Join(base, "custom-state")
	t.Setenv("ENGRAM_STATE_DIR", want)
	t.Setenv("SIRI_HOME", "/should/not/be/used")

	got, err := Dir()
	if err != nil {
		t.Fatalf("Dir() error: %v", err)
	}
	if got != want {
		t.Fatalf("Dir() = %q, want %q", got, want)
	}
	if fi, err := os.Stat(got); err != nil || !fi.IsDir() {
		t.Fatalf("expected dir created at %q: err=%v", got, err)
	}
}

func TestDirSiriHomeOverride(t *testing.T) {
	base := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", "")
	t.Setenv("SIRI_HOME", base)

	want := filepath.Join(base, ".siri")
	got, err := Dir()
	if err != nil {
		t.Fatalf("Dir() error: %v", err)
	}
	if got != want {
		t.Fatalf("Dir() = %q, want %q", got, want)
	}
	if fi, err := os.Stat(got); err != nil || !fi.IsDir() {
		t.Fatalf("expected dir created at %q: err=%v", got, err)
	}
}

// TestDirRefusesFallbackUnderTest proves the P17 test-safety guard: with no
// override and $HOME unset, Dir() refuses the production /root/.siri fallback
// while running under `go test`, so a test can NEVER resolve (and thus write
// to) the production state dir /root/.siri.
func TestDirRefusesFallbackUnderTest(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", "")
	t.Setenv("SIRI_HOME", "")
	t.Setenv("HOME", "") // registers restore; unset below
	if err := os.Unsetenv("HOME"); err != nil {
		t.Fatalf("unset HOME: %v", err)
	}

	got, err := Dir()
	if err == nil {
		t.Fatalf("Dir() = %q, want error refusing the production fallback under go test", got)
	}
	if got != "" {
		t.Fatalf("Dir() returned path %q on refusal; want empty", got)
	}
}

// TestDirAllowsTempHomeUnderTest proves the refined guard does NOT break the
// common test-isolation pattern of pointing $HOME at a t.TempDir(): that
// resolves to <tmp>/.siri (not the production dir) and is allowed.
func TestDirAllowsTempHomeUnderTest(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", "")
	t.Setenv("SIRI_HOME", "")
	base := t.TempDir()
	t.Setenv("HOME", base)

	want := filepath.Join(base, ".siri")
	got, err := Dir()
	if err != nil {
		t.Fatalf("Dir() error: %v", err)
	}
	if got != want {
		t.Fatalf("Dir() = %q, want %q", got, want)
	}
}
