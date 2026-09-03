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

func TestDirHomeUnsetFallback(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", "")
	t.Setenv("SIRI_HOME", "")
	t.Setenv("HOME", "")
	os.Unsetenv("HOME")

	// Sanity: verify os.UserHomeDir actually errors with HOME unset on linux.
	if _, err := os.UserHomeDir(); err == nil {
		t.Skip("os.UserHomeDir did not error with HOME unset on this platform")
	}

	got, err := Dir()
	if err != nil {
		t.Fatalf("Dir() error: %v", err)
	}
	want := "/root/.siri"
	if got != want {
		t.Fatalf("Dir() = %q, want %q", got, want)
	}
	if fi, err := os.Stat(got); err != nil || !fi.IsDir() {
		t.Fatalf("expected dir created at %q: err=%v", got, err)
	}
}
