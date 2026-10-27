package dream

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// gate_test.go reuses fakeStore/buildFakeStore from engine_test.go.

func TestCheckGates_FirstRunGuardBlocks(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")
	store := buildFakeStore(5) // non-empty store, no last_run file

	res, err := CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if !res.FirstRunGuard {
		t.Fatal("expected FirstRunGuard=true when last_run=never and store non-empty")
	}
	if res.ShouldRun {
		t.Fatal("expected ShouldRun=false under first-run guard")
	}
	if res.StoreTotal != 5 {
		t.Fatalf("StoreTotal=%d, want 5", res.StoreTotal)
	}
}

func TestCheckGates_FirstRunGuardBypass(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "1")
	store := buildFakeStore(5)

	res, err := CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("expected FirstRunGuard=false when bypass env set")
	}
	if !res.Gate1Time {
		t.Fatal("expected gate1 to pass (last_run=never) with bypass")
	}
}

func TestCheckGates_EmptyStoreNotGuarded(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")
	store := buildFakeStore(0)

	res, err := CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("expected FirstRunGuard=false for empty store")
	}
}

func TestCheckGates_LastRunPresentNotGuarded(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")
	// Write a last_run file (recent) so lastRun is not zero.
	if err := os.WriteFile(filepath.Join(dir, lastRunFile),
		[]byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		t.Fatal(err)
	}
	store := buildFakeStore(100)

	res, err := CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("expected FirstRunGuard=false when last_run file present")
	}
}

func TestCheckGates_StoreNilGate2Pass(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")

	res, err := CheckGates(nil)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("store-nil path must not be guarded")
	}
	if !res.Gate2Memories {
		t.Fatal("gate2 should pass by default when store is nil")
	}
}

func TestFirstRunGuard_Func(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")

	blocked, reason, err := FirstRunGuard(buildFakeStore(3))
	if err != nil {
		t.Fatalf("FirstRunGuard: %v", err)
	}
	if !blocked || reason == "" {
		t.Fatalf("expected blocked with reason, got blocked=%v reason=%q", blocked, reason)
	}

	// bypass
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "1")
	blocked, _, err = FirstRunGuard(buildFakeStore(3))
	if err != nil {
		t.Fatalf("FirstRunGuard bypass: %v", err)
	}
	if blocked {
		t.Fatal("expected not blocked with bypass env")
	}

	// nil store
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")
	blocked, _, err = FirstRunGuard(nil)
	if err != nil {
		t.Fatalf("FirstRunGuard nil: %v", err)
	}
	if blocked {
		t.Fatal("expected not blocked for nil store")
	}
}

// statsErrStore is a fakeStore whose Stats always errors (Scroll still works).
type statsErrStore struct{ *fakeStore }

func (s statsErrStore) Stats(ctx context.Context) (*memory.CollectionStats, error) {
	return nil, errors.New("qdrant stats boom")
}

// Fail CLOSED: last_run=never + Stats error must guard (CheckGates + the
// FirstRunGuard helper); a present last_run or the bypass env must not.
func TestFirstRunGuard_StatsErrorFailsClosed(t *testing.T) {
	store := statsErrStore{buildFakeStore(5)}

	// no last_run file -> guarded
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "")
	res, err := CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates: %v", err)
	}
	if !res.FirstRunGuard || res.ShouldRun {
		t.Fatalf("stats error + never must guard: FirstRunGuard=%v ShouldRun=%v", res.FirstRunGuard, res.ShouldRun)
	}
	if blocked, reason, gerr := FirstRunGuard(store); gerr != nil || !blocked || reason == "" {
		t.Fatalf("FirstRunGuard stats-error: blocked=%v reason=%q err=%v", blocked, reason, gerr)
	}

	// last_run file present -> NOT guarded (lenient)
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	if werr := os.WriteFile(filepath.Join(dir, lastRunFile),
		[]byte(time.Now().UTC().Format(time.RFC3339)), 0644); werr != nil {
		t.Fatal(werr)
	}
	res, err = CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates(last_run set): %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("stats error with last_run set must NOT guard")
	}
	if blocked, _, gerr := FirstRunGuard(store); gerr != nil || blocked {
		t.Fatalf("FirstRunGuard(last_run set): blocked=%v err=%v", blocked, gerr)
	}

	// bypass env -> NOT guarded even with no last_run + stats error
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "1")
	res, err = CheckGates(store)
	if err != nil {
		t.Fatalf("CheckGates(bypass): %v", err)
	}
	if res.FirstRunGuard {
		t.Fatal("bypass env must disable guard")
	}
	if blocked, _, gerr := FirstRunGuard(store); gerr != nil || blocked {
		t.Fatalf("FirstRunGuard(bypass): blocked=%v err=%v", blocked, gerr)
	}
}
