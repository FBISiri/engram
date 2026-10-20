package server

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/qdrant"
)

// TestMain isolates the whole pkg/server test binary from the production state
// dir: statedir.Dir() now REFUSES the $HOME//root/.siri fallback under `go
// test`, so any test that resolves the state dir (delete-audit, reflection
// last-run, etc.) without its own override would otherwise error. Defaulting
// ENGRAM_STATE_DIR to a throwaway temp dir guarantees no test can ever write to
// /root/.siri; individual tests may still t.Setenv it to a per-test dir.
func TestMain(m *testing.M) {
	if os.Getenv("ENGRAM_STATE_DIR") == "" {
		dir, err := os.MkdirTemp("", "engram-server-statedir-")
		if err != nil {
			panic(err)
		}
		_ = os.Setenv("ENGRAM_STATE_DIR", dir)
		code := m.Run()
		_ = os.RemoveAll(dir)
		os.Exit(code)
	}
	os.Exit(m.Run())
}

// TestExpiryTickDeletedCounterPositive is the POSITIVE control for issue (2):
// the expiry log historically only ever showed "deleted=0", so this proves the
// hard-deleted counter actually INCREMENTS to a non-zero value when a single
// expired memory is deleted. Exercises the legacy (count-only) branch at
// expiry.go ~100-117.
func TestExpiryTickDeletedCounterPositive(t *testing.T) {
	// Even though the legacy branch does not write an audit record, set the
	// override so no code path can ever resolve the production state dir.
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))

	past := float64(time.Now().Add(-time.Hour).Unix())
	store := &expiryObsStore{mems: []memory.Memory{
		{ID: "d1", Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}},
	}}

	m := metrics.New(nil, nil)

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	prev := setStderrWriter(w)
	runExpiryTick(context.Background(), store, m, time.Now())
	setStderrWriter(prev)
	_ = w.Close()
	out, _ := io.ReadAll(r)

	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")); got != 1 {
		t.Fatalf("EvaporationHardDeletedTotal{event}=%v, want 1 (one expired memory deleted)", got)
	}
	if !strings.Contains(string(out), "[expiry] deleted 1 expired memories") {
		t.Fatalf("log missing positive deleted line; got %q", string(out))
	}
}

// TestExpiryTickProtectedHealedCounters proves that when a protected
// (importance>=8) memory is skipped and another is self-healed, the
// skipped/healed COUNTS increment while the deleted counter stays 0 — the
// negative half of issue (2). The detailed branch surfaces healed/protected as
// log-line counts (there is no dedicated prometheus metric for them, and
// pkg/metrics is out of scope), so we assert on those counts plus the
// unchanged hard-deleted metric.
func TestExpiryTickProtectedHealedCounters(t *testing.T) {
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))

	past := float64(time.Now().Add(-time.Hour).Unix())
	evap := func(id string, ty memory.MemoryType) memory.Memory {
		return memory.Memory{ID: id, Type: ty, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}}
	}
	store := &detailedStore{
		expiryObsStore: &expiryObsStore{mems: []memory.Memory{
			evap("prot", memory.TypeEvent),
			evap("heal", memory.TypeInsight),
		}},
		records: []qdrant.ExpiryRecord{
			{ID: "prot", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 9},
			{ID: "heal", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionHealed, Importance: 10},
		},
	}

	m := metrics.New(nil, nil)

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	prev := setStderrWriter(w)
	runExpiryTick(context.Background(), store, m, time.Now())
	setStderrWriter(prev)
	_ = w.Close()
	outBytes, _ := io.ReadAll(r)
	out := string(outBytes)

	// deleted stays 0: nothing was physically deleted.
	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")); got != 0 {
		t.Fatalf("EvaporationHardDeletedTotal{event}=%v, want 0 (protected, not deleted)", got)
	}
	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("insight")); got != 0 {
		t.Fatalf("EvaporationHardDeletedTotal{insight}=%v, want 0 (healed, not deleted)", got)
	}
	// Positive evidence that the skipped/healed counts incremented.
	for _, want := range []string{"deleted=0", "healed=1 [heal]", "protected=1 [prot]"} {
		if !strings.Contains(out, want) {
			t.Fatalf("log missing %q; got %q", want, out)
		}
	}
}
