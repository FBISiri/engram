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

func TestShouldHoldForConsolidate(t *testing.T) {
	cases := []struct {
		failed, total int
		want          bool
	}{
		{0, 0, false},  // no candidates
		{0, 5, false},  // none failed
		{2, 5, false},  // 40% <= 50%
		{3, 5, true},   // 60% > 50%
		{1, 1, true},   // 100%
		{5, 10, false}, // exactly 50% is not > threshold
		{6, 10, true},  // 60%
	}
	for _, c := range cases {
		if got := shouldHoldForConsolidate(c.failed, c.total); got != c.want {
			t.Errorf("shouldHoldForConsolidate(%d,%d)=%v, want %v", c.failed, c.total, got, c.want)
		}
	}
}

// LIVE run with a consolidate candidate but no usable LLM credentials must
// abort up-front with ErrLLMUnavailable, NOT advance last_run, and NOT create
// the last_run file.
func TestRun_LiveNoLLM_AbortsAndHolds(t *testing.T) {
	stateDir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", stateDir)
	t.Setenv("ENGRAM_DREAM_REPORT_DIR", t.TempDir())
	t.Setenv("ENGRAM_DREAM_ALLOW_FIRST_RUN", "1") // isolate from first-run guard
	// openrouter + no key => llm.Preflight fails.
	t.Setenv("ENGRAM_LLM_PROVIDER", "")
	t.Setenv("ENGRAM_LLM_API_KEY", "")

	now := float64(time.Now().Unix())
	events := []memory.Memory{
		{ID: "e1", Type: memory.TypeEvent, Tags: []string{"topic"}, CreatedAt: now, Content: "a"},
		{ID: "e2", Type: memory.TypeEvent, Tags: []string{"topic"}, CreatedAt: now, Content: "b"},
		{ID: "e3", Type: memory.TypeEvent, Tags: []string{"topic"}, CreatedAt: now, Content: "c"},
	}
	store := &fakeStore{all: events}

	eng := NewEngine(store, nil, Config{DryRun: false, Phase: ""})
	err := eng.Run(context.Background())
	if !errors.Is(err, ErrLLMUnavailable) {
		t.Fatalf("expected ErrLLMUnavailable, got %v", err)
	}
	if eng.log.LastRunAdvanced {
		t.Fatal("last_run must NOT be advanced on llm-unavailable abort")
	}
	if !eng.log.Degraded {
		t.Fatal("expected Degraded=true")
	}
	if _, serr := os.Stat(filepath.Join(stateDir, lastRunFile)); !os.IsNotExist(serr) {
		t.Fatalf("last_run file must not be created; stat err=%v", serr)
	}
}
