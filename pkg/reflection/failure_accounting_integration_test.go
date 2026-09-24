package reflection

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/llm"
)

// TestFailureAccounting_Stage1_429_Reachable is the regression test for the
// STRUCTURALLY-UNREACHABLE failure-backoff bug: the dominant production failure
// mode is a Stage 1 LLM 429, and RunV2 returns (result, nil) on that path. The
// OLD code only called recordFailure() inside "zero output produced" branches,
// so a Stage 1 429 never incremented consecutive_failures and the backoff never
// fired (cadence degenerated to a fixed 20 min).
//
// This test drives the REAL integration path — a mocked Stage 1 429 through the
// actual transient-retry stack and the actual RunV2 unified exit — and MUST NOT
// construct the failure count directly (that shortcut is exactly the false-green
// defect B2 that hid the bug for 9 consecutive failed runs).
func TestFailureAccounting_Stage1_429_Reachable(t *testing.T) {
	// Redirect the Siri state dir to a temp dir so the test neither requires
	// root (writes under /root/.siri) nor clobbers the live production state.
	stateDir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", stateDir)
	// Neutralise the other resolution inputs for hermeticity.
	t.Setenv("SIRI_HOME", "")

	// Mock Stage 1 to return a 429-class error, exercising the whole
	// callLLMWithRetry → callWithTransientRetry classification path.
	origLLM := callLLMFunc
	callLLMFunc = func(_ context.Context, _ string) (string, error) {
		return "", &llm.StatusError{StatusCode: 429, RetryAfter: "", Body: "rate limited"}
	}
	t.Cleanup(func() { callLLMFunc = origLLM })

	// Make the transient-retry stack exhaust instantly: no real sleeping, zero
	// jitter, a couple of attempts. This keeps the 429 path faithful (it still
	// classifies 429 as transient and retries) without a multi-minute test.
	origCfg := defaultTransientRetryConfig
	defaultTransientRetryConfig = func() transientRetryConfig {
		return transientRetryConfig{
			maxAttempts: 2,
			baseBackoff: time.Millisecond,
			maxBackoff:  time.Millisecond,
			totalBudget: time.Minute,
			sleep:       func(_ context.Context, _ time.Duration) error { return nil },
			rand:        func() float64 { return 0 },
		}
	}
	t.Cleanup(func() { defaultTransientRetryConfig = origCfg })

	cfg := Config{
		Threshold:      10,
		MinIntervalH:   2.0,
		Mode:           "v2",
		FocalInputSize: 50,
		FocalQuestions: 3,
	}
	eng, store := makeEngineWithEmbedder(t, cfg, &mockEmbedder{dim: 8}) // ENGRAM_STATE_DIR wins over HOME
	// Enough unreflected importance to trigger, and >= minEvidenceCount sources.
	for i := 0; i < 5; i++ {
		addMemory(store, 10, false) // 5*10 = 50 > threshold 10
	}

	countPath := filepath.Join(stateDir, reflectionFailureCountFile)
	attemptPath := filepath.Join(stateDir, reflectionLastAttemptFile)

	// Pre-condition: state files do not exist yet and count reads as 0.
	if _, err := os.Stat(countPath); !os.IsNotExist(err) {
		t.Fatalf("precondition: expected no %s yet, stat err=%v", reflectionFailureCountFile, err)
	}
	if n, _ := readFailureCount(countPath); n != 0 {
		t.Fatalf("precondition: expected count 0, got %d", n)
	}

	// ── First trigger: Stage 1 429 → failed run ──────────────────────────────
	res, err := eng.Run(context.Background())
	if err != nil {
		t.Fatalf("Run returned transport error (want nil, run failure lives in result.Errors): %v", err)
	}
	if !res.Triggered {
		t.Fatalf("expected run to trigger (skip_reason=%q)", res.SkipReason)
	}
	if len(res.Errors) == 0 {
		t.Fatalf("expected result.Errors populated by Stage 1 429, got none")
	}

	// (a) consecutive_failures went 0 → 1 via the real integration path.
	n, err := readFailureCount(countPath)
	if err != nil {
		t.Fatalf("readFailureCount: %v", err)
	}
	if n != 1 {
		t.Fatalf("expected consecutive_failures 0→1 after Stage 1 429, got %d", n)
	}

	// (b) both state files were actually created on disk.
	if _, err := os.Stat(countPath); err != nil {
		t.Fatalf("expected %s created: %v", reflectionFailureCountFile, err)
	}
	if _, err := os.Stat(attemptPath); err != nil {
		t.Fatalf("expected %s created: %v", reflectionLastAttemptFile, err)
	}

	// (c) On the next trigger the backoff is in effect: the failure-backoff gate
	// blocks the run and the remaining wait exceeds the fixed 20-min cadence
	// (computeFailureBackoff(1) = 40m). This proves the cadence is no longer
	// pinned at 20 min after a failure.
	check, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("second Check: %v", err)
	}
	if check.ShouldTrigger {
		t.Fatalf("expected second Check blocked by failure backoff, but ShouldTrigger=true")
	}
	if check.ConsecutiveFailures != 1 {
		t.Fatalf("expected ConsecutiveFailures=1 on second Check, got %d", check.ConsecutiveFailures)
	}
	if check.FailureBackoffRemaining <= 20 {
		t.Fatalf("expected next-run backoff > 20 min, got %.1f min (skip_reason=%q)",
			check.FailureBackoffRemaining, check.SkipReason)
	}
}
