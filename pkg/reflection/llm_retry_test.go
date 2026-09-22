package reflection

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/llm"
)

// fastTransientCfg returns a transientRetryConfig with a fake sleep that records
// requested durations instead of sleeping, deterministic rand=1.0 (full backoff),
// and a generous wall budget so tests run instantly with no real network/time.
func fastTransientCfg(sleeps *[]time.Duration) transientRetryConfig {
	return transientRetryConfig{
		maxAttempts: 4,
		baseBackoff: 1 * time.Millisecond,
		maxBackoff:  10 * time.Millisecond,
		totalBudget: 60 * time.Second,
		rand:        func() float64 { return 1.0 },
		sleep: func(_ context.Context, d time.Duration) error {
			*sleeps = append(*sleeps, d)
			return nil
		},
	}
}

func TestDefaultTransientRetryConfig_BackoffUnderRunBudget(t *testing.T) {
	// INVARIANT: the max total backoff implied by the PRODUCTION policy must be
	// strictly less than the reflection run budget (30min, per
	// pkg/server/reflection_async.go:111). Assert against
	// defaultTransientRetryConfig() itself so any future retune that breaks the
	// bound fails CI. Uses cfg.backoffFor math (rand=1.0) — no real sleeping.
	const runBudget = 30 * time.Minute
	cfg := defaultTransientRetryConfig()
	cfg.rand = func() float64 { return 1.0 } // worst-case full-jitter sleeps

	// Theoretical worst case: sum of the max backoff for every inter-attempt gap
	// (attempts 0..maxAttempts-2; the last attempt never sleeps).
	var sumMax time.Duration
	for attempt := 0; attempt < cfg.maxAttempts-1; attempt++ {
		sumMax += cfg.backoffFor(attempt)
	}
	if sumMax >= runBudget {
		t.Errorf("sum of max backoffs %s >= run budget %s", sumMax, runBudget)
	}

	// The WithTimeout deadline HARD-CAPS actual total backoff at totalBudget, so
	// that too must stay under the run budget.
	if cfg.totalBudget >= runBudget {
		t.Errorf("totalBudget %s >= run budget %s", cfg.totalBudget, runBudget)
	}

	// The effective (deadline-capped) worst-case total backoff.
	effective := sumMax
	if cfg.totalBudget < effective {
		effective = cfg.totalBudget
	}
	if effective >= runBudget {
		t.Errorf("effective max total backoff %s >= run budget %s", effective, runBudget)
	}

	// Sanity: policy must actually be minute-scale (guards against a regression
	// back to the too-small budget that motivated this retune).
	if cfg.baseBackoff < time.Minute {
		t.Errorf("baseBackoff %s not minute-scale", cfg.baseBackoff)
	}
	if cfg.maxBackoff <= 20*time.Second {
		t.Errorf("maxBackoff %s too low (single sleep can't reach minute scale)", cfg.maxBackoff)
	}
	if cfg.maxAttempts < 7 {
		t.Errorf("maxAttempts %d too low", cfg.maxAttempts)
	}
}

// Retry-After larger than the remaining budget is NOT silently swallowed: the
// loop fails fast and the returned error names the requested delay.
func TestCallWithTransientRetry_RetryAfterOverBudget(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, _ int) (string, llm.Meta, error) {
		calls++
		// Server asks for a 1h wait, far beyond the tiny totalBudget.
		return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, RetryAfter: "3600", Body: "rate"}
	}
	var sleeps []time.Duration
	cfg := transientRetryConfig{
		maxAttempts: 8, baseBackoff: time.Second, maxBackoff: 10 * time.Second,
		totalBudget: 100 * time.Millisecond, rand: func() float64 { return 1.0 },
		sleep: func(_ context.Context, d time.Duration) error {
			sleeps = append(sleeps, d)
			return nil
		},
	}
	_, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Fatalf("over-budget Retry-After must stop after 1 call, got %d", calls)
	}
	if len(sleeps) != 0 {
		t.Fatalf("must not sleep past the deadline, got %v", sleeps)
	}
	msg := err.Error()
	for _, want := range []string{"rate limited", "Retry-After", "1h0m0s", "exceeds remaining budget"} {
		if !strings.Contains(msg, want) {
			t.Errorf("over-budget error missing %q: %q", want, msg)
		}
	}
}

func TestTruncationError(t *testing.T) {
	// Non-length finish reason => nil.
	if err := truncationError("focal", llm.Meta{FinishReason: "stop"}); err != nil {
		t.Fatalf("expected nil for finish_reason=stop, got %v", err)
	}
	// Length => wrapped ErrTruncatedResponse, no format/JSON wording.
	err := truncationError("dialectic q1", llm.Meta{
		FinishReason: "length", MaxTokens: 4000, CompletionTokens: 4000, ReasoningTokens: 3900, RawLen: 12,
	})
	if err == nil {
		t.Fatal("expected error for finish_reason=length")
	}
	if !errors.Is(err, ErrTruncatedResponse) {
		t.Errorf("errors.Is(ErrTruncatedResponse) = false, err=%v", err)
	}
	msg := strings.ToLower(err.Error())
	for _, bad := range []string{"json", "parse", "format"} {
		if strings.Contains(msg, bad) {
			t.Errorf("truncation message must not mention %q: %q", bad, err.Error())
		}
	}
}

func TestCallLLMWithRetry_RetriesOnLength(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var budgets []int
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		budgets = append(budgets, maxTokens)
		calls++
		if calls == 1 {
			return "trunc", llm.Meta{FinishReason: "length", MaxTokens: maxTokens}, nil
		}
		return "full", llm.Meta{FinishReason: "stop", MaxTokens: maxTokens}, nil
	}

	resp, meta, err := callLLMWithRetry(context.Background(), "p", 4000, 8000, "dialectic q1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != "full" || meta.FinishReason != "stop" {
		t.Errorf("expected second (retry) result, got resp=%q meta=%+v", resp, meta)
	}
	if calls != 2 {
		t.Fatalf("expected exactly 2 calls, got %d", calls)
	}
	if len(budgets) != 2 || budgets[0] != 4000 || budgets[1] != 8000 {
		t.Errorf("budget escalation = %v, want [4000 8000]", budgets)
	}
}

func TestCallLLMWithRetry_BothLength(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		calls++
		return "trunc", llm.Meta{FinishReason: "length", MaxTokens: maxTokens}, nil
	}

	_, meta, err := callLLMWithRetry(context.Background(), "p", 4000, 8000, "focal")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
	if meta.FinishReason != "length" {
		t.Errorf("expected length meta so caller emits truncation, got %+v", meta)
	}
	if terr := truncationError("focal", meta); terr == nil {
		t.Error("expected truncationError to be non-nil for both-length outcome")
	}
}

func TestDialecticPath_RecordsTruncationNotParseError(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	// Both attempts return a truncated fragment with finish_reason=length.
	callLLMMetaBudget = func(_ context.Context, _ string, maxTokens int) (string, llm.Meta, error) {
		return `{"content":"partial frag`, llm.Meta{
			FinishReason: "length", MaxTokens: maxTokens, CompletionTokens: maxTokens, ReasoningTokens: maxTokens - 100,
		}, nil
	}

	e := dialecticTestEngine()
	evidenceList := []PerQuestionEvidence{
		{Question: "q1", Evidence: makeEvidence("e1", "e2", "e3")},
	}

	_, stats, err := e.generateDialecticInsights(context.Background(), evidenceList, e.cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if stats.FailedCount != 1 {
		t.Errorf("expected FailedCount=1, got %d", stats.FailedCount)
	}
	if len(stats.Errors) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(stats.Errors), stats.Errors)
	}
	msg := strings.ToLower(stats.Errors[0])
	if strings.Contains(msg, "json") || strings.Contains(msg, "parse") {
		t.Errorf("expected budget/truncation error, got format error: %q", stats.Errors[0])
	}
	if !strings.Contains(msg, "truncated") {
		t.Errorf("expected truncation wording, got %q", stats.Errors[0])
	}
}

// (a) 429-then-200 succeeds: two calls, second result returned.
func TestCallWithTransientRetry_429ThenOK(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var sleeps []time.Duration
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, mt int) (string, llm.Meta, error) {
		calls++
		if calls == 1 {
			return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, Body: "rate"}
		}
		return "full", llm.Meta{FinishReason: "stop", MaxTokens: mt}, nil
	}

	resp, meta, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", fastTransientCfg(&sleeps))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 {
		t.Fatalf("expected 2 calls, got %d", calls)
	}
	if resp != "full" || meta.FinishReason != "stop" {
		t.Errorf("expected success result, got resp=%q meta=%+v", resp, meta)
	}
	if len(sleeps) != 1 {
		t.Errorf("expected 1 backoff sleep, got %v", sleeps)
	}
}

// (b) Retry-After honoured: recorded sleep == 2s, not the computed backoff.
func TestCallWithTransientRetry_RetryAfterHonoured(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var sleeps []time.Duration
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, mt int) (string, llm.Meta, error) {
		calls++
		if calls == 1 {
			return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, RetryAfter: "2", Body: "rate"}
		}
		return "ok", llm.Meta{FinishReason: "stop", MaxTokens: mt}, nil
	}

	if _, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", fastTransientCfg(&sleeps)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(sleeps) != 1 || sleeps[0] != 2*time.Second {
		t.Errorf("expected Retry-After 2s sleep, got %v", sleeps)
	}
}

// (c) 5xx retried then 200.
func TestCallWithTransientRetry_5xxRetried(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var sleeps []time.Duration
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, mt int) (string, llm.Meta, error) {
		calls++
		if calls == 1 {
			return "", llm.Meta{}, &llm.StatusError{StatusCode: 503, Body: "unavailable"}
		}
		return "ok", llm.Meta{FinishReason: "stop", MaxTokens: mt}, nil
	}

	resp, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", fastTransientCfg(&sleeps))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if calls != 2 || resp != "ok" {
		t.Errorf("expected 5xx retried to success, calls=%d resp=%q", calls, resp)
	}
}

// (d) 4xx-other NOT retried: returns on attempt 1, error surfaced.
func TestCallWithTransientRetry_4xxNotRetried(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var sleeps []time.Duration
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, _ int) (string, llm.Meta, error) {
		calls++
		return "", llm.Meta{}, &llm.StatusError{StatusCode: 400, Body: "bad request"}
	}

	_, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", fastTransientCfg(&sleeps))
	if err == nil {
		t.Fatal("expected error for 400")
	}
	if calls != 1 {
		t.Fatalf("expected exactly 1 call (no retry), got %d", calls)
	}
	if !strings.Contains(err.Error(), "llm returned status 400") {
		t.Errorf("expected surfaced 400 error, got %q", err.Error())
	}
	if len(sleeps) != 0 {
		t.Errorf("expected no sleeps, got %v", sleeps)
	}
}

// (e) budget/ctx-cancel bounds: (1) fake sleep returns ctx.Err() -> loop stops;
// (2) tiny totalBudget -> loop can't fit another sleep.
func TestCallWithTransientRetry_CtxCancelBoundsLoop(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, _ int) (string, llm.Meta, error) {
		calls++
		return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, Body: "rate"}
	}
	cfg := transientRetryConfig{
		maxAttempts: 4, baseBackoff: time.Millisecond, maxBackoff: 10 * time.Millisecond,
		totalBudget: 60 * time.Second, rand: func() float64 { return 1.0 },
		sleep: func(_ context.Context, _ time.Duration) error { return context.Canceled },
	}
	_, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", cfg)
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Fatalf("sleep-cancel must stop loop after 1 call, got %d", calls)
	}
}

func TestCallWithTransientRetry_TinyBudgetBounds(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, _ int) (string, llm.Meta, error) {
		calls++
		return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, Body: "rate"}
	}
	cfg := transientRetryConfig{
		maxAttempts: 4, baseBackoff: time.Second, maxBackoff: 10 * time.Second,
		totalBudget: 50 * time.Millisecond, rand: func() float64 { return 1.0 },
		sleep: func(_ context.Context, _ time.Duration) error { return nil },
	}
	_, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", cfg)
	if err == nil {
		t.Fatal("expected exhaustion error")
	}
	if calls != 1 {
		t.Fatalf("tiny budget must not fit another sleep, got %d calls", calls)
	}
}

// (f) exhausted-429 error text: rate limited + attempt count + elapsed + status.
func TestCallWithTransientRetry_Exhausted429ErrorText(t *testing.T) {
	orig := callLLMMetaBudget
	t.Cleanup(func() { callLLMMetaBudget = orig })

	var sleeps []time.Duration
	callLLMMetaBudget = func(_ context.Context, _ string, _ int) (string, llm.Meta, error) {
		return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, Body: "rate"}
	}
	_, _, err := callWithTransientRetry(context.Background(), "p", 4000, "focal", fastTransientCfg(&sleeps))
	if err == nil {
		t.Fatal("expected exhaustion error")
	}
	msg := err.Error()
	for _, want := range []string{"rate limited", "exhausted 4 attempts", "over", "llm returned status 429"} {
		if !strings.Contains(msg, want) {
			t.Errorf("exhaustion error missing %q: %q", want, msg)
		}
	}
}

// (g) length-doubling composes with a first-budget transient 429: 429 then
// length at budget 4000, then success at doubled budget 8000.
func TestCallLLMWithRetry_TransientThenLengthThenDoubledSuccess(t *testing.T) {
	origSeam := callLLMMetaBudget
	origCfg := defaultTransientRetryConfig
	t.Cleanup(func() {
		callLLMMetaBudget = origSeam
		defaultTransientRetryConfig = origCfg
	})

	var sleeps []time.Duration
	defaultTransientRetryConfig = func() transientRetryConfig { return fastTransientCfg(&sleeps) }

	var budgets []int
	calls := 0
	callLLMMetaBudget = func(_ context.Context, _ string, mt int) (string, llm.Meta, error) {
		calls++
		budgets = append(budgets, mt)
		switch calls {
		case 1:
			return "", llm.Meta{}, &llm.StatusError{StatusCode: 429, Body: "rate"}
		case 2:
			return "trunc", llm.Meta{FinishReason: "length", MaxTokens: mt}, nil
		default:
			return "full", llm.Meta{FinishReason: "stop", MaxTokens: mt}, nil
		}
	}

	resp, meta, err := callLLMWithRetry(context.Background(), "p", 4000, 8000, "dialectic q1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != "full" || meta.FinishReason != "stop" {
		t.Errorf("expected doubled-budget success, got resp=%q meta=%+v", resp, meta)
	}
	if calls != 3 {
		t.Fatalf("expected 3 calls (429, length, success), got %d", calls)
	}
	if len(budgets) != 3 || budgets[0] != 4000 || budgets[1] != 4000 || budgets[2] != 8000 {
		t.Errorf("budget sequence = %v, want [4000 4000 8000]", budgets)
	}
}
