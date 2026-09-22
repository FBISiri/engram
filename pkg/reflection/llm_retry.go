package reflection

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net"
	"reflect"
	"time"

	"github.com/FBISiri/engram/pkg/llm"
)

// ErrTruncatedResponse marks an LLM response that was cut off because the token
// budget was exhausted (finish_reason="length"), as opposed to a malformed
// payload. Callers wrap it so the failure is reported as a BUDGET problem, never
// a FORMAT / JSON parse problem.
var ErrTruncatedResponse = errors.New("llm response truncated: token budget exhausted")

// truncationError returns a wrapped ErrTruncatedResponse with diagnostics when
// the response was cut off by the budget; nil otherwise. The message must NOT
// mention "JSON"/"parse"/"format" — this is a budget signal, not a format one.
func truncationError(stage string, meta llm.Meta) error {
	if meta.FinishReason != "length" {
		return nil
	}
	return fmt.Errorf("%s: %w (finish_reason=%q max_tokens=%d completion_tokens=%d reasoning_tokens=%d raw_len=%d)",
		stage, ErrTruncatedResponse, meta.FinishReason, meta.MaxTokens, meta.CompletionTokens, meta.ReasoningTokens, meta.RawLen)
}

// callLLMMetaBudget is the test seam for the budgeted retry path. It honors the
// callLLMFunc override property: when the seam still points at the real client,
// real metadata (finish_reason, usage) is captured via llm.CallWithBudget; when
// a test overrides callLLMFunc, best-effort meta is returned.
var callLLMMetaBudget = func(ctx context.Context, prompt string, maxTokens int) (string, llm.Meta, error) {
	if reflect.ValueOf(callLLMFunc).Pointer() == reflect.ValueOf(llm.Call).Pointer() {
		return llm.CallWithBudget(ctx, prompt, maxTokens)
	}
	content, err := callLLMFunc(ctx, prompt)
	return content, llm.Meta{RawLen: len(content), MaxTokens: maxTokens,
		PromptTokens: -1, CompletionTokens: -1, TotalTokens: -1, ReasoningTokens: -1}, err
}

// callLLMWithRetry runs prompt at budget; on finish_reason=="length" it retries
// ONCE at min(2*budget, ceiling). At most one retry. Logs the retry (attempt,
// old/new budget, outcome). Each budget level goes through callWithTransientRetry
// so a transient upstream failure (429/5xx/network) is retried with backoff
// BEFORE the finish_reason=="length" budget-doubling dimension composes on top.
func callLLMWithRetry(ctx context.Context, prompt string, budget, ceiling int, stage string) (string, llm.Meta, error) {
	resp, meta, err := callWithTransientRetry(ctx, prompt, budget, stage, defaultTransientRetryConfig())
	if err != nil || meta.FinishReason != "length" {
		return resp, meta, err
	}
	newBudget := budget * 2
	if ceiling > 0 && newBudget > ceiling {
		newBudget = ceiling
	}
	if newBudget <= budget {
		return resp, meta, err
	}
	log.Printf("[reflection] %s truncated at max_tokens=%d (completion_tokens=%d reasoning_tokens=%d) — retry attempt 2 old_budget=%d new_budget=%d",
		stage, budget, meta.CompletionTokens, meta.ReasoningTokens, budget, newBudget)
	resp2, meta2, err2 := callWithTransientRetry(ctx, prompt, newBudget, stage, defaultTransientRetryConfig())
	if err2 != nil {
		log.Printf("[reflection] %s retry attempt 2 error: %v", stage, err2)
		return resp2, meta2, err2
	}
	log.Printf("[reflection] %s retry attempt 2 outcome: finish_reason=%q raw_len=%d max_tokens=%d completion_tokens=%d reasoning_tokens=%d",
		stage, meta2.FinishReason, meta2.RawLen, meta2.MaxTokens, meta2.CompletionTokens, meta2.ReasoningTokens)
	return resp2, meta2, err2
}

// transientRetryConfig controls transient-failure retry + exponential backoff +
// jitter around a single budgeted LLM call. It mirrors pkg/embedding/retry.go
// but wraps a call-func (the callLLMMetaBudget seam) instead of an *http.Request.
type transientRetryConfig struct {
	maxAttempts int           // total attempts incl. first
	baseBackoff time.Duration // first-retry backoff base
	maxBackoff  time.Duration // per-sleep cap
	totalBudget time.Duration // TOTAL wall bound across all attempts

	// sleep is injectable; nil => real ctx-aware sleep. It must be ctx-aware and
	// return ctx.Err() if ctx is done before d elapses.
	sleep func(ctx context.Context, d time.Duration) error
	// rand is injectable [0,1) for jitter; nil => package rand.
	rand func() float64
}

// defaultTransientRetryConfig returns the production transient-retry policy. It
// is a package var so tests can inject a fast (fake-sleep) config; production
// always uses the numbers below.
//
// JUSTIFICATION: the whole async reflection run has a 30-minute ctx budget
// (pkg/server/reflection_async.go:111) and min-interval is measured in hours; a
// run makes ~1 focal + N per-question dialectic LLM calls. A 90s per-call
// transient budget with 4 attempts keeps even a ~6-call run (≈9min worst case)
// well inside the 30min run budget, so a run can never hang behind backoff,
// while still riding out a short 429 burst. Each call gets its own fresh 90s
// budget; the caller ctx bounds the total via min-semantics (WithTimeout).
var defaultTransientRetryConfig = func() transientRetryConfig {
	return transientRetryConfig{
		maxAttempts: 4,
		baseBackoff: 1 * time.Second,
		maxBackoff:  20 * time.Second,
		totalBudget: 90 * time.Second,
	}
}

// ctxSleep sleeps for d or until ctx is done, whichever comes first.
func ctxSleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// backoffFor computes the full-jitter backoff for a given attempt index
// (0-based): random in [0, min(maxBackoff, base*2^attempt)].
func (rc transientRetryConfig) backoffFor(attempt int) time.Duration {
	exp := rc.baseBackoff << attempt
	if exp <= 0 || exp > rc.maxBackoff {
		exp = rc.maxBackoff
	}
	r := rc.rand
	if r == nil {
		r = rand.Float64
	}
	return time.Duration(r() * float64(exp))
}

// callWithTransientRetry wraps the callLLMMetaBudget seam with retry +
// exponential backoff + jitter on transient failures (HTTP 429, HTTP 5xx,
// network errors), honouring Retry-After. It is bounded by cfg.maxAttempts,
// cfg.totalBudget, and the caller ctx (min-semantics via WithTimeout — never
// extends an earlier deadline). finish_reason=="length" is NOT transient: a
// successful (err==nil) call is returned immediately regardless of finish_reason
// so the budget-doubling dimension in callLLMWithRetry composes on top.
func callWithTransientRetry(ctx context.Context, prompt string, budget int, stage string, cfg transientRetryConfig) (string, llm.Meta, error) {
	if cfg.sleep == nil {
		cfg.sleep = ctxSleep
	}
	ctx, cancel := context.WithTimeout(ctx, cfg.totalBudget)
	defer cancel()
	deadline := time.Now().Add(cfg.totalBudget)
	start := time.Now()

	var (
		lastResp   string
		lastMeta   llm.Meta
		lastErr    error
		lastStatus int
		retryAfter string
		attempts   int
	)

	for attempt := 0; attempt < cfg.maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return lastResp, lastMeta, lastErr
			}
			return "", llm.Meta{}, err
		}

		resp, meta, err := callLLMMetaBudget(ctx, prompt, budget)
		attempts = attempt + 1
		if err == nil {
			// Success — finish_reason=="length" is the OTHER retry dimension and
			// MUST NOT be treated as transient here.
			return resp, meta, nil
		}

		// Classify the failure.
		transient := false
		lastStatus, retryAfter = 0, ""
		var se *llm.StatusError
		if errors.As(err, &se) {
			transient = llm.IsTransientStatus(se.StatusCode)
			lastStatus = se.StatusCode
			retryAfter = se.RetryAfter
		} else {
			var ne net.Error
			if errors.As(err, &ne) {
				transient = true
			}
		}
		if !transient {
			// Permanent (4xx-other, decode errors, "no choices", etc.).
			return resp, meta, err
		}
		lastResp, lastMeta, lastErr = resp, meta, err

		if attempt == cfg.maxAttempts-1 {
			break
		}

		wait := cfg.backoffFor(attempt)
		if lastStatus == 429 {
			if ra, ok := llm.ParseRetryAfter(retryAfter, time.Now()); ok {
				wait = ra
			}
		}
		if wait > time.Until(deadline) {
			// Can't fit another backoff+attempt within the budget: stop.
			break
		}
		if err := cfg.sleep(ctx, wait); err != nil {
			return lastResp, lastMeta, lastErr
		}
	}

	elapsed := time.Since(start)
	if lastStatus == 429 {
		return lastResp, lastMeta, fmt.Errorf("llm rate limited (429): exhausted %d attempts over %s: %w",
			attempts, elapsed.Round(time.Millisecond), lastErr)
	}
	return lastResp, lastMeta, fmt.Errorf("llm transient failure: exhausted %d attempts over %s: %w",
		attempts, elapsed.Round(time.Millisecond), lastErr)
}
