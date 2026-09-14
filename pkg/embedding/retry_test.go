package embedding

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// fastRetryConfig returns a retry policy that records sleeps instead of really
// sleeping, so tests are fast and deterministic (R6).
func fastRetryConfig(sleeps *[]time.Duration) retryConfig {
	rc := defaultRetryConfig()
	rc.baseBackoff = 10 * time.Millisecond
	rc.maxBackoff = 100 * time.Millisecond
	rc.rand = func() float64 { return 1.0 } // deterministic: full backoff
	rc.sleep = func(ctx context.Context, d time.Duration) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		*sleeps = append(*sleeps, d)
		return nil
	}
	return rc
}

func newReqFor(t *testing.T, url string) func(context.Context) (*http.Request, error) {
	t.Helper()
	return func(ctx context.Context) (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodPost, url, nil)
	}
}

// (a) transient-then-success: 503 then 200 → success, N attempts observed.
func TestRetryTransientThenSuccess(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if attempts.Load() < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	status, body, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if status != http.StatusOK || string(body) != "ok" {
		t.Fatalf("got status=%d body=%q", status, body)
	}
	if int(attempts.Load()) != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts.Load())
	}
	if len(sleeps) != 1 {
		t.Fatalf("expected 1 backoff sleep, got %d (%v)", len(sleeps), sleeps)
	}
}

// (b) permanent-4xx-no-retry: 400 → exactly ONE attempt, no retry.
func TestRetryPermanent4xxNoRetry(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`bad`))
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	status, body, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if status != http.StatusBadRequest || string(body) != "bad" {
		t.Fatalf("got status=%d body=%q", status, body)
	}
	if int(attempts.Load()) != 1 {
		t.Fatalf("expected exactly 1 attempt (no retry on 4xx), got %d", attempts.Load())
	}
	if len(sleeps) != 0 {
		t.Fatalf("expected no sleeps, got %v", sleeps)
	}
}

// (c) budget/attempts exhaustion: persistent 503 → returns error/last status,
// bounded attempts.
func TestRetryExhaustsAttempts(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`down`))
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	status, body, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	if err != nil {
		t.Fatalf("unexpected transport err: %v", err)
	}
	if status != http.StatusServiceUnavailable || string(body) != "down" {
		t.Fatalf("expected last 503 result, got status=%d body=%q", status, body)
	}
	if int(attempts.Load()) != rc.maxAttempts {
		t.Fatalf("expected %d attempts, got %d", rc.maxAttempts, attempts.Load())
	}
	if len(sleeps) != rc.maxAttempts-1 {
		t.Fatalf("expected %d sleeps, got %d", rc.maxAttempts-1, len(sleeps))
	}
}

// (d) ctx-cancellation-mid-backoff: cancel during the backoff → returns ctx.Err().
func TestRetryCtxCancelledMidBackoff(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	rc := defaultRetryConfig()
	rc.baseBackoff = 10 * time.Millisecond
	rc.rand = func() float64 { return 1.0 }
	// sleep that cancels ctx then behaves ctx-aware (returns ctx.Err()).
	rc.sleep = func(ctx context.Context, d time.Duration) error {
		cancel()
		return ctx.Err()
	}

	_, _, err := doRequestWithRetry(ctx, srv.Client(), newReqFor(t, srv.URL), rc)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if int(attempts.Load()) != 1 {
		t.Fatalf("expected 1 attempt before cancel, got %d", attempts.Load())
	}
}

// R2: Retry-After (delta-seconds) is honoured on 429.
func TestRetryAfterHonoured(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if attempts.Load() < 2 {
			w.Header().Set("Retry-After", "2")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`ok`))
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	status, _, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("expected 200, got %d", status)
	}
	if len(sleeps) != 1 {
		t.Fatalf("expected 1 sleep, got %v", sleeps)
	}
	if sleeps[0] != 2*time.Second {
		t.Fatalf("expected Retry-After 2s honoured, got %v", sleeps[0])
	}
}

// R2: Retry-After HTTP-date form is honoured.
func TestParseRetryAfterHTTPDate(t *testing.T) {
	now := time.Date(2026, 9, 14, 0, 0, 0, 0, time.UTC)
	future := now.Add(5 * time.Second).UTC().Format(http.TimeFormat)
	d, ok := parseRetryAfter(future, now)
	if !ok {
		t.Fatal("expected HTTP-date Retry-After to parse")
	}
	// Truncated to whole seconds by the header format; allow ~1s slack.
	if d < 4*time.Second || d > 6*time.Second {
		t.Fatalf("expected ~5s, got %v", d)
	}
}

// R1: network errors are transient and retried, then surfaced.
func TestRetryNetworkErrorRetried(t *testing.T) {
	var attempts atomic.Int64
	newReq := func(ctx context.Context) (*http.Request, error) {
		attempts.Add(1)
		return http.NewRequestWithContext(ctx, http.MethodPost, "http://127.0.0.1:0/nope", nil)
	}
	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	_, _, err := doRequestWithRetry(context.Background(), &http.Client{Timeout: time.Second}, newReq, rc)
	if err == nil {
		t.Fatal("expected a network error")
	}
	if int(attempts.Load()) != rc.maxAttempts {
		t.Fatalf("expected %d attempts, got %d", rc.maxAttempts, attempts.Load())
	}
}

// R4: totalBudget stops retries before exceeding the wall bound.
func TestRetryTotalBudgetBounds(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	rc.maxAttempts = 100
	rc.baseBackoff = 40 * time.Millisecond
	rc.maxBackoff = time.Hour
	rc.totalBudget = 50 * time.Millisecond // only the first backoff can fit
	_, _, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	if err != nil {
		t.Fatalf("unexpected transport err: %v", err)
	}
	// budget only fits one 40ms sleep; the next (80ms) exceeds remaining budget.
	if len(sleeps) > 2 {
		t.Fatalf("expected budget to bound sleeps, got %d (%v)", len(sleeps), sleeps)
	}
	if int(attempts.Load()) >= rc.maxAttempts {
		t.Fatalf("expected budget to stop well before maxAttempts, got %d", attempts.Load())
	}
}

// R4: totalBudget bounds the WHOLE operation even when the server is slow to
// RESPOND (not just slow via backoff). A tiny budget must abort with ctx.Err()
// within budget and after a bounded number of attempts.
func TestRetryTotalBudgetBoundsSlowResponse(t *testing.T) {
	release := make(chan struct{})
	defer close(release)
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		// Block until the request ctx is cancelled (budget elapses) or test ends.
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()

	var sleeps []time.Duration
	rc := fastRetryConfig(&sleeps)
	rc.maxAttempts = 100
	rc.totalBudget = 80 * time.Millisecond // whole operation must finish ~here

	start := time.Now()
	_, _, err := doRequestWithRetry(context.Background(), srv.Client(), newReqFor(t, srv.URL), rc)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a deadline/ctx error from budget exhaustion")
	}
	if elapsed > 2*time.Second {
		t.Fatalf("operation exceeded budget: took %v", elapsed)
	}
	if int(attempts.Load()) >= rc.maxAttempts {
		t.Fatalf("expected budget to bound attempts, got %d", attempts.Load())
	}
}

// Sanity: OpenAI.EmbedBatch drives the retry path end-to-end via a fake server.
func TestOpenAIEmbedBatchRetries(t *testing.T) {
	var attempts atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts.Add(1)
		if attempts.Load() < 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`)
	}))
	defer srv.Close()

	o := NewOpenAI(OpenAIConfig{APIKey: "k", BaseURL: srv.URL, Dimension: 3})
	var sleeps []time.Duration
	o.retry = fastRetryConfig(&sleeps)
	vecs, err := o.EmbedBatch(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != 3 {
		t.Fatalf("unexpected vecs: %v", vecs)
	}
	if int(attempts.Load()) != 2 {
		t.Fatalf("expected 2 attempts, got %d", attempts.Load())
	}
}
