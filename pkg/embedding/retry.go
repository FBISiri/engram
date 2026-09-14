package embedding

import (
	"context"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"time"
)

// retryConfig controls retry + exponential backoff + jitter behaviour shared by
// the OpenAI and Voyage embedders.
type retryConfig struct {
	maxAttempts int           // total attempts incl. first (e.g. 4)
	baseBackoff time.Duration // e.g. 500ms
	maxBackoff  time.Duration // e.g. 8s cap per-sleep
	totalBudget time.Duration // TOTAL wall bound across all attempts (R4)

	// sleep is injectable (R6); nil => real ctx-aware sleep. It must be ctx-aware
	// (select on ctx.Done()) and return ctx.Err() if ctx is done before d elapses.
	sleep func(ctx context.Context, d time.Duration) error
	// rand is injectable [0,1) for jitter; nil => package rand.
	rand func() float64
}

// defaultRetryConfig returns the production retry policy.
func defaultRetryConfig() retryConfig {
	return retryConfig{
		maxAttempts: 4,
		baseBackoff: 500 * time.Millisecond,
		maxBackoff:  8 * time.Second,
		totalBudget: 45 * time.Second,
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

// isTransientStatus reports whether an HTTP status warrants a retry (R1):
// 429 or any 5xx. All other statuses (including 4xx-other) are permanent.
func isTransientStatus(status int) bool {
	return status == http.StatusTooManyRequests || status >= 500
}

// parseRetryAfter parses a Retry-After header value (R2). It supports the
// delta-seconds integer form and the HTTP-date form. Returns (0, false) when
// the header is absent or unparseable.
func parseRetryAfter(h string, now time.Time) (time.Duration, bool) {
	if h == "" {
		return 0, false
	}
	if secs, err := strconv.Atoi(h); err == nil {
		if secs < 0 {
			return 0, false
		}
		return time.Duration(secs) * time.Second, true
	}
	if t, err := http.ParseTime(h); err == nil {
		d := t.Sub(now)
		if d < 0 {
			d = 0
		}
		return d, true
	}
	return 0, false
}

// backoffFor computes the full-jitter backoff for a given attempt index
// (0-based): random in [0, min(maxBackoff, base*2^attempt)].
func (rc retryConfig) backoffFor(attempt int) time.Duration {
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

// doRequestWithRetry executes newReq with retry + exponential backoff + jitter.
// It rebuilds the request each attempt (the body reader is consumed), executes
// it, reads the full body, and returns the final (statusCode, body, err).
//
// Retries occur only on transient failures (network errors, HTTP 429, HTTP 5xx)
// and are bounded by rc.maxAttempts, rc.totalBudget, and ctx (R1-R4).
func doRequestWithRetry(ctx context.Context, client *http.Client,
	newReq func(context.Context) (*http.Request, error), rc retryConfig) (int, []byte, error) {

	if rc.sleep == nil {
		rc.sleep = ctxSleep
	}
	// Bound the WHOLE operation (requests + sleeps) by totalBudget (R4).
	// WithTimeout takes the min of any earlier caller deadline and totalBudget,
	// so a shorter caller deadline is preserved, never extended.
	ctx, cancel := context.WithTimeout(ctx, rc.totalBudget)
	defer cancel()
	deadline := time.Now().Add(rc.totalBudget)

	var lastStatus int
	var lastBody []byte
	var lastErr error

	for attempt := 0; attempt < rc.maxAttempts; attempt++ {
		// Respect ctx before doing any work (R3).
		if err := ctx.Err(); err != nil {
			if lastErr != nil {
				return lastStatus, lastBody, lastErr
			}
			return 0, nil, err
		}

		req, err := newReq(ctx)
		if err != nil {
			return 0, nil, err
		}

		resp, err := client.Do(req)
		if err != nil {
			// Network error: transient. Record and maybe retry.
			lastStatus, lastBody, lastErr = 0, nil, err
		} else {
			body, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr != nil {
				lastStatus, lastBody, lastErr = 0, nil, readErr
			} else if !isTransientStatus(resp.StatusCode) {
				// Success or permanent error (incl. 4xx-other): return now (R1).
				return resp.StatusCode, body, nil
			} else {
				// Transient HTTP status: record and maybe retry.
				lastStatus, lastBody, lastErr = resp.StatusCode, body, nil
			}
		}

		// No more attempts left.
		if attempt == rc.maxAttempts-1 {
			break
		}

		// Compute the wait: honour Retry-After on 429, else computed backoff.
		wait := rc.backoffFor(attempt)
		if lastErr == nil && lastStatus == http.StatusTooManyRequests && resp != nil {
			if ra, ok := parseRetryAfter(resp.Header.Get("Retry-After"), time.Now()); ok {
				wait = ra
			}
		}

		// Bound the wait by remaining total budget (R2/R4).
		remaining := time.Until(deadline)
		if wait > remaining {
			// Can't fit another backoff+attempt within the budget: stop.
			break
		}

		if err := rc.sleep(ctx, wait); err != nil {
			// ctx cancelled/deadline during backoff (R3).
			return lastStatus, lastBody, err
		}
	}

	return lastStatus, lastBody, lastErr
}
