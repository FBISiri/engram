package llm

import (
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// StatusError is a typed error carrying the upstream HTTP status, the raw
// Retry-After header (if any), and a short body excerpt. It is returned by both
// providers' non-200 branches so callers (package reflection) can classify
// transient failures (429/5xx) and honour Retry-After.
//
// The Error() message MUST keep the exact prefix "llm returned status %d: " —
// /health alerting, reflection error strings, and pkg/llm/anthropic_test.go all
// key off it.
type StatusError struct {
	StatusCode int
	RetryAfter string
	Body       string
	// RateLimit is populated for 429 responses; nil otherwise.
	RateLimit *RateLimitSnapshot
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("llm returned status %d: %s", e.StatusCode, e.Body)
}

// IsTransientStatus reports whether an HTTP status warrants a retry: 429 or any
// 5xx. All other statuses (including 4xx-other) are permanent.
func IsTransientStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}

// ParseRetryAfter parses a Retry-After header value. It supports the
// delta-seconds integer form and the HTTP-date form. Returns (0, false) when
// the header is empty, negative, or unparseable. Mirrors
// pkg/embedding/retry.go parseRetryAfter.
func ParseRetryAfter(h string, now time.Time) (time.Duration, bool) {
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
