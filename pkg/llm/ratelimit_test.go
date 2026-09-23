package llm

import (
	"net/http"
	"testing"
)

func TestNewRateLimitSnapshot_HeadersPresent(t *testing.T) {
	h := http.Header{}
	h.Set("anthropic-ratelimit-requests-limit", "50")
	h.Set("anthropic-ratelimit-requests-remaining", "0")
	h.Set("anthropic-ratelimit-requests-reset", "2026-09-23T07:05:00Z")
	h.Set("anthropic-ratelimit-tokens-limit", "40000")
	h.Set("anthropic-ratelimit-tokens-remaining", "12")
	h.Set("anthropic-ratelimit-tokens-reset", "2026-09-23T07:06:00Z")
	h.Set("anthropic-ratelimit-input-tokens-limit", "30000")
	h.Set("anthropic-ratelimit-input-tokens-remaining", "5")
	h.Set("anthropic-ratelimit-input-tokens-reset", "2026-09-23T07:07:00Z")
	h.Set("anthropic-ratelimit-output-tokens-limit", "8000")
	h.Set("anthropic-ratelimit-output-tokens-remaining", "3")
	h.Set("anthropic-ratelimit-output-tokens-reset", "2026-09-23T07:08:00Z")
	h.Set("retry-after", "42")
	h.Set("request-id", "req_abc123")

	resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: h}
	s := newRateLimitSnapshot(resp, nil)

	checks := map[string]string{
		"RequestsLimit":         s.RequestsLimit,
		"RequestsRemaining":     s.RequestsRemaining,
		"RequestsReset":         s.RequestsReset,
		"TokensLimit":           s.TokensLimit,
		"TokensRemaining":       s.TokensRemaining,
		"TokensReset":           s.TokensReset,
		"InputTokensLimit":      s.InputTokensLimit,
		"InputTokensRemaining":  s.InputTokensRemaining,
		"InputTokensReset":      s.InputTokensReset,
		"OutputTokensLimit":     s.OutputTokensLimit,
		"OutputTokensRemaining": s.OutputTokensRemaining,
		"OutputTokensReset":     s.OutputTokensReset,
	}
	want := map[string]string{
		"RequestsLimit":         "50",
		"RequestsRemaining":     "0",
		"RequestsReset":         "2026-09-23T07:05:00Z",
		"TokensLimit":           "40000",
		"TokensRemaining":       "12",
		"TokensReset":           "2026-09-23T07:06:00Z",
		"InputTokensLimit":      "30000",
		"InputTokensRemaining":  "5",
		"InputTokensReset":      "2026-09-23T07:07:00Z",
		"OutputTokensLimit":     "8000",
		"OutputTokensRemaining": "3",
		"OutputTokensReset":     "2026-09-23T07:08:00Z",
	}
	for field, got := range checks {
		if got != want[field] {
			t.Errorf("%s = %q, want %q", field, got, want[field])
		}
	}
	if s.RetryAfter != "42" {
		t.Errorf("RetryAfter = %q, want 42", s.RetryAfter)
	}
	if s.HeadersAbsent {
		t.Error("HeadersAbsent = true, want false")
	}
	if s.RequestID != "req_abc123" {
		t.Errorf("RequestID = %q, want req_abc123", s.RequestID)
	}
	if s.StatusCode != http.StatusTooManyRequests {
		t.Errorf("StatusCode = %d, want 429", s.StatusCode)
	}
	if s.Timestamp == "" {
		t.Error("Timestamp empty")
	}
}

func TestNewStatusError_429_RecordsSnapshot(t *testing.T) {
	lastRateLimit.Store(nil) // reset for isolation

	h := http.Header{}
	h.Set("anthropic-ratelimit-requests-remaining", "0")
	h.Set("request-id", "req_xyz")
	resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: h}

	se := newStatusError(providerAnthropic, resp, []byte(`{"error":"rate_limited"}`))
	if se.RateLimit == nil {
		t.Fatal("se.RateLimit is nil, want populated for 429")
	}
	if se.RateLimit.RequestsRemaining != "0" {
		t.Errorf("RequestsRemaining = %q, want 0", se.RateLimit.RequestsRemaining)
	}
	if se.RateLimit.Provider != "anthropic" {
		t.Errorf("Provider = %q, want anthropic", se.RateLimit.Provider)
	}
	last := LastRateLimit()
	if last == nil {
		t.Fatal("LastRateLimit() nil after 429")
	}
	if last.RequestID != "req_xyz" {
		t.Errorf("LastRateLimit RequestID = %q, want req_xyz", last.RequestID)
	}
	if last.Provider != "anthropic" {
		t.Errorf("LastRateLimit Provider = %q, want anthropic", last.Provider)
	}
	// Error() prefix must be unchanged.
	if got := se.Error(); got[:len("llm returned status 429: ")] != "llm returned status 429: " {
		t.Errorf("Error() = %q, missing expected prefix", got)
	}
}

func TestNewRateLimitSnapshot_HeadersAbsent(t *testing.T) {
	lastRateLimit.Store(nil)

	resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: http.Header{}}
	se := newStatusError(providerAnthropic, resp, []byte(`{"request_id":"req_from_body","error":"overloaded"}`))
	if se.RateLimit == nil {
		t.Fatal("se.RateLimit nil for 429")
	}
	if !se.RateLimit.HeadersAbsent {
		t.Error("HeadersAbsent = false, want true when no headers present")
	}
	if se.RateLimit.StatusCode != http.StatusTooManyRequests {
		t.Errorf("StatusCode = %d, want 429", se.RateLimit.StatusCode)
	}
	if se.RateLimit.RequestID != "req_from_body" {
		t.Errorf("RequestID = %q, want req_from_body (parsed from body)", se.RateLimit.RequestID)
	}
	if LastRateLimit() == nil {
		t.Error("LastRateLimit() nil after headers-absent 429")
	}
}

func TestNewStatusError_Non429_NoRecord(t *testing.T) {
	lastRateLimit.Store(nil)

	resp := &http.Response{StatusCode: http.StatusInternalServerError, Header: http.Header{}}
	se := newStatusError(providerAnthropic, resp, []byte("boom"))
	if se.RateLimit != nil {
		t.Error("se.RateLimit populated for non-429")
	}
	if LastRateLimit() != nil {
		t.Error("LastRateLimit() should stay nil for non-429")
	}
}

// TestNewRateLimitSnapshot_RetryAfterOnly asserts a 429 carrying ONLY
// retry-after (no anthropic-ratelimit-*) still reports HeadersAbsent==true
// while capturing RetryAfter.
func TestNewRateLimitSnapshot_RetryAfterOnly(t *testing.T) {
	h := http.Header{}
	h.Set("retry-after", "30")
	resp := &http.Response{StatusCode: http.StatusTooManyRequests, Header: h}
	s := newRateLimitSnapshot(resp, nil)
	if !s.HeadersAbsent {
		t.Error("HeadersAbsent = false, want true when only retry-after present")
	}
	if s.RetryAfter != "30" {
		t.Errorf("RetryAfter = %q, want 30", s.RetryAfter)
	}
}
