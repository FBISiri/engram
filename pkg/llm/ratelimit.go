package llm

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"
)

// RateLimitSnapshot captures Anthropic rate-limit response headers (plus a few
// diagnostics) at the moment a 429 was observed. Empty strings mean the
// corresponding header was absent. It is surfaced via /health.last_ratelimit so
// operators can see the most recent throttling event without scraping logs.
type RateLimitSnapshot struct {
	// Anthropic ratelimit headers (empty string = that header absent).
	RequestsLimit         string `json:"requests_limit,omitempty"`
	RequestsRemaining     string `json:"requests_remaining,omitempty"`
	RequestsReset         string `json:"requests_reset,omitempty"`
	TokensLimit           string `json:"tokens_limit,omitempty"`
	TokensRemaining       string `json:"tokens_remaining,omitempty"`
	TokensReset           string `json:"tokens_reset,omitempty"`
	InputTokensLimit      string `json:"input_tokens_limit,omitempty"`
	InputTokensRemaining  string `json:"input_tokens_remaining,omitempty"`
	InputTokensReset      string `json:"input_tokens_reset,omitempty"`
	OutputTokensLimit     string `json:"output_tokens_limit,omitempty"`
	OutputTokensRemaining string `json:"output_tokens_remaining,omitempty"`
	OutputTokensReset     string `json:"output_tokens_reset,omitempty"`
	RetryAfter            string `json:"retry_after,omitempty"`
	// Provenance
	Provider string `json:"provider,omitempty"`
	// Diagnostics
	HeadersAbsent bool   `json:"headers_absent"` // true when NONE of the ratelimit/retry-after headers were present
	StatusCode    int    `json:"status_code"`
	RequestID     string `json:"request_id,omitempty"`
	Timestamp     string `json:"timestamp"` // RFC3339 when the sample was captured
}

// newRateLimitSnapshot builds a RateLimitSnapshot from an HTTP response and its
// (already-read) body. Header lookups go through http.Header.Get, which
// canonicalises keys, so casing in the constants below is irrelevant.
func newRateLimitSnapshot(resp *http.Response, body []byte) RateLimitSnapshot {
	h := resp.Header
	s := RateLimitSnapshot{
		RequestsLimit:         h.Get("anthropic-ratelimit-requests-limit"),
		RequestsRemaining:     h.Get("anthropic-ratelimit-requests-remaining"),
		RequestsReset:         h.Get("anthropic-ratelimit-requests-reset"),
		TokensLimit:           h.Get("anthropic-ratelimit-tokens-limit"),
		TokensRemaining:       h.Get("anthropic-ratelimit-tokens-remaining"),
		TokensReset:           h.Get("anthropic-ratelimit-tokens-reset"),
		InputTokensLimit:      h.Get("anthropic-ratelimit-input-tokens-limit"),
		InputTokensRemaining:  h.Get("anthropic-ratelimit-input-tokens-remaining"),
		InputTokensReset:      h.Get("anthropic-ratelimit-input-tokens-reset"),
		OutputTokensLimit:     h.Get("anthropic-ratelimit-output-tokens-limit"),
		OutputTokensRemaining: h.Get("anthropic-ratelimit-output-tokens-remaining"),
		OutputTokensReset:     h.Get("anthropic-ratelimit-output-tokens-reset"),
		RetryAfter:            h.Get("retry-after"),
		StatusCode:            resp.StatusCode,
		Timestamp:             time.Now().UTC().Format(time.RFC3339),
	}

	// HeadersAbsent iff ALL 12 anthropic-ratelimit-* headers are empty.
	// retry-after is deliberately excluded: a retry-after-only 429 still means
	// no ratelimit headers were present.
	s.HeadersAbsent = s.RequestsLimit == "" && s.RequestsRemaining == "" && s.RequestsReset == "" &&
		s.TokensLimit == "" && s.TokensRemaining == "" && s.TokensReset == "" &&
		s.InputTokensLimit == "" && s.InputTokensRemaining == "" && s.InputTokensReset == "" &&
		s.OutputTokensLimit == "" && s.OutputTokensRemaining == "" && s.OutputTokensReset == ""

	// RequestID: prefer request-id header, then anthropic-request-id, then body.
	s.RequestID = h.Get("request-id")
	if s.RequestID == "" {
		s.RequestID = h.Get("anthropic-request-id")
	}
	if s.RequestID == "" {
		var parsed struct {
			RequestID string `json:"request_id"`
		}
		if err := json.Unmarshal(body, &parsed); err == nil {
			s.RequestID = parsed.RequestID
		}
	}

	return s
}

// lastRateLimit holds the most recent 429 snapshot; nil until the first 429.
var lastRateLimit atomic.Pointer[RateLimitSnapshot]

func recordRateLimit(s RateLimitSnapshot) { lastRateLimit.Store(&s) }

// LastRateLimit returns the most recently recorded 429 rate-limit snapshot, or
// nil if no 429 has been observed since process start.
func LastRateLimit() *RateLimitSnapshot { return lastRateLimit.Load() }

// newStatusError builds a *StatusError from a non-200 response and its body.
// For 429 responses it also captures and records a RateLimitSnapshot so
// /health.last_ratelimit always reflects a real throttling event.
func newStatusError(provider string, resp *http.Response, body []byte) *StatusError {
	se := &StatusError{
		StatusCode: resp.StatusCode,
		RetryAfter: resp.Header.Get("Retry-After"),
		Body:       bodyExcerpt(body),
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		snap := newRateLimitSnapshot(resp, body)
		snap.Provider = provider
		se.RateLimit = &snap
		recordRateLimit(snap)
	}
	return se
}
