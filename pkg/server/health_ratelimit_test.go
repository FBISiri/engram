package server

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/llm"
)

// TestHealthResponse_LastRatelimitMarshals asserts the /health wiring for
// last_ratelimit: a populated snapshot marshals into JSON containing the
// substring "ratelimit" (satisfying the curl grep -c ratelimit acceptance),
// and an absent snapshot is omitted.
func TestHealthResponse_LastRatelimitMarshals(t *testing.T) {
	snap := &llm.RateLimitSnapshot{
		RequestsRemaining: "0",
		StatusCode:        429,
		RequestID:         "req_health",
		Timestamp:         "2026-09-23T07:05:00Z",
	}
	b, err := json.Marshal(healthResponse{Status: "ok", LastRatelimit: snap})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	if !strings.Contains(s, "last_ratelimit") {
		t.Errorf("missing last_ratelimit key: %s", s)
	}
	if strings.Count(s, "ratelimit") < 1 {
		t.Errorf("expected >=1 occurrence of 'ratelimit': %s", s)
	}

	// Absent => omitted.
	b2, err := json.Marshal(healthResponse{Status: "ok"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(b2), "last_ratelimit") {
		t.Errorf("last_ratelimit should be omitted when nil: %s", b2)
	}
}

// TestHealthResponse_HeadersAbsentSnapshotStillAppears asserts that a
// headers-absent finding is a first-class object (not skipped).
func TestHealthResponse_HeadersAbsentSnapshotStillAppears(t *testing.T) {
	snap := &llm.RateLimitSnapshot{HeadersAbsent: true, StatusCode: 429, Timestamp: "2026-09-23T09:31:00Z"}
	b, err := json.Marshal(healthResponse{Status: "ok", LastRatelimit: snap})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(b), "headers_absent") {
		t.Errorf("headers_absent finding missing: %s", b)
	}
}
