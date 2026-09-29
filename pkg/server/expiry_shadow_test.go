package server

import (
	"bytes"
	"context"
	"io"
	"log"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
)

// TestExpiryTickHeartbeat (R4a): when DeleteExpired returns 0, runExpiryTick
// emits exactly one heartbeat line so "tick ran, deleted 0" is distinguishable
// from "tick never ran".
func TestExpiryTickHeartbeat(t *testing.T) {
	future := float64(time.Now().Add(time.Hour).Unix())
	store := &expiryObsStore{mems: []memory.Memory{
		{ID: "a", Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: future},
	}}

	// runExpiryTick writes to os.Stderr directly (like the rest of expiry.go),
	// not via package log, so capture stderr through a pipe.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStderr := os.Stderr
	os.Stderr = w

	runExpiryTick(context.Background(), store, nil, time.Now())

	os.Stderr = oldStderr
	_ = w.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	out := string(b)
	if !strings.Contains(out, "[expiry] tick ok") || !strings.Contains(out, "deleted=0") {
		t.Fatalf("heartbeat line missing; got %q", out)
	}
	if strings.Count(out, "[expiry] tick ok") != 1 {
		t.Fatalf("want exactly one heartbeat line; got %q", out)
	}
	if !strings.Contains(out, "scanned=0") {
		t.Fatalf("want scanned=0 (no expired mems); got %q", out)
	}
}

// TestFindExpiryCandidatesExemptByRule (R4b): an evaporation-deprecated memory
// inside the observation window is counted as an exemption (rule
// observation_window) in both the summary log line and the shadow counter,
// while a memory past the window is returned as a candidate.
func TestFindExpiryCandidatesExemptByRule(t *testing.T) {
	recent := evapDeprecated("recent", memory.TypeEvent, 3, 100, 1) // inside window → exempt
	old := evapDeprecated("old", memory.TypeEvent, 3, 100, 31)      // past window → candidate
	store := &expiryObsStore{mems: []memory.Memory{*recent, *old}}

	srv := &Server{store: store, evapCfg: memory.DefaultEvaporationConfig()}
	srv.SetMetrics(metrics.New(nil, nil))
	h := &HTTPServer{srv: srv}

	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer log.SetOutput(os.Stderr)

	req := httptest.NewRequest("GET", "/memories/expiry-candidates", nil)
	candidates, err := h.findExpiryCandidates(req)
	if err != nil {
		t.Fatal(err)
	}
	if len(candidates) != 1 || candidates[0].ID != "old" {
		t.Fatalf("candidates=%v, want exactly [old]", candidates)
	}

	if got := counterValue(t, srv.metrics.ShadowExemptByRule.WithLabelValues("observation_window")); got != 1 {
		t.Fatalf("ShadowExemptByRule{observation_window}=%v, want 1", got)
	}

	out := buf.String()
	if !strings.Contains(out, "[expiry-policy] candidates=1 exempted=1 by_rule=map[observation_window:1]") {
		t.Fatalf("summary log line wrong; got %q", out)
	}
	if strings.Count(out, "[expiry-policy]") != 1 {
		t.Fatalf("want exactly one summary line; got %q", out)
	}
}

// TestExpiryCandidateDecisionUnchanged (R4c): expiryCandidateDecision's boolean
// is identical to isExpiryCandidate across representative fixtures, proving the
// reason-carrying refactor is behaviour-preserving (R3).
func TestExpiryCandidateDecisionUnchanged(t *testing.T) {
	cfg := memory.DefaultEvaporationConfig()
	now := time.Now()
	nowUnix := float64(now.Unix())

	cases := []struct {
		name string
		m    *memory.Memory
		want bool
	}{
		{"identity-never", &memory.Memory{Type: memory.TypeIdentity, Importance: 1, CreatedAt: nowUnix - 400*86400}, false},
		{"high-importance-event", &memory.Memory{Type: memory.TypeEvent, Importance: 9, CreatedAt: nowUnix - 400*86400}, false},
		{"old-low-event", &memory.Memory{Type: memory.TypeEvent, Importance: 2, CreatedAt: nowUnix - 60*86400}, true},
		{"young-low-event", &memory.Memory{Type: memory.TypeEvent, Importance: 2, CreatedAt: nowUnix - 5*86400}, false},
		{"evap-inside-window", evapDeprecated("r", memory.TypeEvent, 3, 100, 1), false},
		{"evap-past-window", evapDeprecated("o", memory.TypeEvent, 3, 100, 31), true},
		{"evap-importance-guard", evapDeprecated("g", memory.TypeEvent, 9, 100, 60), false},
	}
	for _, c := range cases {
		want := isExpiryCandidate(c.m, cfg, now)
		got, _ := expiryCandidateDecision(c.m, cfg, now)
		if got != want {
			t.Errorf("%s: decision bool=%v != isExpiryCandidate=%v", c.name, got, want)
		}
		if want != c.want {
			t.Errorf("%s: candidate=%v, want %v", c.name, want, c.want)
		}
	}
}
