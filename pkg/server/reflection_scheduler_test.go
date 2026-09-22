package server

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/reflection"
)

// TestReflectionRunnerRecoversFromPanic verifies the single-flight runner
// resets running=false even when the run function panics, so a panic cannot
// wedge the runner permanently.
func TestReflectionRunnerRecoversFromPanic(t *testing.T) {
	r := &reflectionRunner{
		runFn: func(_ context.Context) (*reflection.RunResult, error) {
			panic("boom")
		},
	}

	started, _, _ := r.start(nil, nil)
	if !started {
		t.Fatalf("expected start to launch a run, got started=false")
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if !r.status().Running {
			return // recovered: single-flight released
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("runner still running after panic: single-flight wedged")
}

// TestReflectionRunnerReleasesOnError verifies the runner also releases the
// single-flight guard on a normal (non-panic) error return.
func TestReflectionRunnerReleasesOnError(t *testing.T) {
	r := &reflectionRunner{
		runFn: func(_ context.Context) (*reflection.RunResult, error) {
			return nil, errors.New("nope")
		},
	}

	started, _, _ := r.start(nil, nil)
	if !started {
		t.Fatalf("expected start to launch a run, got started=false")
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if !r.status().Running {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("runner still running after error return")
}

// ── Quiet-window gate (Task D) ──────────────────────────────────────────────

func TestInReflectionWindow_MidnightWrap(t *testing.T) {
	// Window "22:00-01:00" wraps midnight: start=1320, end=60.
	start, end := 1320, 60
	at := func(h, m int) time.Time { return time.Date(2026, 1, 2, h, m, 0, 0, time.UTC) }

	cases := []struct {
		h, m int
		want bool
		name string
	}{
		{23, 30, true, "23:30 in"},
		{0, 30, true, "00:30 in"},
		{12, 0, false, "12:00 out"},
		{21, 59, false, "21:59 out"},
		{22, 0, true, "22:00 in (start inclusive)"},
		{1, 0, false, "01:00 out (end exclusive)"},
	}
	for _, c := range cases {
		if got := inReflectionWindow(at(c.h, c.m), start, end); got != c.want {
			t.Errorf("%s: inReflectionWindow(%02d:%02d)=%v, want %v", c.name, c.h, c.m, got, c.want)
		}
	}
}

func TestParseReflectionWindow(t *testing.T) {
	start, end, err := parseReflectionWindow("22:00-01:00")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if start != 1320 || end != 60 {
		t.Errorf("parseReflectionWindow(22:00-01:00)=(%d,%d), want (1320,60)", start, end)
	}
	if _, _, err := parseReflectionWindow("nonsense"); err == nil {
		t.Error("expected error for malformed 'nonsense'")
	}
	if _, _, err := parseReflectionWindow("25:00-01:00"); err == nil {
		t.Error("expected error for out-of-range '25:00-01:00'")
	}
}
