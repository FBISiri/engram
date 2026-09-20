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
