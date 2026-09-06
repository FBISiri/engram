package server

import (
	"context"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/FBISiri/engram/pkg/reflection"
)

// waitForRunnerIdle polls the runner until no run is in flight, failing the
// test if it does not settle within the deadline. Used only for cleanup/await,
// not for correctness of the concurrency assertions.
func waitForRunnerIdle(t *testing.T, srv *Server) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if !srv.reflectionRunner.status().Running {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("reflection runner did not become idle within deadline")
}

// TestReflectionRun_ConcurrencySingleFlight fires N concurrent reflection_run
// calls and asserts exactly one launches (started=true) while the rest observe
// already_running=true. The runFn hook blocks on a channel so the single run
// stays in flight for the duration of the race.
func TestReflectionRun_ConcurrencySingleFlight(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	srv, _ := newTestServer()

	release := make(chan struct{})
	var runCount int32
	srv.reflectionRunner.runFn = func(context.Context) (*reflection.RunResult, error) {
		atomic.AddInt32(&runCount, 1)
		<-release
		return &reflection.RunResult{Triggered: true, InsightsCreated: 3, Duration: "1s", Mode: "v2-focal"}, nil
	}

	const n = 8
	results := make([]string, n)
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			defer wg.Done()
			res, err := callTool(srv, "reflection_run", map[string]any{})
			if err != nil {
				results[i] = `{"error":"` + err.Error() + `"}`
				return
			}
			results[i] = extractText(res)
		}(i)
	}
	wg.Wait()

	started, alreadyRunning := 0, 0
	for _, txt := range results {
		var m map[string]any
		if err := json.Unmarshal([]byte(txt), &m); err != nil {
			t.Fatalf("bad response json %q: %v", txt, err)
		}
		if b, _ := m["started"].(bool); b {
			started++
		}
		if b, _ := m["already_running"].(bool); b {
			alreadyRunning++
		}
	}
	if started != 1 {
		t.Fatalf("expected exactly 1 started=true, got %d (responses: %v)", started, results)
	}
	if alreadyRunning != n-1 {
		t.Fatalf("expected %d already_running=true, got %d (responses: %v)", n-1, alreadyRunning, results)
	}

	// Release the single in-flight run and await completion.
	close(release)
	waitForRunnerIdle(t, srv)

	if got := atomic.LoadInt32(&runCount); got != 1 {
		t.Fatalf("expected runFn to execute exactly once, got %d", got)
	}
}

// TestReflectionRun_CtxDecoupled verifies the async run uses a Background-derived
// context, not the (cancelled) request context.
func TestReflectionRun_CtxDecoupled(t *testing.T) {
	t.Setenv("ENGRAM_STATE_DIR", t.TempDir())
	srv, _ := newTestServer()

	proceed := make(chan struct{})
	done := make(chan struct{})
	var gotErr error
	srv.reflectionRunner.runFn = func(ctx context.Context) (*reflection.RunResult, error) {
		<-proceed // block until the request ctx has been cancelled
		gotErr = ctx.Err()
		close(done)
		return &reflection.RunResult{Triggered: true, InsightsCreated: 1, Duration: "2s", Mode: "v1-flat"}, nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{Name: "reflection_run", Arguments: map[string]any{}},
	}
	res, err := srv.handleReflectionRun(ctx, req)
	if err != nil {
		t.Fatalf("handleReflectionRun: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(extractText(res)), &m); err != nil {
		t.Fatalf("bad response: %v", err)
	}
	if b, _ := m["started"].(bool); !b {
		t.Fatalf("expected started=true, got %v", m)
	}

	// Cancel the request context, THEN let runFn observe its own ctx.
	cancel()
	close(proceed)
	<-done

	if gotErr != nil {
		t.Fatalf("expected background-derived ctx (ctx.Err()==nil), got %v", gotErr)
	}

	waitForRunnerIdle(t, srv)
	st := srv.reflectionRunner.status()
	if st.Running {
		t.Fatal("expected running=false after completion")
	}
	if st.InsightsCreated != 1 {
		t.Fatalf("expected insights_created=1 in status, got %d", st.InsightsCreated)
	}
	if !st.Triggered {
		t.Fatal("expected triggered=true in status")
	}
}

// TestReflectionStatus_BeforeAnyRun asserts the zero-value status.
func TestReflectionStatus_BeforeAnyRun(t *testing.T) {
	srv, _ := newTestServer()
	res, err := callTool(srv, "reflection_status", map[string]any{})
	if err != nil {
		t.Fatalf("reflection_status: %v", err)
	}
	var st reflectionStatus
	if err := json.Unmarshal([]byte(extractText(res)), &st); err != nil {
		t.Fatalf("bad status json: %v", err)
	}
	if st.Running {
		t.Error("expected running=false before any run")
	}
	if st.RunID != "" {
		t.Errorf("expected empty run_id before any run, got %q", st.RunID)
	}
	if st.LastRunID != "" || st.InsightsCreated != 0 || st.LastDuration != "" {
		t.Errorf("expected zero-value status before any run, got %+v", st)
	}
}
