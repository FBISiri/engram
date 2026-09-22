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

// TestReflectionRun_LastErrorSummary asserts that status().LastError summarizes
// a result's Errors slice when runFn returns (result, nil err).
func TestReflectionRun_LastErrorSummary(t *testing.T) {
	cases := []struct {
		name    string
		errs    []string
		wantErr string
	}{
		{"no errors", nil, ""},
		{"single", []string{"boom"}, "boom"},
		{"multi", []string{"first", "second", "third"}, "first (+2 more)"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := newTestServer()
			srv.reflectionRunner.runFn = func(context.Context) (*reflection.RunResult, error) {
				return &reflection.RunResult{Triggered: true, Duration: "1s", Mode: "v2-focal", Errors: tc.errs}, nil
			}
			started, _, _ := srv.reflectionRunner.start(
				func(context.Context) (*reflection.RunResult, error) { return nil, nil }, nil)
			if !started {
				t.Fatal("expected start to launch")
			}
			waitForRunnerIdle(t, srv)
			if got := srv.reflectionRunner.status().LastError; got != tc.wantErr {
				t.Errorf("LastError = %q, want %q", got, tc.wantErr)
			}
		})
	}
}

// TestStatusFunnelFieldsAlwaysPresent verifies the reflection funnel is
// serialized unconditionally on /health .reflection_runner: all 10 funnel keys
// are present, the 9 int fields emit literal 0 (never omitted) when the last
// run produced nothing, and per_question_counts serializes as [] (never null).
func TestStatusFunnelFieldsAlwaysPresent(t *testing.T) {
	funnelKeys := []string{
		"input_count", "evidence_count", "per_question_counts",
		"dialectic_ok", "dialectic_dropped_no_evidence", "dialectic_dropped_low_conf",
		"insights_written", "insights_skipped", "insights_write_failed", "drafts_written",
	}
	intKeys := []string{
		"input_count", "evidence_count", "dialectic_ok",
		"dialectic_dropped_no_evidence", "dialectic_dropped_low_conf",
		"insights_written", "insights_skipped", "insights_write_failed", "drafts_written",
	}

	assertKeysPresent := func(t *testing.T, raw map[string]json.RawMessage) {
		t.Helper()
		for _, k := range funnelKeys {
			if _, ok := raw[k]; !ok {
				t.Errorf("funnel key %q missing from status JSON", k)
			}
		}
	}

	// Negative case: a run that produced NOTHING — all counters 0, nil slice.
	t.Run("empty_run_emits_zeros", func(t *testing.T) {
		r := &reflectionRunner{lastResult: &reflection.RunResult{}}
		data, err := json.Marshal(r.status())
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		assertKeysPresent(t, raw)
		for _, k := range intKeys {
			if string(raw[k]) != "0" {
				t.Errorf("%s = %s, want 0", k, raw[k])
			}
		}
		if string(raw["per_question_counts"]) != "[]" {
			t.Errorf("per_question_counts = %s, want []", raw["per_question_counts"])
		}
	})

	// nil lastResult: same guarantees (all zero, [] not null).
	t.Run("nil_last_result_emits_zeros", func(t *testing.T) {
		r := &reflectionRunner{}
		data, err := json.Marshal(r.status())
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		assertKeysPresent(t, raw)
		for _, k := range intKeys {
			if string(raw[k]) != "0" {
				t.Errorf("%s = %s, want 0", k, raw[k])
			}
		}
		if string(raw["per_question_counts"]) != "[]" {
			t.Errorf("per_question_counts = %s, want []", raw["per_question_counts"])
		}
	})

	// Positive case: candidates came in and per-question counts surface.
	t.Run("populated_run_surfaces_values", func(t *testing.T) {
		r := &reflectionRunner{lastResult: &reflection.RunResult{
			InputCount:                 7,
			EvidenceCount:              5,
			PerQuestionCounts:          []int{2, 3},
			DialecticOkCount:           4,
			DialecticDroppedNoEvidence: 1,
			DialecticDroppedLowConf:    2,
			InsightsWritten:            3,
			InsightsSkipped:            1,
			InsightsWriteFailed:        0,
			DraftsWritten:              2,
		}}
		st := r.status()
		if st.InputCount != 7 || st.EvidenceCount != 5 || st.DialecticOk != 4 ||
			st.DialecticDroppedNoEvidence != 1 || st.DialecticDroppedLowConf != 2 ||
			st.InsightsWritten != 3 || st.InsightsSkipped != 1 ||
			st.InsightsWriteFailed != 0 || st.DraftsWritten != 2 {
			t.Errorf("funnel counters not copied through: %+v", st)
		}
		data, err := json.Marshal(st)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		assertKeysPresent(t, raw)
		if string(raw["per_question_counts"]) != "[2,3]" {
			t.Errorf("per_question_counts = %s, want [2,3]", raw["per_question_counts"])
		}
		if string(raw["input_count"]) != "7" {
			t.Errorf("input_count = %s, want 7", raw["input_count"])
		}
	})
}
