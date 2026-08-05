package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"
)

// waitForFile polls until path exists (or deadline), since Log is asynchronous.
func waitForFile(t *testing.T, path string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("file %s never appeared", path)
}

func readRecords(t *testing.T, path string) []Record {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer func() { _ = f.Close() }()
	var recs []Record
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var r Record
		if err := json.Unmarshal(line, &r); err != nil {
			t.Fatalf("unmarshal %q: %v", line, err)
		}
		recs = append(recs, r)
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	return recs
}

func TestLogger_WritesRecordsToDatedFile(t *testing.T) {
	dir := t.TempDir()
	l := New(dir)

	ts := "2026-07-17T10:00:00Z"
	l.Log(Record{
		Timestamp: ts,
		Operation: "retrieve",
		Query:     "what is engram",
		Strategy:  "mmr",
		Results: []ResultItem{
			{ID: "m1", Content: "engram is memory", Score: 0.91},
		},
		LatencyMs: 12,
		Caller:    "unit-test",
	})
	l.Log(Record{
		Timestamp: ts,
		Operation: "update",
		Content:   "new memory",
		Type:      "event",
		Tags:      []string{"a", "b"},
		DedupHit:  true,
		LatencyMs: 3,
	})
	// Close drains the queue before the goroutine exits.
	l.Close()
	// Give the drain goroutine a moment to finish its final writes.
	waitForFile(t, filepath.Join(dir, "2026-07-17.jsonl"))
	// Poll for both records to be flushed.
	var recs []Record
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		recs = readRecords(t, filepath.Join(dir, "2026-07-17.jsonl"))
		if len(recs) == 2 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if len(recs) != 2 {
		t.Fatalf("expected 2 records, got %d", len(recs))
	}
	if recs[0].Operation != "retrieve" || recs[0].Query != "what is engram" {
		t.Errorf("record 0 wrong: %+v", recs[0])
	}
	if len(recs[0].Results) != 1 || recs[0].Results[0].ID != "m1" {
		t.Errorf("record 0 results wrong: %+v", recs[0].Results)
	}
	if recs[1].Operation != "update" || !recs[1].DedupHit {
		t.Errorf("record 1 wrong: %+v", recs[1])
	}
}

func TestLogger_SplitsByDate(t *testing.T) {
	dir := t.TempDir()
	l := New(dir)
	l.Log(Record{Timestamp: "2026-07-17T23:59:59Z", Operation: "update", Content: "day1"})
	l.Log(Record{Timestamp: "2026-07-18T00:00:01Z", Operation: "update", Content: "day2"})
	l.Close()

	waitForFile(t, filepath.Join(dir, "2026-07-17.jsonl"))
	waitForFile(t, filepath.Join(dir, "2026-07-18.jsonl"))

	if r := readRecords(t, filepath.Join(dir, "2026-07-17.jsonl")); len(r) != 1 || r[0].Content != "day1" {
		t.Errorf("2026-07-17 file wrong: %+v", r)
	}
	if r := readRecords(t, filepath.Join(dir, "2026-07-18.jsonl")); len(r) != 1 || r[0].Content != "day2" {
		t.Errorf("2026-07-18 file wrong: %+v", r)
	}
}

// TestLogger_LogAfterClose_PanicsKnownGap is a CHARACTERIZATION test that pins
// down current — and buggy — behavior: calling Log after Close panics with
// "send on closed channel".
//
// The non-blocking `select { case l.ch <- r: default: }` in Log protects only
// against a FULL channel, not a CLOSED one: a send on a closed channel panics
// regardless of the default case. This violates Log's documented contract
// ("Drops silently") and is a real crash risk when a Log call races with Close
// during shutdown. See the report's 遗留建议 section. If production is hardened
// to drop silently after Close, flip this test to assert no panic.
func TestLogger_LogAfterClose_PanicsKnownGap(t *testing.T) {
	dir := t.TempDir()
	l := New(dir)
	l.Log(Record{Timestamp: "2026-07-17T10:00:00Z", Operation: "update", Content: "before"})
	l.Close()
	// Allow the run goroutine to drain and exit.
	time.Sleep(20 * time.Millisecond)

	done := make(chan struct{})
	var recovered any
	go func() {
		defer func() {
			recovered = recover()
			close(done)
		}()
		l.Log(Record{Timestamp: "2026-07-17T10:00:01Z", Operation: "update", Content: "after"})
	}()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("Log after Close blocked instead of panicking (behavior changed)")
	}
	if recovered == nil {
		t.Fatal("expected Log-after-Close to panic (known gap); it did not — " +
			"production may have been hardened, update this test and the report")
	}
}

func TestLogger_ConcurrentLoggersNoLeak(t *testing.T) {
	// Settle first so goroutines from prior tests (async run() drains, race
	// worker, netpoller) have exited before we snapshot the baseline. Without
	// this the baseline can be understated and the assertion falsely trips.
	for i := 0; i < 5; i++ {
		runtime.GC()
		time.Sleep(10 * time.Millisecond)
	}
	before := runtime.NumGoroutine()
	dir := t.TempDir()

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			l := New(dir)
			for j := 0; j < 20; j++ {
				l.Log(Record{Timestamp: "2026-07-17T10:00:00Z", Operation: "update", Content: "x"})
			}
			l.Close()
		}()
	}
	wg.Wait()

	// Each Logger's run goroutine exits when its channel closes; allow scheduling.
	// The +4 slack is a heuristic tolerance for unrelated runtime/test-harness
	// goroutines (GC assist, race worker) that NumGoroutine (a process-global
	// counter) may transiently report; the point is that the 10 run() goroutines
	// must NOT persist, not an exact count.
	const slack = 4
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		runtime.GC()
		if runtime.NumGoroutine() <= before+slack {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := runtime.NumGoroutine(); got > before+slack {
		t.Errorf("goroutine leak: before=%d after=%d (slack=%d)", before, got, slack)
	}
}

func TestLogger_MkdirIfMissing(t *testing.T) {
	// New must tolerate a not-yet-existing dir; run() creates it lazily.
	base := t.TempDir()
	dir := filepath.Join(base, "nested", "traj")
	l := New(dir)
	l.Log(Record{Timestamp: "2026-07-17T10:00:00Z", Operation: "update", Content: "hi"})
	l.Close()
	waitForFile(t, filepath.Join(dir, "2026-07-17.jsonl"))
}

// TestRecord_Phase2Fields_RoundTrip verifies the Phase 2 task_id / task_result
// fields marshal and unmarshal round-trip.
func TestRecord_Phase2Fields_RoundTrip(t *testing.T) {
	in := Record{
		Timestamp:  "2026-08-02T10:00:00Z",
		Operation:  "retrieve",
		Query:      "q",
		LatencyMs:  5,
		TaskID:     "task-123",
		TaskResult: "success",
	}
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out Record
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if out.TaskID != "task-123" || out.TaskResult != "success" {
		t.Errorf("phase2 fields lost: %+v", out)
	}
}

// TestRecord_OldJSON_BackwardCompatible verifies an OLD trajectory line without
// the Phase 2 fields still unmarshals, leaving them empty.
func TestRecord_OldJSON_BackwardCompatible(t *testing.T) {
	old := `{"timestamp":"2026-08-02T10:00:00Z","operation":"retrieve","query":"q","latency_ms":5,"caller":"user"}`
	var r Record
	if err := json.Unmarshal([]byte(old), &r); err != nil {
		t.Fatalf("unmarshal old JSON: %v", err)
	}
	if r.TaskID != "" || r.TaskResult != "" {
		t.Errorf("expected empty phase2 fields, got %+v", r)
	}
	if r.Operation != "retrieve" || r.Query != "q" {
		t.Errorf("old fields wrong: %+v", r)
	}
}

// TestRecord_Phase2Fields_OmitEmpty verifies omitempty drops the new fields
// from output when unset (keeps existing JSONL byte-compatible).
func TestRecord_Phase2Fields_OmitEmpty(t *testing.T) {
	data, err := json.Marshal(Record{Timestamp: "2026-08-02T10:00:00Z", Operation: "update"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(data, []byte("task_id")) || bytes.Contains(data, []byte("task_result")) {
		t.Errorf("omitempty failed, task fields present: %s", data)
	}
}
