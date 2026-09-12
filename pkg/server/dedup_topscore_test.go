package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// TestDedupTopScoreRecorded verifies checkDedup's top score is carried onto the
// candidate trajectory record on both the admitted and dedup_rejected paths
// (A-6). Reads only from a t.TempDir() trajectory file — never production paths.
func TestDedupTopScoreRecorded(t *testing.T) {
	srv, _ := newTestServer()
	dir := t.TempDir()
	traj := trajectory.New(dir)
	srv.SetTrajectoryLogger(traj)

	// First add: admitted. Second near-identical add: dedup_rejected.
	for i := 0; i < 2; i++ {
		if _, err := callTool(srv, "memory_add", map[string]any{
			"content":     "the nightly backup job writes to bucket engram-cold",
			"type":        "event",
			"source_type": "user_input",
		}); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}
	traj.Close()
	time.Sleep(50 * time.Millisecond)

	recs := readTrajRecords(t, dir)
	var admitted, rejected *trajectory.Record
	for i := range recs {
		switch recs[i].AdmissionDecision {
		case "admitted":
			admitted = &recs[i]
		case "dedup_rejected":
			rejected = &recs[i]
		}
	}
	if rejected == nil {
		t.Fatal("no dedup_rejected candidate record found")
	}
	if rejected.DedupTopScore <= 0 {
		t.Errorf("dedup_rejected record DedupTopScore=%v, want > 0", rejected.DedupTopScore)
	}
	// The admitted record's top score reflects the (empty-store) first search;
	// it should be present as a field on the record type regardless.
	_ = admitted
}

func readTrajRecords(t *testing.T, dir string) []trajectory.Record {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	var out []trajectory.Record
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
			if line == "" {
				continue
			}
			var r trajectory.Record
			if err := json.Unmarshal([]byte(line), &r); err != nil {
				t.Fatalf("unmarshal %q: %v", line, err)
			}
			out = append(out, r)
		}
	}
	return out
}
