package replay

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// LoadTrace reads one trajectory JSONL file and returns the retrieve-only
// operations converted to ReplayCases. Malformed lines and non-retrieve
// records are skipped silently (production logs may contain both operations).
func LoadTrace(path string) ([]ReplayCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open trace %s: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	var cases []ReplayCase
	sc := bufio.NewScanner(f)
	// Trajectory records can be large (many long result contents); grow the
	// scanner buffer well past the 64KB default.
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec trajectory.Record
		if err := json.Unmarshal(line, &rec); err != nil {
			continue // skip malformed line
		}
		if rec.Operation != "retrieve" {
			continue // updates are context, not replay targets (MVP)
		}
		cases = append(cases, recordToCase(rec))
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("scan trace %s: %w", path, err)
	}
	return cases, nil
}

func recordToCase(rec trajectory.Record) ReplayCase {
	ts, _ := time.Parse(time.RFC3339, rec.Timestamp)
	return ReplayCase{
		Timestamp:       ts,
		Query:           rec.Query,
		Strategy:        rec.Strategy,
		Caller:          rec.Caller,
		RecordedResults: rec.Results,
		RecordedLatency: rec.LatencyMs,
	}
}
