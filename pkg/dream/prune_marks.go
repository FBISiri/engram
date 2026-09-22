package dream

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"
)

// pruneMarksFile is the persistent grace queue: it records, per memory id, how
// many consecutive prune runs the memory has been eligible for and when it was
// first seen eligible. It survives restarts so the grace period is real wall
// time, not per-process state.
const pruneMarksFile = "dream_prune_marks.json"

// prunePendingFile is a human-readable + machine-parsable snapshot of the grace
// queue and what is being deleted this run. It is written BEFORE any Delete
// call (constraint C1 / R2): there is no direct delete path.
const prunePendingFile = "dream_pending_delete.json"

// pruneMark is a single entry in the grace queue.
type pruneMark struct {
	ID            string  `json:"id"`
	FirstMarkedAt float64 `json:"first_marked_at"` // unix seconds, set once
	LastMarkedAt  float64 `json:"last_marked_at"`  // unix seconds, updated each run
	RunsMarked    int     `json:"runs_marked"`     // consecutive eligible prune runs
	Type          string  `json:"type"`
	Importance    float64 `json:"importance"`
	CreatedAt     float64 `json:"created_at"` // memory created_at
	Content       string  `json:"content"`    // rune-truncated snippet (<=80)
}

// pruneMarks is the on-disk file shape.
type pruneMarks struct {
	Marks map[string]pruneMark `json:"marks"`
}

// loadPruneMarks reads the marks file. A missing file yields an empty set; a
// corrupt file also yields an empty set without crashing (self-healing).
func loadPruneMarks(stateDir string) *pruneMarks {
	m := &pruneMarks{Marks: map[string]pruneMark{}}
	data, err := os.ReadFile(filepath.Join(stateDir, pruneMarksFile))
	if err != nil {
		return m
	}
	var parsed pruneMarks
	if err := json.Unmarshal(data, &parsed); err != nil || parsed.Marks == nil {
		return m
	}
	m.Marks = parsed.Marks
	return m
}

// save writes the marks file atomically-ish (temp then rename, 0644).
func (m *pruneMarks) save(stateDir string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	final := filepath.Join(stateDir, pruneMarksFile)
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

// pendingEntry is one row of the pending-delete snapshot.
type pendingEntry struct {
	ID            string  `json:"id"`
	Type          string  `json:"type"`
	Importance    float64 `json:"importance"`
	CreatedAt     float64 `json:"created_at"`
	FirstMarkedAt float64 `json:"first_marked_at"`
	RunsMarked    int     `json:"runs_marked"`
	AgeDays       int     `json:"age_days"`
	DueForDelete  bool    `json:"due_for_delete"`
}

// pendingDelete is the on-disk shape of dream_pending_delete.json.
type pendingDelete struct {
	GeneratedAt      string         `json:"generated_at"`
	GraceMinAgeHours int            `json:"grace_min_age_hours"`
	GraceMinRuns     int            `json:"grace_min_runs"`
	Pending          []pendingEntry `json:"pending"`
	DeletingNow      []string       `json:"deleting_now"`
}

// writePendingDelete writes the pending-delete snapshot atomically-ish.
func writePendingDelete(stateDir string, pd *pendingDelete) error {
	data, err := json.MarshalIndent(pd, "", "  ")
	if err != nil {
		return err
	}
	final := filepath.Join(stateDir, prunePendingFile)
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

// runeTruncate truncates s to at most n runes (never splitting a UTF-8 char).
func runeTruncate(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}

// nowUnix is a small indirection so tests need not stub time; it returns the
// current unix seconds as float64.
func nowUnix() float64 { return float64(time.Now().Unix()) }
