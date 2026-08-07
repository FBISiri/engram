package replay

import (
	"bufio"
	"encoding/json"
	"os"
	"sort"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// LoadRecords reads every well-formed record from a trajectory JSONL file (both
// retrieve and update ops), for Memory Worth joining. Malformed lines are
// skipped. Unlike LoadTrace it does not filter or convert to ReplayCases,
// because MW needs the raw task_id / task_result fields.
func LoadRecords(path string) ([]trajectory.Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var recs []trajectory.Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var rec trajectory.Record
		if err := json.Unmarshal(line, &rec); err != nil {
			continue
		}
		recs = append(recs, rec)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return recs, nil
}

// -----------------------------------------------------------------------------
// D. Memory Worth (MW) — Phase 2 P0
// -----------------------------------------------------------------------------
//
// Memory Worth quantifies which retrieved memories contributed to task success.
// It joins trajectory records by TaskID: each task has an outcome (task_result)
// and a set of retrieved memory IDs. For every memory, MW = (tasks where it was
// retrieved AND the task succeeded) / (tasks where it was retrieved at all).

// MemoryWorth holds the aggregated worth of a single memory across all tasks.
type MemoryWorth struct {
	MemoryID     string  `json:"memory_id"`
	Content      string  `json:"content"`       // first 200 chars, for reporting
	TimesUsed    int     `json:"times_used"`    // total tasks where this memory was retrieved
	TimesSuccess int     `json:"times_success"` // tasks where retrieved AND task_result == "success"
	TimesFailed  int     `json:"times_failed"`  // tasks where retrieved AND task_result in {"error","failed"}
	MW           float64 `json:"mw"`            // TimesSuccess / TimesUsed
}

// MWReport summarizes Memory Worth across a full trajectory.
type MWReport struct {
	TotalTasks      int           `json:"total_tasks"`
	TasksWithMemory int           `json:"tasks_with_memory"` // tasks that retrieved >=1 memory
	MWDistribution  [10]int       `json:"mw_distribution"`   // 0-10%,10-20%,...,90-100%
	TopMemories     []MemoryWorth `json:"top_memories"`      // top 10 by MW
	BottomMemories  []MemoryWorth `json:"bottom_memories"`   // bottom 10 (high usage, low MW)
	MeanMW          float64       `json:"mean_mw"`
	MedianMW        float64       `json:"median_mw"`
}

// taskOutcome classifies a task_result string.
type taskOutcome int

const (
	outcomeNeither taskOutcome = iota // unknown / skip / other
	outcomeSuccess
	outcomeFailure
)

func classifyResult(taskResult string) taskOutcome {
	switch taskResult {
	case "success":
		return outcomeSuccess
	case "error", "failed":
		return outcomeFailure
	default:
		return outcomeNeither
	}
}

// ComputeMW joins trajectory records by task_id and computes per-memory Memory
// Worth. Only records with a non-empty TaskID are considered. Returns nil if no
// such records exist (backward-compatible: legacy JSONL without task_id yields
// no MW report).
//
// A task's outcome is taken from the first non-empty task_result seen among its
// records. A memory is counted at most once per task (retrieval within a task is
// deduplicated). task_result "success" => success; "error"/"failed" => failure;
// anything else => neither (counted in TimesUsed only).
func ComputeMW(records []trajectory.Record) *MWReport {
	// Group records by TaskID, preserving first-seen task order for determinism.
	type taskAgg struct {
		outcome taskOutcome
		mems    map[string]string // memory ID -> content (first seen)
	}
	tasks := make(map[string]*taskAgg)
	var taskOrder []string

	for _, rec := range records {
		if rec.TaskID == "" {
			continue
		}
		t, ok := tasks[rec.TaskID]
		if !ok {
			t = &taskAgg{mems: make(map[string]string)}
			tasks[rec.TaskID] = t
			taskOrder = append(taskOrder, rec.TaskID)
		}
		// First non-empty task_result wins for this task.
		if t.outcome == outcomeNeither {
			t.outcome = classifyResult(rec.TaskResult)
		}
		for _, item := range rec.Results {
			if _, seen := t.mems[item.ID]; !seen {
				t.mems[item.ID] = truncate200(item.Content)
			}
		}
	}
	if len(tasks) == 0 {
		return nil
	}

	// Aggregate per-memory counts across tasks.
	worth := make(map[string]*MemoryWorth)
	tasksWithMemory := 0
	for _, taskID := range taskOrder {
		t := tasks[taskID]
		if len(t.mems) > 0 {
			tasksWithMemory++
		}
		for id, content := range t.mems {
			w, ok := worth[id]
			if !ok {
				w = &MemoryWorth{MemoryID: id, Content: content}
				worth[id] = w
			}
			w.TimesUsed++
			switch t.outcome {
			case outcomeSuccess:
				w.TimesSuccess++
			case outcomeFailure:
				w.TimesFailed++
			}
		}
	}

	rep := &MWReport{TotalTasks: len(tasks), TasksWithMemory: tasksWithMemory}

	all := make([]MemoryWorth, 0, len(worth))
	mws := make([]float64, 0, len(worth))
	for _, w := range worth {
		if w.TimesUsed > 0 {
			w.MW = float64(w.TimesSuccess) / float64(w.TimesUsed)
		}
		b := int(w.MW * 10)
		if b > 9 {
			b = 9
		}
		if b < 0 {
			b = 0
		}
		rep.MWDistribution[b]++
		all = append(all, *w)
		mws = append(mws, w.MW)
	}

	rep.MeanMW = mean(mws)
	rep.MedianMW = median(mws)

	// Top: highest MW first, tie-break by MemoryID for determinism.
	top := make([]MemoryWorth, len(all))
	copy(top, all)
	sort.Slice(top, func(i, j int) bool {
		if top[i].MW != top[j].MW {
			return top[i].MW > top[j].MW
		}
		return top[i].MemoryID < top[j].MemoryID
	})
	rep.TopMemories = firstN(top, 10)

	// Bottom: surfaces the "high usage, low MW" liabilities. Ordered ascending
	// MW, then descending TimesUsed (a low-MW memory used often is worse than one
	// used rarely), then MemoryID for a stable tie-break.
	bottom := make([]MemoryWorth, len(all))
	copy(bottom, all)
	sort.Slice(bottom, func(i, j int) bool {
		if bottom[i].MW != bottom[j].MW {
			return bottom[i].MW < bottom[j].MW
		}
		if bottom[i].TimesUsed != bottom[j].TimesUsed {
			return bottom[i].TimesUsed > bottom[j].TimesUsed
		}
		return bottom[i].MemoryID < bottom[j].MemoryID
	})
	rep.BottomMemories = firstN(bottom, 10)

	return rep
}

func firstN(xs []MemoryWorth, n int) []MemoryWorth {
	if len(xs) < n {
		n = len(xs)
	}
	return xs[:n]
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

func median(xs []float64) float64 {
	n := len(xs)
	if n == 0 {
		return 0
	}
	cp := make([]float64, n)
	copy(cp, xs)
	sort.Float64s(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

func truncate200(s string) string {
	r := []rune(s)
	if len(r) <= 200 {
		return s
	}
	return string(r[:200])
}
