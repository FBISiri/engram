package replay

import (
	"testing"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// retrieveRec builds a retrieve record with a task id/result and result IDs.
func retrieveRec(taskID, taskResult string, ids ...string) trajectory.Record {
	items := make([]trajectory.ResultItem, len(ids))
	for i, id := range ids {
		items[i] = trajectory.ResultItem{ID: id, Content: "content-" + id, Score: 1.0}
	}
	return trajectory.Record{
		Operation:  "retrieve",
		TaskID:     taskID,
		TaskResult: taskResult,
		Results:    items,
	}
}

func findWorth(t *testing.T, mw *MWReport, id string) MemoryWorth {
	t.Helper()
	for _, m := range mw.TopMemories {
		if m.MemoryID == id {
			return m
		}
	}
	t.Fatalf("memory %s not found in report", id)
	return MemoryWorth{}
}

func TestComputeMW_EmptyRecords(t *testing.T) {
	if mw := ComputeMW(nil); mw != nil {
		t.Fatalf("expected nil for empty records, got %+v", mw)
	}
}

func TestComputeMW_NoTaskID(t *testing.T) {
	recs := []trajectory.Record{
		retrieveRec("", "success", "m1", "m2"),
		retrieveRec("", "error", "m3"),
	}
	if mw := ComputeMW(recs); mw != nil {
		t.Fatalf("expected nil when no task_id present, got %+v", mw)
	}
}

func TestComputeMW_SuccessOnly(t *testing.T) {
	recs := []trajectory.Record{
		retrieveRec("t1", "success", "m1"),
		retrieveRec("t2", "success", "m1"),
	}
	mw := ComputeMW(recs)
	if mw == nil {
		t.Fatal("expected non-nil report")
	}
	m1 := findWorth(t, mw, "m1")
	if m1.MW != 1.0 || m1.TimesUsed != 2 || m1.TimesSuccess != 2 || m1.TimesFailed != 0 {
		t.Fatalf("unexpected m1: %+v", m1)
	}
	if mw.MeanMW != 1.0 || mw.MedianMW != 1.0 {
		t.Fatalf("mean/median = %.2f/%.2f, want 1.0/1.0", mw.MeanMW, mw.MedianMW)
	}
	if mw.TotalTasks != 2 || mw.TasksWithMemory != 2 {
		t.Fatalf("tasks = %d / with-memory %d, want 2/2", mw.TotalTasks, mw.TasksWithMemory)
	}
}

func TestComputeMW_MixedSuccessFailure(t *testing.T) {
	recs := []trajectory.Record{
		retrieveRec("t1", "success", "m1"),
		retrieveRec("t2", "error", "m1"),
		retrieveRec("t3", "failed", "m1"),
		retrieveRec("t4", "skip", "m1"), // neither: counts in TimesUsed only
	}
	mw := ComputeMW(recs)
	m1 := findWorth(t, mw, "m1")
	if m1.TimesUsed != 4 || m1.TimesSuccess != 1 || m1.TimesFailed != 2 {
		t.Fatalf("unexpected counts: %+v", m1)
	}
	if m1.MW != 0.25 {
		t.Fatalf("MW = %.4f, want 0.25", m1.MW)
	}
}

func TestComputeMW_MemoryOncePerTask(t *testing.T) {
	// Same memory retrieved twice within one task counts once.
	recs := []trajectory.Record{
		retrieveRec("t1", "success", "m1"),
		retrieveRec("t1", "", "m1", "m2"), // same task, second retrieve op
	}
	mw := ComputeMW(recs)
	m1 := findWorth(t, mw, "m1")
	if m1.TimesUsed != 1 || m1.TimesSuccess != 1 {
		t.Fatalf("m1 should be counted once per task: %+v", m1)
	}
	m2 := findWorth(t, mw, "m2")
	if m2.TimesUsed != 1 || m2.TimesSuccess != 1 {
		t.Fatalf("m2 should inherit task outcome: %+v", m2)
	}
	if mw.TotalTasks != 1 {
		t.Fatalf("TotalTasks = %d, want 1", mw.TotalTasks)
	}
}

func TestComputeMW_Distribution(t *testing.T) {
	// m1: MW=1.0 -> bucket 9; m2: MW=0.0 -> bucket 0; m3: MW=0.5 -> bucket 5.
	recs := []trajectory.Record{
		retrieveRec("t1", "success", "m1", "m3"),
		retrieveRec("t2", "error", "m2", "m3"),
	}
	mw := ComputeMW(recs)
	if mw.MWDistribution[9] != 1 {
		t.Errorf("bucket 9 = %d, want 1", mw.MWDistribution[9])
	}
	if mw.MWDistribution[0] != 1 {
		t.Errorf("bucket 0 = %d, want 1", mw.MWDistribution[0])
	}
	if mw.MWDistribution[5] != 1 {
		t.Errorf("bucket 5 = %d, want 1", mw.MWDistribution[5])
	}
	if mw.MedianMW != 0.5 {
		t.Errorf("median = %.2f, want 0.5", mw.MedianMW)
	}
}

func TestComputeMW_TopBottomSorting(t *testing.T) {
	// good: MW=1.0 used once; bad: MW=0.0 used twice; mid: MW=0.5 used twice.
	recs := []trajectory.Record{
		retrieveRec("t1", "success", "good", "mid"),
		retrieveRec("t2", "error", "bad", "mid"),
		retrieveRec("t3", "error", "bad"),
	}
	mw := ComputeMW(recs)
	if mw.TopMemories[0].MemoryID != "good" {
		t.Errorf("top[0] = %s, want good", mw.TopMemories[0].MemoryID)
	}
	// bottom: lowest MW first; "bad" (MW 0, used 2) before others.
	if mw.BottomMemories[0].MemoryID != "bad" {
		t.Errorf("bottom[0] = %s, want bad", mw.BottomMemories[0].MemoryID)
	}
	if mw.BottomMemories[0].TimesUsed != 2 {
		t.Errorf("bad TimesUsed = %d, want 2", mw.BottomMemories[0].TimesUsed)
	}
}

func TestComputeMW_ContentTruncated(t *testing.T) {
	long := make([]rune, 300)
	for i := range long {
		long[i] = 'x'
	}
	rec := trajectory.Record{
		Operation:  "retrieve",
		TaskID:     "t1",
		TaskResult: "success",
		Results:    []trajectory.ResultItem{{ID: "m1", Content: string(long)}},
	}
	mw := ComputeMW([]trajectory.Record{rec})
	m1 := findWorth(t, mw, "m1")
	if len([]rune(m1.Content)) != 200 {
		t.Fatalf("content len = %d runes, want 200", len([]rune(m1.Content)))
	}
}
