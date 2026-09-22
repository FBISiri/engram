package dream

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// helper: does any item contain substr?
func itemsContain(items []string, substr string) bool {
	for _, it := range items {
		if strings.Contains(it, substr) {
			return true
		}
	}
	return false
}

// deletedContains reports whether the fakeStore recorded a delete of id.
func deletedContains(f *fakeStore, id string) bool {
	for _, d := range f.deleted {
		if d == id {
			return true
		}
	}
	return false
}

// readMarks loads the on-disk marks file from the temp state dir.
func readMarks(t *testing.T, dir string) *pruneMarks {
	t.Helper()
	return loadPruneMarks(dir)
}

// seedMark writes a marks file with a single pre-seeded mark.
func seedMark(t *testing.T, dir string, mk pruneMark) {
	t.Helper()
	m := &pruneMarks{Marks: map[string]pruneMark{mk.ID: mk}}
	if err := m.save(dir); err != nil {
		t.Fatalf("seed marks: %v", err)
	}
}

// TestPrune_AgeGate validates the 7d age gate under grace semantics: a fresh
// memory is skipped; an old memory is MARKED (not deleted) on a single run; a
// dry-run writes nothing and deletes nothing.
func TestPrune_AgeGate(t *testing.T) {
	now := time.Now()
	oneHourAgo := float64(now.Add(-1 * time.Hour).Unix())
	eightDaysAgo := float64(now.Add(-8 * 24 * time.Hour).Unix())

	tests := []struct {
		name        string
		dryRun      bool
		mem         memory.Memory
		wantDeleted bool
		wantItem    string
		wantSkipped bool
	}{
		{
			name:        "fresh memory is skipped, not deleted",
			mem:         memory.Memory{ID: "8980c8a1-fresh-id", Type: memory.TypeEvent, Content: "inbound email from founders@xerj.org", Importance: 3, AccessCount: 0, CreatedAt: oneHourAgo},
			wantDeleted: false,
			wantItem:    "prune skipped (age<7d): 1",
			wantSkipped: true,
		},
		{
			name:        "old memory is marked (not deleted) on first eligible run",
			mem:         memory.Memory{ID: "deadbeef-old-memory-id", Type: memory.TypeEvent, Content: "stale note", Importance: 3, AccessCount: 0, CreatedAt: eightDaysAgo},
			wantDeleted: false,
			wantItem:    "marked [deadbeef-old-memory-id]",
		},
		{
			name:        "dry-run old memory uses would-mark and does not delete",
			dryRun:      true,
			mem:         memory.Memory{ID: "cafe1234-old-memory-id", Type: memory.TypeEvent, Content: "stale note", Importance: 3, AccessCount: 0, CreatedAt: eightDaysAgo},
			wantDeleted: false,
			wantItem:    "would-mark [cafe1234-old-memory-id]",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("ENGRAM_STATE_DIR", dir)
			store := &fakeStore{all: []memory.Memory{tc.mem}}
			e := NewEngine(store, nil, Config{DryRun: tc.dryRun})

			items, err := e.prune(context.Background())
			if err != nil {
				t.Fatalf("prune: %v", err)
			}

			if got := deletedContains(store, tc.mem.ID); got != tc.wantDeleted {
				t.Errorf("delete of %s = %v, want %v (deleted=%v)", tc.mem.ID, got, tc.wantDeleted, store.deleted)
			}
			if !itemsContain(items, tc.wantItem) {
				t.Errorf("items missing %q; got:\n%s", tc.wantItem, strings.Join(items, "\n"))
			}
			if tc.wantSkipped && !itemsContain(items, "prune skipped (age<7d): 1") {
				t.Errorf("expected skipped line; got:\n%s", strings.Join(items, "\n"))
			}

			// Dry-run must write nothing.
			if tc.dryRun {
				if _, err := os.Stat(filepath.Join(dir, pruneMarksFile)); !os.IsNotExist(err) {
					t.Errorf("dry-run wrote marks file")
				}
				if _, err := os.Stat(filepath.Join(dir, prunePendingFile)); !os.IsNotExist(err) {
					t.Errorf("dry-run wrote pending file")
				}
			}
		})
	}
}

// (a) FIRST-ELIGIBLE run does NOT delete; it only marks.
func TestPrune_FirstEligibleMarksOnly(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	now := time.Now()
	mem := memory.Memory{ID: "first-eligible-id", Type: memory.TypeEvent, Content: "stale note", Importance: 3, AccessCount: 0, CreatedAt: float64(now.Add(-8 * 24 * time.Hour).Unix())}
	store := &fakeStore{all: []memory.Memory{mem}}
	e := NewEngine(store, nil, Config{})

	items, err := e.prune(context.Background())
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if len(store.deleted) != 0 {
		t.Errorf("expected no deletes, got %v", store.deleted)
	}
	if !itemsContain(items, "marked [first-eligible-id]") {
		t.Errorf("expected marked line; got:\n%s", strings.Join(items, "\n"))
	}
	marks := readMarks(t, dir)
	mk, ok := marks.Marks[mem.ID]
	if !ok {
		t.Fatalf("mark not written for %s", mem.ID)
	}
	if mk.RunsMarked != 1 {
		t.Errorf("RunsMarked = %d, want 1", mk.RunsMarked)
	}
}

// (b) DELETION AFTER GRACE, plus NEGATIVE first-run case.
func TestPrune_DeletionAfterGrace(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	now := time.Now()
	createdAt := float64(now.Add(-10 * 24 * time.Hour).Unix())
	firstMarked := float64(now.Add(-73 * time.Hour).Unix()) // >72h ago

	graceMem := memory.Memory{ID: "grace-due-id", Type: memory.TypeEvent, Content: "stale", Importance: 3, AccessCount: 0, CreatedAt: createdAt}
	// Pre-seed a mark: first marked >72h ago, RunsMarked=1. This run makes it 2.
	seedMark(t, dir, pruneMark{
		ID: graceMem.ID, FirstMarkedAt: firstMarked, LastMarkedAt: firstMarked,
		RunsMarked: 1, Type: string(memory.TypeEvent), Importance: 3, CreatedAt: createdAt, Content: "stale",
	})

	store := &fakeStore{all: []memory.Memory{graceMem}}
	e := NewEngine(store, nil, Config{})
	items, err := e.prune(context.Background())
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if !deletedContains(store, graceMem.ID) {
		t.Errorf("grace-due memory not deleted; deleted=%v", store.deleted)
	}
	if !itemsContain(items, "deleted [grace-due-id]") {
		t.Errorf("expected deleted line; got:\n%s", strings.Join(items, "\n"))
	}
	// pending_delete.json deleting_now must include it.
	pd := readPending(t, dir)
	if !containsStr(pd.DeletingNow, graceMem.ID) {
		t.Errorf("deleting_now missing %s: %v", graceMem.ID, pd.DeletingNow)
	}
	// deleted mark removed from marks file.
	if _, ok := readMarks(t, dir).Marks[graceMem.ID]; ok {
		t.Errorf("deleted mark should be removed from marks file")
	}

	// NEGATIVE: same-age memory but FIRST run (no pre-seed) -> not deleted.
	dir2 := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir2)
	freshMem := memory.Memory{ID: "grace-firstrun-id", Type: memory.TypeEvent, Content: "stale", Importance: 3, AccessCount: 0, CreatedAt: createdAt}
	store2 := &fakeStore{all: []memory.Memory{freshMem}}
	e2 := NewEngine(store2, nil, Config{})
	if _, err := e2.prune(context.Background()); err != nil {
		t.Fatalf("prune2: %v", err)
	}
	if deletedContains(store2, freshMem.ID) {
		t.Errorf("first-run memory must NOT be deleted; deleted=%v", store2.deleted)
	}
}

// (c) DIRECTIVE EXEMPT: never deleted, never marked, even when pre-seeded to
// look grace-eligible. This test MUST FAIL if the exemption code is removed.
func TestPrune_DirectiveExempt(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	now := time.Now()
	createdAt := float64(now.Add(-10 * 24 * time.Hour).Unix())
	firstMarked := float64(now.Add(-73 * time.Hour).Unix())

	dirMem := memory.Memory{ID: "directive-id", Type: memory.TypeDirective, Content: "always do X", Importance: 3, AccessCount: 0, CreatedAt: createdAt}
	// Pre-seed a grace-eligible mark to prove exemption wins over grace.
	seedMark(t, dir, pruneMark{
		ID: dirMem.ID, FirstMarkedAt: firstMarked, LastMarkedAt: firstMarked,
		RunsMarked: 1, Type: string(memory.TypeDirective), Importance: 3, CreatedAt: createdAt, Content: "always do X",
	})

	store := &fakeStore{all: []memory.Memory{dirMem}}
	e := NewEngine(store, nil, Config{})
	items, err := e.prune(context.Background())
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if deletedContains(store, dirMem.ID) {
		t.Errorf("directive must never be deleted; deleted=%v", store.deleted)
	}
	if !itemsContain(items, "prune exempt (directive): 1") {
		t.Errorf("expected directive-exempt line; got:\n%s", strings.Join(items, "\n"))
	}
	// The pre-seeded mark is reconciled away (no longer eligible) — absent.
	if _, ok := readMarks(t, dir).Marks[dirMem.ID]; ok {
		t.Errorf("directive must never be marked; found in marks file")
	}
}

// (d) PENDING-DELETE LIST WRITTEN even on a first-eligible run (nothing deleted).
func TestPrune_PendingDeleteWritten(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	now := time.Now()
	mem := memory.Memory{ID: "pending-id", Type: memory.TypeEvent, Content: "stale", Importance: 3, AccessCount: 0, CreatedAt: float64(now.Add(-8 * 24 * time.Hour).Unix())}
	store := &fakeStore{all: []memory.Memory{mem}}
	e := NewEngine(store, nil, Config{})
	if _, err := e.prune(context.Background()); err != nil {
		t.Fatalf("prune: %v", err)
	}
	if len(store.deleted) != 0 {
		t.Fatalf("expected no deletes on first-eligible run; got %v", store.deleted)
	}
	pd := readPending(t, dir)
	found := false
	for _, p := range pd.Pending {
		if p.ID == mem.ID {
			found = true
		}
	}
	if !found {
		t.Errorf("pending array missing %s: %+v", mem.ID, pd.Pending)
	}
	if len(pd.DeletingNow) != 0 {
		t.Errorf("deleting_now should be empty on first-eligible run: %v", pd.DeletingNow)
	}
}

func readPending(t *testing.T, dir string) *pendingDelete {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(dir, prunePendingFile))
	if err != nil {
		t.Fatalf("read pending: %v", err)
	}
	var pd pendingDelete
	if err := json.Unmarshal(data, &pd); err != nil {
		t.Fatalf("parse pending: %v", err)
	}
	return &pd
}

// (e) PENDING-DELETE WRITE FAILURE aborts the delete (C1).
func TestPrune_PendingWriteFailureAbortsDelete(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ENGRAM_STATE_DIR", dir)
	now := time.Now()
	createdAt := float64(now.Add(-10 * 24 * time.Hour).Unix())
	firstMarked := float64(now.Add(-73 * time.Hour).Unix())

	mem := memory.Memory{ID: "abort-id", Type: memory.TypeEvent, Content: "stale", Importance: 3, AccessCount: 0, CreatedAt: createdAt}
	seedMark(t, dir, pruneMark{
		ID: mem.ID, FirstMarkedAt: firstMarked, LastMarkedAt: firstMarked,
		RunsMarked: 1, Type: string(memory.TypeEvent), Importance: 3, CreatedAt: createdAt, Content: "stale",
	})
	// Force the pending-delete write to fail: pre-create the target path as a
	// directory so WriteFile(tmp)+Rename(tmp,final) cannot succeed.
	if err := os.Mkdir(filepath.Join(dir, prunePendingFile), 0755); err != nil {
		t.Fatalf("mkdir pending: %v", err)
	}

	store := &fakeStore{all: []memory.Memory{mem}}
	e := NewEngine(store, nil, Config{})
	items, err := e.prune(context.Background())
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if len(store.deleted) != 0 {
		t.Errorf("delete must be aborted on pending-write failure; deleted=%v", store.deleted)
	}
	if !itemsContain(items, "delete ABORTED") {
		t.Errorf("expected abort log; got:\n%s", strings.Join(items, "\n"))
	}
	// Marks must persist (streak retained) — mark not removed.
	if _, ok := readMarks(t, dir).Marks[mem.ID]; !ok {
		t.Errorf("mark must persist after aborted delete")
	}
}

func containsStr(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}
