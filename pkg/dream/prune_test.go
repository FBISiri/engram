package dream

import (
	"context"
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

func TestPrune_AgeGate(t *testing.T) {
	now := time.Now()
	oneHourAgo := float64(now.Add(-1 * time.Hour).Unix())
	eightDaysAgo := float64(now.Add(-8 * 24 * time.Hour).Unix())

	tests := []struct {
		name        string
		dryRun      bool
		mem         memory.Memory
		wantDeleted bool // recorded in store.Delete
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
			name:        "old memory is deleted with full-id audit line",
			mem:         memory.Memory{ID: "deadbeef-old-memory-id", Type: memory.TypeEvent, Content: "stale note", Importance: 3, AccessCount: 0, CreatedAt: eightDaysAgo},
			wantDeleted: true,
			wantItem:    "deleted [deadbeef-old-memory-id]",
		},
		{
			name:        "dry-run old memory uses would-delete and does not delete",
			dryRun:      true,
			mem:         memory.Memory{ID: "cafe1234-old-memory-id", Type: memory.TypeEvent, Content: "stale note", Importance: 3, AccessCount: 0, CreatedAt: eightDaysAgo},
			wantDeleted: false,
			wantItem:    "would-delete [cafe1234-old-memory-id]",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
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
		})
	}
}
