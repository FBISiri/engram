package main

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

type fakeStore struct {
	mu   sync.Mutex
	mems map[string]memory.Memory
}

func newFakeStore(ms ...memory.Memory) *fakeStore {
	store := &fakeStore{mems: make(map[string]memory.Memory)}
	for _, m := range ms {
		store.mems[m.ID] = m
	}
	return store
}

func (f *fakeStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []memory.Memory
	for _, m := range f.mems {
		match := true
		for _, fl := range opts.Filters {
			if fl.Field == "type" && string(m.Type) != fl.Value.(string) {
				match = false
			}
		}
		if match {
			out = append(out, m)
		}
	}
	return out, "", nil
}

func (f *fakeStore) Update(_ context.Context, id string, fields map[string]any) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	m, ok := f.mems[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	if v, ok := fields["valid_until"]; ok {
		m.ValidUntil = v.(float64)
	}
	f.mems[id] = m
	return nil
}

func (f *fakeStore) SearchByIDs(_ context.Context, ids []string) ([]memory.Memory, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []memory.Memory
	for _, id := range ids {
		if m, ok := f.mems[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}

func (f *fakeStore) EnsureCollection(_ context.Context) error { return nil }

func TestScan_FindsMissingValidUntil(t *testing.T) {
	now := float64(time.Now().Unix())
	store := newFakeStore(
		memory.Memory{
			ID: "has-ttl", Type: memory.TypeInsight, Content: "already set",
			Source: "system", ValidUntil: now + thirtyDays, CreatedAt: now,
		},
		memory.Memory{
			ID: "missing-ttl", Type: memory.TypeInsight, Content: "needs backfill",
			Source: "system", CreatedAt: now - 86400,
		},
		memory.Memory{
			ID: "not-reflection", Type: memory.TypeInsight, Content: "user insight",
			Source: "agent", CreatedAt: now,
		},
	)

	candidates, scanned, alreadySet, err := scan(context.Background(), store)
	if err != nil {
		t.Fatal(err)
	}
	if scanned != 2 {
		t.Errorf("scanned: got %d, want 2", scanned)
	}
	if alreadySet != 1 {
		t.Errorf("alreadySet: got %d, want 1", alreadySet)
	}
	if len(candidates) != 1 {
		t.Fatalf("candidates: got %d, want 1", len(candidates))
	}
	if candidates[0].ID != "missing-ttl" {
		t.Errorf("candidate ID: got %s, want missing-ttl", candidates[0].ID)
	}
	expectedTTL := (now - 86400) + thirtyDays
	if candidates[0].NewValidUntil != expectedTTL {
		t.Errorf("NewValidUntil: got %f, want %f", candidates[0].NewValidUntil, expectedTTL)
	}
}

func TestPatch_SetsValidUntil(t *testing.T) {
	now := float64(time.Now().Unix())
	store := newFakeStore(memory.Memory{
		ID: "m1", Type: memory.TypeInsight, Content: "test",
		Source: "system", CreatedAt: now - 86400,
	})

	cs := []candidate{{
		ID:            "m1",
		CreatedAt:     now - 86400,
		NewValidUntil: (now - 86400) + thirtyDays,
	}}

	patched, errs := patch(context.Background(), store, cs)
	if patched != 1 {
		t.Errorf("patched: got %d, want 1", patched)
	}
	if len(errs) != 0 {
		t.Errorf("errors: %v", errs)
	}

	m := store.mems["m1"]
	if m.ValidUntil != cs[0].NewValidUntil {
		t.Errorf("ValidUntil: got %f, want %f", m.ValidUntil, cs[0].NewValidUntil)
	}
}

func TestRun_DryRunDoesNotPatch(t *testing.T) {
	now := float64(time.Now().Unix())
	store := newFakeStore(memory.Memory{
		ID: "m1", Type: memory.TypeInsight, Content: "test",
		Source: "system", CreatedAt: now,
	})

	res, err := run(context.Background(), store, options{dryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.CandidateCount != 1 {
		t.Errorf("candidates: got %d, want 1", res.CandidateCount)
	}
	if res.Patched != 0 {
		t.Errorf("patched: got %d, want 0 (dry run)", res.Patched)
	}
	if store.mems["m1"].ValidUntil != 0 {
		t.Error("dry run should not modify store")
	}
}

func TestRun_PatchAndVerify(t *testing.T) {
	now := float64(time.Now().Unix())
	store := newFakeStore(memory.Memory{
		ID: "m1", Type: memory.TypeInsight, Content: "test",
		Source: "system", CreatedAt: now - 172800,
	})

	res, err := run(context.Background(), store, options{dryRun: false, verify: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Patched != 1 {
		t.Errorf("patched: got %d, want 1", res.Patched)
	}
	if !res.VerifyPassed {
		t.Errorf("verify should pass, failed IDs: %v", res.VerifyFailed)
	}
}
