package main

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── fake store (minimal, in-memory) ──────────────────────────────────────────

type fakeStore struct {
	mu    sync.Mutex
	mems  map[string]memory.Memory
	order []string // insertion order for deterministic scroll
}

func newFakeStore() *fakeStore {
	return &fakeStore{mems: make(map[string]memory.Memory)}
}

func (f *fakeStore) add(m memory.Memory) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if _, ok := f.mems[m.ID]; !ok {
		f.order = append(f.order, m.ID)
	}
	f.mems[m.ID] = m
}

func (f *fakeStore) EnsureCollection(_ context.Context) error { return nil }

func (f *fakeStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// Apply filters in-memory.
	var matched []memory.Memory
	for _, id := range f.order {
		m := f.mems[id]
		if !matchFilters(m, opts.Filters) {
			continue
		}
		matched = append(matched, m)
	}

	// Deterministic order: sort by ID so tests don't flake.
	sort.Slice(matched, func(i, j int) bool { return matched[i].ID < matched[j].ID })

	// Simple offset = last seen ID; advance past it.
	start := 0
	if opts.Offset != "" {
		for i, m := range matched {
			if m.ID == opts.Offset {
				start = i + 1
				break
			}
		}
	}
	if start >= len(matched) {
		return nil, "", nil
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	end := start + limit
	if end > len(matched) {
		end = len(matched)
	}
	page := matched[start:end]
	var next string
	if end < len(matched) {
		next = page[len(page)-1].ID
	}
	return append([]memory.Memory(nil), page...), next, nil
}

func (f *fakeStore) Update(_ context.Context, id string, fields map[string]any) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	m, ok := f.mems[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	if tags, ok := fields["tags"].([]string); ok {
		m.Tags = append([]string(nil), tags...)
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

// matchFilters — mirrors the subset we need: type=eq, tags=in (contains-any).
func matchFilters(m memory.Memory, filters []memory.Filter) bool {
	for _, f := range filters {
		switch f.Field {
		case "type":
			if f.Op == memory.OpEq && string(m.Type) != f.Value.(string) {
				return false
			}
		case "tags":
			if f.Op == memory.OpIn {
				want := f.Value.([]string)
				found := false
				for _, w := range want {
					for _, t := range m.Tags {
						if t == w {
							found = true
							break
						}
					}
					if found {
						break
					}
				}
				if !found {
					return false
				}
			}
		}
	}
	return true
}

// ── tests ────────────────────────────────────────────────────────────────────

func mkInsight(id, source string, tags []string, md map[string]any) memory.Memory {
	return memory.Memory{
		ID:       id,
		Type:     memory.TypeInsight,
		Content:  "content of " + id,
		Source:   source,
		Tags:     tags,
		Metadata: md,
	}
}

// Happy path: candidates identified by source=system OR reflection_source_ids,
// and NOT already tagged. Others untouched.
func TestScan_Candidates(t *testing.T) {
	store := newFakeStore()

	// Candidate: source=system, no tag.
	store.add(mkInsight("a", "system", []string{"x"}, nil))
	// Candidate: source=agent, but reflection_source_ids populated, no tag.
	store.add(mkInsight("b", "agent", []string{"y"}, map[string]any{
		"reflection_source_ids": []string{"src1", "src2"},
	}))
	// NOT a candidate: source=system but tag already present.
	store.add(mkInsight("c", "system", []string{"source:reflection", "z"}, nil))
	// NOT a candidate: source=agent, no metadata signal.
	store.add(mkInsight("d", "agent", []string{"q"}, nil))
	// NOT a candidate by type (filter level): event should be excluded at Scroll.
	store.add(memory.Memory{ID: "e", Type: memory.TypeEvent, Source: "system"})

	cs, scanned, already, err := scan(context.Background(), store)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}

	// Scrolled: a,b,c,d  (e filtered by type). Candidates: a,b. Already tagged: c.
	if scanned != 4 {
		t.Errorf("scanned = %d, want 4", scanned)
	}
	if already != 1 {
		t.Errorf("already_tagged = %d, want 1", already)
	}
	ids := []string{}
	for _, c := range cs {
		ids = append(ids, c.ID)
	}
	sort.Strings(ids)
	if strings.Join(ids, ",") != "a,b" {
		t.Errorf("candidate ids = %v, want [a b]", ids)
	}
}

// Patch adds the tag exactly once; preserves other tags; idempotent on re-run.
func TestPatch_AddsTagAndIsIdempotent(t *testing.T) {
	store := newFakeStore()
	store.add(mkInsight("a", "system", []string{"x"}, nil))
	store.add(mkInsight("b", "system", []string{"y", "z"}, nil))

	// First run.
	res, err := run(context.Background(), store, options{dryRun: false, batchSize: 10})
	if err != nil {
		t.Fatalf("run1: %v", err)
	}
	if res.Patched != 2 {
		t.Fatalf("patched = %d, want 2", res.Patched)
	}

	// Check tag layout.
	ms, _ := store.SearchByIDs(context.Background(), []string{"a", "b"})
	for _, m := range ms {
		count := 0
		for _, tg := range m.Tags {
			if tg == sourceReflectionTag {
				count++
			}
		}
		if count != 1 {
			t.Errorf("%s: tag appears %d times, want 1 (tags=%v)", m.ID, count, m.Tags)
		}
	}
	// Verify original tags preserved.
	for _, m := range ms {
		if m.ID == "a" && !containsTag(m.Tags, "x") {
			t.Errorf("a lost original tag 'x': %v", m.Tags)
		}
		if m.ID == "b" && (!containsTag(m.Tags, "y") || !containsTag(m.Tags, "z")) {
			t.Errorf("b lost original tags y/z: %v", m.Tags)
		}
	}

	// Second run: should be a no-op (idempotent).
	res2, err := run(context.Background(), store, options{dryRun: false, batchSize: 10})
	if err != nil {
		t.Fatalf("run2: %v", err)
	}
	if res2.CandidateCount != 0 {
		t.Errorf("2nd run candidate_count = %d, want 0 (idempotent)", res2.CandidateCount)
	}
	if res2.Patched != 0 {
		t.Errorf("2nd run patched = %d, want 0", res2.Patched)
	}
}

// DryRun path does not mutate store.
func TestRun_DryRun_NoMutation(t *testing.T) {
	store := newFakeStore()
	store.add(mkInsight("a", "system", []string{"x"}, nil))
	store.add(mkInsight("b", "system", []string{"y"}, map[string]any{
		"reflection_source_ids": []string{"s1"},
	}))

	res, err := run(context.Background(), store, options{dryRun: true, batchSize: 10})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.CandidateCount != 2 {
		t.Errorf("candidate_count = %d, want 2", res.CandidateCount)
	}
	if res.Patched != 0 {
		t.Errorf("patched in dry-run = %d, want 0", res.Patched)
	}

	// Tags unchanged.
	ms, _ := store.SearchByIDs(context.Background(), []string{"a", "b"})
	for _, m := range ms {
		if containsTag(m.Tags, sourceReflectionTag) {
			t.Errorf("%s got tagged in dry-run: %v", m.ID, m.Tags)
		}
	}
}

// VERIFY: pre + post count + delta assertion passes on happy path.
func TestRun_Verify_Passes(t *testing.T) {
	store := newFakeStore()
	// 1 already tagged (pre=1), 2 candidates, expected delta=2.
	store.add(mkInsight("pre", "system", []string{"source:reflection"}, nil))
	store.add(mkInsight("a", "system", []string{}, nil))
	store.add(mkInsight("b", "agent", nil, map[string]any{
		"reflection_source_ids": []string{"s1"},
	}))

	res, err := run(context.Background(), store, options{dryRun: false, batchSize: 10, verify: true})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.VerifyPreCount != 1 {
		t.Errorf("pre = %d, want 1", res.VerifyPreCount)
	}
	if res.VerifyPostCount != 3 {
		t.Errorf("post = %d, want 3", res.VerifyPostCount)
	}
	if !res.VerifyPassed {
		t.Errorf("verify_passed = false; res=%+v", res)
	}
}

// appendTag contract: adds iff absent, never mutates input.
func TestAppendTag(t *testing.T) {
	in := []string{"a", "b"}
	out := appendTag(in, "c")
	if len(in) != 2 {
		t.Errorf("input mutated: %v", in)
	}
	if len(out) != 3 || out[2] != "c" {
		t.Errorf("out = %v, want [a b c]", out)
	}

	out2 := appendTag([]string{"a", "c"}, "c")
	if len(out2) != 2 {
		t.Errorf("duplicate added: %v", out2)
	}
}

// hasReflectionSourceIDs recognises the common metadata shapes.
func TestHasReflectionSourceIDs(t *testing.T) {
	cases := []struct {
		name string
		md   map[string]any
		want bool
	}{
		{"nil", nil, false},
		{"missing", map[string]any{"x": 1}, false},
		{"empty_slice", map[string]any{"reflection_source_ids": []any{}}, false},
		{"populated_any", map[string]any{"reflection_source_ids": []any{"s1"}}, true},
		{"populated_string_slice", map[string]any{"reflection_source_ids": []string{"s1"}}, true},
		{"empty_string", map[string]any{"reflection_source_ids": ""}, false},
		{"nil_value", map[string]any{"reflection_source_ids": nil}, false},
	}
	for _, c := range cases {
		if got := hasReflectionSourceIDs(c.md); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}
