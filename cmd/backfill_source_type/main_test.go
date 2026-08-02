package main

import (
	"context"
	"os"
	"sort"
	"sync"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── fake store ───────────────────────────────────────────────────────────────

type fakeStore struct {
	mu    sync.Mutex
	mems  map[string]memory.Memory
	order []string
}

func newFakeStore() *fakeStore { return &fakeStore{mems: make(map[string]memory.Memory)} }

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

	ids := append([]string(nil), f.order...)
	sort.Strings(ids)

	start := 0
	if opts.Offset != "" {
		for i, id := range ids {
			if id == opts.Offset {
				start = i + 1
				break
			}
		}
	}
	if start >= len(ids) {
		return nil, "", nil
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	end := start + limit
	if end > len(ids) {
		end = len(ids)
	}
	var page []memory.Memory
	for _, id := range ids[start:end] {
		page = append(page, f.mems[id])
	}
	var next string
	if end < len(ids) {
		next = ids[end-1]
	}
	return page, next, nil
}

func (f *fakeStore) Update(_ context.Context, id string, fields map[string]any) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	m := f.mems[id]
	if md, ok := fields["metadata"].(map[string]any); ok {
		m.Metadata = md
	}
	f.mems[id] = m
	return nil
}

// ── mapping logic ────────────────────────────────────────────────────────────

func TestClassifySourceType(t *testing.T) {
	cases := []struct {
		name    string
		content string
		source  string
		tags    []string
		want    string
	}{
		// Content prefixes (agent-recorded Frank directives -> user_input).
		{"content_directive", "Frank directive: always run tests", "agent", nil, string(memory.SourceTypeUserInput)},
		{"content_instructed", "Frank instructed me to be terse", "agent", nil, string(memory.SourceTypeUserInput)},
		{"content_feedback", "Frank feedback on the PR", "agent", nil, string(memory.SourceTypeUserInput)},
		{"content_lowercase", "frank told me to stop", "agent", nil, string(memory.SourceTypeUserInput)},
		{"content_cjk_zhishi", "Frank 指示：优先简洁", "agent", nil, string(memory.SourceTypeUserInput)},
		{"content_cjk_yaoqiu", "Frank 要求每次都测试", "system", nil, string(memory.SourceTypeUserInput)},
		{"content_prefers", "Frank prefers small diffs", "agent", nil, string(memory.SourceTypeUserInput)},
		// Tag-based.
		{"tag_frank_feedback", "some note", "agent", []string{"frank-feedback"}, string(memory.SourceTypeUserInput)},
		{"tag_frank_and_directive", "some note", "agent", []string{"frank", "directive"}, string(memory.SourceTypeUserInput)},
		// No false positives.
		{"normal_agent", "Learned that the cache TTL is 5 minutes", "agent", nil, string(memory.SourceTypeReflection)},
		{"mentions_frank_midtext", "The team including Frank agreed", "agent", nil, string(memory.SourceTypeReflection)},
		{"tag_frank_only", "some note", "agent", []string{"frank"}, string(memory.SourceTypeReflection)},
		{"tag_directive_only", "some note", "agent", []string{"directive"}, string(memory.SourceTypeReflection)},
		// Existing behaviour preserved.
		{"user_source", "my preference", "user", nil, string(memory.SourceTypeUserInput)},
		{"user_source_with_directive_content", "Frank directive: x", "user", nil, string(memory.SourceTypeUserInput)},
		{"system_source", "system event", "system", nil, string(memory.SourceTypeReflection)},
	}
	for _, c := range cases {
		m := memory.Memory{Content: c.content, Source: c.source, Tags: c.tags}
		if got := classifySourceType(m); got != c.want {
			t.Errorf("%s: classifySourceType = %q, want %q", c.name, got, c.want)
		}
	}
}

func TestSourceTypeForSource(t *testing.T) {
	cases := map[string]string{
		"user":    string(memory.SourceTypeUserInput),
		"agent":   string(memory.SourceTypeReflection),
		"system":  string(memory.SourceTypeReflection),
		"":        string(memory.DefaultSourceType),
		"unknown": string(memory.DefaultSourceType),
	}
	for src, want := range cases {
		if got := sourceTypeForSource(src); got != want {
			t.Errorf("sourceTypeForSource(%q) = %q, want %q", src, got, want)
		}
	}
}

func TestHasSourceType(t *testing.T) {
	cases := []struct {
		name string
		md   map[string]any
		want bool
	}{
		{"nil", nil, false},
		{"missing", map[string]any{"x": 1}, false},
		{"empty", map[string]any{"source_type": ""}, false},
		{"set", map[string]any{"source_type": "user_input"}, true},
		{"nil_value", map[string]any{"source_type": nil}, false},
		{"non_string", map[string]any{"source_type": 7}, true},
	}
	for _, c := range cases {
		if got := hasSourceType(c.md); got != c.want {
			t.Errorf("%s: got %v, want %v", c.name, got, c.want)
		}
	}
}

// ── flag parsing ─────────────────────────────────────────────────────────────

func withArgs(args []string, fn func()) {
	old := os.Args
	os.Args = append([]string{"backfill_source_type"}, args...)
	defer func() { os.Args = old }()
	fn()
}

func TestParseFlags_Defaults(t *testing.T) {
	withArgs(nil, func() {
		o := parseFlags()
		if !o.dryRun {
			t.Errorf("dry-run should default true")
		}
		if o.batchSize != 20 {
			t.Errorf("batch-size default = %d, want 20", o.batchSize)
		}
	})
}

func TestParseFlags_Apply(t *testing.T) {
	withArgs([]string{"--apply"}, func() {
		o := parseFlags()
		if o.dryRun {
			t.Errorf("--apply should force dry-run=false")
		}
	})
}

func TestParseFlags_BatchOverride(t *testing.T) {
	withArgs([]string{"--batch-size", "5", "--dry-run=false"}, func() {
		o := parseFlags()
		if o.batchSize != 5 {
			t.Errorf("batch-size = %d, want 5", o.batchSize)
		}
		if o.dryRun {
			t.Errorf("dry-run should be false")
		}
	})
}

// ── run: scan + patch ────────────────────────────────────────────────────────

func mk(id, source string, md map[string]any) memory.Memory {
	return memory.Memory{ID: id, Source: source, Content: "c-" + id, Metadata: md}
}

func TestRun_DryRun_NoMutation(t *testing.T) {
	store := newFakeStore()
	store.add(mk("a", "user", nil))
	store.add(mk("b", "agent", map[string]any{"k": "v"}))
	store.add(mk("c", "system", map[string]any{"source_type": "document"})) // already set

	res, err := run(context.Background(), store, options{dryRun: true, batchSize: 10})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Scanned != 3 {
		t.Errorf("scanned = %d, want 3", res.Scanned)
	}
	if res.AlreadySet != 1 {
		t.Errorf("already_set = %d, want 1", res.AlreadySet)
	}
	if res.CandidateCount != 2 {
		t.Errorf("candidate_count = %d, want 2", res.CandidateCount)
	}
	if res.BySourceType["user_input"] != 1 || res.BySourceType["reflection"] != 1 {
		t.Errorf("by_source_type = %v", res.BySourceType)
	}
	if res.Patched != 0 {
		t.Errorf("patched in dry-run = %d, want 0", res.Patched)
	}
	// No mutation.
	if _, has := store.mems["a"].Metadata["source_type"]; has {
		t.Errorf("dry-run mutated memory a")
	}
}

func TestRun_Apply_PatchesAndPreservesMeta(t *testing.T) {
	store := newFakeStore()
	store.add(mk("a", "user", nil))
	store.add(mk("b", "agent", map[string]any{"k": "v"}))

	res, err := run(context.Background(), store, options{dryRun: false, batchSize: 10})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Patched != 2 {
		t.Fatalf("patched = %d, want 2", res.Patched)
	}
	if store.mems["a"].Metadata["source_type"] != "user_input" {
		t.Errorf("a source_type = %v, want user_input", store.mems["a"].Metadata["source_type"])
	}
	if store.mems["b"].Metadata["source_type"] != "reflection" {
		t.Errorf("b source_type = %v, want reflection", store.mems["b"].Metadata["source_type"])
	}
	if store.mems["b"].Metadata["k"] != "v" {
		t.Errorf("b lost pre-existing metadata: %v", store.mems["b"].Metadata)
	}

	// Idempotent: second run finds no candidates.
	res2, _ := run(context.Background(), store, options{dryRun: false, batchSize: 10})
	if res2.CandidateCount != 0 {
		t.Errorf("2nd run candidate_count = %d, want 0", res2.CandidateCount)
	}
}
