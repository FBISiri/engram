package memory

import (
	"context"
	"reflect"
	"sort"
	"testing"
)

func TestBuildTimeWindowScrollOptions(t *testing.T) {
	tests := []struct {
		name string
		in   ListMemoriesOptions
		want ScrollOptions
	}{
		{
			name: "empty defaults to limit 50, no filters",
			in:   ListMemoriesOptions{},
			want: ScrollOptions{Limit: DefaultListLimit},
		},
		{
			name: "time_start only -> gte filter",
			in:   ListMemoriesOptions{TimeStart: 100, Limit: 10},
			want: ScrollOptions{
				Limit:   10,
				Filters: []Filter{{Field: FieldCreatedAt, Op: OpGte, Value: 100.0}},
			},
		},
		{
			name: "time_end only -> lte filter",
			in:   ListMemoriesOptions{TimeEnd: 200, Limit: 10},
			want: ScrollOptions{
				Limit:   10,
				Filters: []Filter{{Field: FieldCreatedAt, Op: OpLte, Value: 200.0}},
			},
		},
		{
			name: "full window + collections",
			in:   ListMemoriesOptions{TimeStart: 100, TimeEnd: 200, Collections: []string{"engram_user"}, Limit: 5},
			want: ScrollOptions{
				Limit: 5,
				Filters: []Filter{
					{Field: FieldCreatedAt, Op: OpGte, Value: 100.0},
					{Field: FieldCreatedAt, Op: OpLte, Value: 200.0},
					{Field: FieldCollection, Op: OpIn, Value: []string{"engram_user"}},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := BuildTimeWindowScrollOptions(tt.in)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("BuildTimeWindowScrollOptions(%+v) = %+v, want %+v", tt.in, got, tt.want)
			}
		})
	}
}

// pagedStore is a fake Store that serves Memories page-by-page via opaque
// offsets, emulating Qdrant's point-ID-ordered (NOT time-ordered) scroll.
type pagedStore struct {
	items []Memory // returned in this (non-time) order across pages
}

func (p *pagedStore) Scroll(_ context.Context, opts ScrollOptions) ([]Memory, string, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	start := 0
	if opts.Offset != "" {
		for i, m := range p.items {
			if m.ID == opts.Offset {
				start = i + 1
				break
			}
		}
	}
	end := start + limit
	if end > len(p.items) {
		end = len(p.items)
	}
	page := p.items[start:end]
	var next string
	if len(page) == limit && end < len(p.items) {
		next = p.items[end-1].ID
	}
	return page, next, nil
}

// Unused Store methods.
func (p *pagedStore) Insert(context.Context, *Memory, []float32) error { return nil }
func (p *pagedStore) Search(context.Context, []float32, SearchOptions) ([]ScoredMemory, error) {
	return nil, nil
}
func (p *pagedStore) Delete(context.Context, []string) (int, error)           { return 0, nil }
func (p *pagedStore) Update(context.Context, string, map[string]any) error    { return nil }
func (p *pagedStore) SearchByIDs(context.Context, []string) ([]Memory, error) { return nil, nil }
func (p *pagedStore) EnsureCollection(context.Context) error                  { return nil }
func (p *pagedStore) Stats(context.Context) (*CollectionStats, error)         { return nil, nil }
func (p *pagedStore) DeleteExpired(context.Context) (int, error)              { return 0, nil }

func TestListMemories_PaginatesAndReturnsNewestN(t *testing.T) {
	// 5 items served in point-ID (non-time) order across pages of size 2.
	// created_at intentionally out of order to prove time-sorting.
	store := &pagedStore{items: []Memory{
		{ID: "e", CreatedAt: 50},
		{ID: "a", CreatedAt: 10},
		{ID: "c", CreatedAt: 30},
		{ID: "b", CreatedAt: 20},
		{ID: "d", CreatedAt: 40},
	}}

	got, err := ListMemories(context.Background(), store, ListMemoriesOptions{Limit: 3})
	if err != nil {
		t.Fatalf("ListMemories error: %v", err)
	}
	// Newest 3 by created_at, returned ascending: c(30), d(40), e(50).
	wantIDs := []string{"c", "d", "e"}
	gotIDs := make([]string, len(got))
	for i, m := range got {
		gotIDs[i] = m.ID
	}
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("newest-N mismatch: got %v, want %v", gotIDs, wantIDs)
	}
}

func TestListMemories_TiebreakByID(t *testing.T) {
	// All equal created_at → deterministic ID ascending order.
	store := &pagedStore{items: []Memory{
		{ID: "c3", CreatedAt: 100},
		{ID: "c1", CreatedAt: 100},
		{ID: "c2", CreatedAt: 100},
	}}
	got, err := ListMemories(context.Background(), store, ListMemoriesOptions{Limit: 10})
	if err != nil {
		t.Fatalf("ListMemories error: %v", err)
	}
	gotIDs := make([]string, len(got))
	for i, m := range got {
		gotIDs[i] = m.ID
	}
	want := []string{"c1", "c2", "c3"}
	if !sort.StringsAreSorted(gotIDs) || !reflect.DeepEqual(gotIDs, want) {
		t.Fatalf("tiebreak mismatch: got %v, want %v", gotIDs, want)
	}
}

// inclusivePagedStore models Qdrant scroll offset as INCLUSIVE of the boundary
// point ID: each new page re-emits the point named by the offset, so callers
// must dedup by ID and terminate on no-new-IDs.
type inclusivePagedStore struct {
	items   []Memory
	pageCap int // if >0, caps per-page size regardless of opts.Limit (forces multi-page)
}

func (p *inclusivePagedStore) Scroll(_ context.Context, opts ScrollOptions) ([]Memory, string, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}
	if p.pageCap > 0 && limit > p.pageCap {
		limit = p.pageCap
	}
	start := 0
	if opts.Offset != "" {
		for i, m := range p.items {
			if m.ID == opts.Offset {
				start = i // INCLUSIVE: re-include the boundary point.
				break
			}
		}
	}
	end := start + limit
	if end > len(p.items) {
		end = len(p.items)
	}
	page := p.items[start:end]
	var next string
	if end < len(p.items) {
		next = p.items[end-1].ID
	}
	return page, next, nil
}

func (p *inclusivePagedStore) Insert(context.Context, *Memory, []float32) error { return nil }
func (p *inclusivePagedStore) Search(context.Context, []float32, SearchOptions) ([]ScoredMemory, error) {
	return nil, nil
}
func (p *inclusivePagedStore) Delete(context.Context, []string) (int, error)        { return 0, nil }
func (p *inclusivePagedStore) Update(context.Context, string, map[string]any) error { return nil }
func (p *inclusivePagedStore) SearchByIDs(context.Context, []string) ([]Memory, error) {
	return nil, nil
}
func (p *inclusivePagedStore) EnsureCollection(context.Context) error          { return nil }
func (p *inclusivePagedStore) Stats(context.Context) (*CollectionStats, error) { return nil, nil }
func (p *inclusivePagedStore) DeleteExpired(context.Context) (int, error)      { return 0, nil }

func TestListMemories_InclusiveOffsetNoDuplicates(t *testing.T) {
	store := &inclusivePagedStore{items: []Memory{
		{ID: "e", CreatedAt: 50},
		{ID: "a", CreatedAt: 10},
		{ID: "c", CreatedAt: 30},
		{ID: "b", CreatedAt: 20},
		{ID: "d", CreatedAt: 40},
	}}

	got, err := ListMemories(context.Background(), store, ListMemoriesOptions{Limit: 10})
	if err != nil {
		t.Fatalf("ListMemories error: %v", err)
	}

	// No duplicate IDs despite inclusive-offset boundary re-emission.
	seen := map[string]bool{}
	for _, m := range got {
		if seen[m.ID] {
			t.Fatalf("duplicate ID %q in result", m.ID)
		}
		seen[m.ID] = true
	}
	// All 5 unique (limit=10 > window), sorted ascending.
	gotIDs := make([]string, len(got))
	for i, m := range got {
		gotIDs[i] = m.ID
	}
	want := []string{"a", "b", "c", "d", "e"}
	if !reflect.DeepEqual(gotIDs, want) {
		t.Fatalf("ascending order mismatch: got %v, want %v", gotIDs, want)
	}
}

func TestListMemories_SmallLimitReturnsNewest_InclusiveMultiPage(t *testing.T) {
	// Window has 5 matches; a small page cap (2) + inclusive offset forces
	// multi-page pagination. Regardless of a tiny result limit, the result
	// must be the true NEWEST-N by created_at (returned ascending), not an
	// arbitrary point-ID page.
	newStore := func() *inclusivePagedStore {
		return &inclusivePagedStore{
			pageCap: 2,
			items: []Memory{
				{ID: "e", CreatedAt: 50},
				{ID: "a", CreatedAt: 10},
				{ID: "c", CreatedAt: 30},
				{ID: "b", CreatedAt: 20},
				{ID: "d", CreatedAt: 40},
			},
		}
	}

	cases := []struct {
		limit int
		want  []string
	}{
		{limit: 1, want: []string{"e"}},
		{limit: 2, want: []string{"d", "e"}},
	}
	for _, tc := range cases {
		got, err := ListMemories(context.Background(), newStore(), ListMemoriesOptions{Limit: tc.limit})
		if err != nil {
			t.Fatalf("limit=%d: ListMemories error: %v", tc.limit, err)
		}
		gotIDs := make([]string, len(got))
		for i, m := range got {
			gotIDs[i] = m.ID
		}
		if !reflect.DeepEqual(gotIDs, tc.want) {
			t.Fatalf("limit=%d: got %v, want newest-N %v", tc.limit, gotIDs, tc.want)
		}
	}
}

// filterStore is a fake Store that applies created_at/collection/type/tags
// filters and the IncludeSuperseded semantics, for ListSuperseded tests.
type filterStore struct {
	items []Memory
}

func (f *filterStore) Scroll(_ context.Context, opts ScrollOptions) ([]Memory, string, error) {
	var out []Memory
	for _, m := range f.items {
		if !opts.IncludeSuperseded && m.SupersededBy != "" {
			continue
		}
		if !fsMatch(m, opts.Filters) {
			continue
		}
		out = append(out, m)
	}
	return out, "", nil
}

func fsMatch(m Memory, filters []Filter) bool {
	for _, fl := range filters {
		switch fl.Field {
		case FieldCreatedAt:
			switch fl.Op {
			case OpGte:
				if m.CreatedAt < fl.Value.(float64) {
					return false
				}
			case OpLte:
				if m.CreatedAt > fl.Value.(float64) {
					return false
				}
			}
		case FieldCollection:
			if fl.Op == OpIn && !fsInAny([]string{m.Collection}, fl.Value.([]string)) {
				return false
			}
		case "type":
			if fl.Op == OpIn && !fsInAny([]string{string(m.Type)}, fl.Value.([]string)) {
				return false
			}
		case "tags":
			if fl.Op == OpIn && !fsInAny(m.Tags, fl.Value.([]string)) {
				return false
			}
		}
	}
	return true
}

func fsInAny(have, want []string) bool {
	for _, w := range want {
		for _, h := range have {
			if h == w {
				return true
			}
		}
	}
	return false
}

func (f *filterStore) Insert(context.Context, *Memory, []float32) error { return nil }
func (f *filterStore) Search(context.Context, []float32, SearchOptions) ([]ScoredMemory, error) {
	return nil, nil
}
func (f *filterStore) Delete(context.Context, []string) (int, error)           { return 0, nil }
func (f *filterStore) Update(context.Context, string, map[string]any) error    { return nil }
func (f *filterStore) SearchByIDs(context.Context, []string) ([]Memory, error) { return nil, nil }
func (f *filterStore) EnsureCollection(context.Context) error                  { return nil }
func (f *filterStore) Stats(context.Context) (*CollectionStats, error)         { return nil, nil }
func (f *filterStore) DeleteExpired(context.Context) (int, error)              { return 0, nil }

func lsIDs(mems []Memory) []string {
	ids := make([]string, len(mems))
	for i, m := range mems {
		ids[i] = m.ID
	}
	return ids
}

func lsContains(ss []string, want string) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func TestListSuperseded(t *testing.T) {
	store := &filterStore{items: []Memory{
		{ID: "A", Type: TypeEvent, Tags: []string{"x"}, CreatedAt: 100, Collection: "engram_user", SupersededBy: "B"},
		{ID: "C", Type: TypeInsight, Tags: []string{"y"}, CreatedAt: 200, Collection: "engram_user", SupersededBy: "D"},
		{ID: "E", Type: TypeEvent, Tags: []string{"x"}, CreatedAt: 300, Collection: "engram_reflection", SupersededBy: "F"},
		{ID: "N", Type: TypeEvent, Tags: []string{"x"}, CreatedAt: 150, Collection: "engram_user"}, // not superseded
	}}
	ctx := context.Background()

	// Only superseded returned.
	got, err := ListSuperseded(ctx, store, ListSupersededOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ids := lsIDs(got)
	if lsContains(ids, "N") {
		t.Errorf("must exclude non-superseded N; ids=%v", ids)
	}
	for _, want := range []string{"A", "C", "E"} {
		if !lsContains(ids, want) {
			t.Errorf("want %s in %v", want, ids)
		}
	}

	// Type filter narrows.
	got, _ = ListSuperseded(ctx, store, ListSupersededOptions{Types: []string{"insight"}})
	if ids = lsIDs(got); len(ids) != 1 || ids[0] != "C" {
		t.Errorf("type filter: want [C], got %v", ids)
	}

	// Tag filter narrows.
	got, _ = ListSuperseded(ctx, store, ListSupersededOptions{Tags: []string{"y"}})
	if ids = lsIDs(got); len(ids) != 1 || ids[0] != "C" {
		t.Errorf("tag filter: want [C], got %v", ids)
	}

	// Collection filter narrows.
	got, _ = ListSuperseded(ctx, store, ListSupersededOptions{Collections: []string{"engram_reflection"}})
	if ids = lsIDs(got); len(ids) != 1 || ids[0] != "E" {
		t.Errorf("collection filter: want [E], got %v", ids)
	}

	// Time window narrows.
	got, _ = ListSuperseded(ctx, store, ListSupersededOptions{TimeStart: 150, TimeEnd: 250})
	if ids = lsIDs(got); len(ids) != 1 || ids[0] != "C" {
		t.Errorf("time filter: want [C], got %v", ids)
	}
}
