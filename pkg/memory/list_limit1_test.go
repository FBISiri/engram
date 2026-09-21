package memory

import (
	"context"
	"reflect"
	"testing"
)

// recordingScrollStore embeds inclusivePagedStore (which models Qdrant's
// INCLUSIVE-offset scroll with a small per-page cap) and records the per-page
// fetch size that ListMemories requests. It exists to lock in the R3 fix: the
// internal page size MUST be decoupled from the caller's result limit
// (scrollPageSize). If they were coupled, a limit=1 request would fetch one
// point per page, re-emit the inclusive boundary, and advance by zero — either
// looping forever or dropping every result but one.
type recordingScrollStore struct {
	*inclusivePagedStore
	pageSizes *[]int
}

func (r recordingScrollStore) Scroll(ctx context.Context, opts ScrollOptions) ([]Memory, string, error) {
	*r.pageSizes = append(*r.pageSizes, opts.Limit)
	return r.inclusivePagedStore.Scroll(ctx, opts)
}

// TestListMemories_Limit1_DecouplesPageSize_R3 is the R3 regression test. It
// proves that even at a tiny result limit (including the pathological limit=1),
// ListMemories (a) terminates, (b) returns the correct newest-N window, and
// (c) always fetches with the fixed scrollPageSize rather than the caller's
// limit. pageCap=2 forces genuine multi-page pagination so the loop-advance
// path is exercised.
func TestListMemories_Limit1_DecouplesPageSize_R3(t *testing.T) {
	items := []Memory{
		{ID: "a", CreatedAt: 100},
		{ID: "b", CreatedAt: 200},
		{ID: "c", CreatedAt: 300},
		{ID: "d", CreatedAt: 400},
		{ID: "e", CreatedAt: 500},
	}
	cases := []struct {
		limit int
		want  []string // newest-N, ascending display order
	}{
		{limit: 1, want: []string{"e"}},
		{limit: 2, want: []string{"d", "e"}},
		{limit: 3, want: []string{"c", "d", "e"}},
	}
	for _, tc := range cases {
		var pageSizes []int
		base := &inclusivePagedStore{items: items, pageCap: 2} // force multi-page
		store := recordingScrollStore{inclusivePagedStore: base, pageSizes: &pageSizes}

		got, err := ListMemories(context.Background(), store, ListMemoriesOptions{Limit: tc.limit})
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
		if len(pageSizes) == 0 {
			t.Fatalf("limit=%d: expected at least one Scroll call", tc.limit)
		}
		// Decoupling invariant: never fetch with the caller's tiny limit.
		for _, ps := range pageSizes {
			if ps != scrollPageSize {
				t.Fatalf("limit=%d: Scroll requested page size %d, want decoupled scrollPageSize=%d",
					tc.limit, ps, scrollPageSize)
			}
		}
	}
}
