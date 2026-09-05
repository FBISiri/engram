package server

import (
	"testing"

	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/memory"
)

// TestSearchTopScoreMetric verifies that a search returning ≥1 result observes
// the top similarity score into engram_search_top_score for the top result's
// collection.
func TestSearchTopScoreMetric(t *testing.T) {
	srv, store := newTestServer()
	srv.SetMetrics(engrammetrics.New(nil, nil))

	mem := injectMemory(t, srv, store, "BMO is the local daemon", memory.TypeEvent, []string{"bmo"}, 0)

	if _, err := callTool(srv, "memory_search", map[string]any{"query": "daemon"}); err != nil {
		t.Fatalf("search failed: %v", err)
	}

	fams, err := srv.metrics.Registry.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	var count uint64
	found := false
	for _, f := range fams {
		if f.GetName() != "engram_search_top_score" {
			continue
		}
		for _, mm := range f.GetMetric() {
			for _, lp := range mm.GetLabel() {
				if lp.GetName() == "collection" && lp.GetValue() == mem.Collection {
					found = true
					count = mm.GetHistogram().GetSampleCount()
				}
			}
		}
	}
	if !found {
		t.Fatalf("no engram_search_top_score series for collection %q", mem.Collection)
	}
	if count != 1 {
		t.Errorf("SearchTopScore sample count for %q = %d, want 1", mem.Collection, count)
	}
}
