package server

import (
	"context"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// TestDedupTypeLabel verifies the dedup_type metric label is derived
// deterministically from the threshold actually used for the decision, so a
// non-0.92 threshold (e.g. the A-MAC per-type directive ~0.90) produces a
// DIFFERENT label than the 0.92 path.
func TestDedupTypeLabel(t *testing.T) {
	cases := []struct {
		threshold float64
		want      string
	}{
		{0.90, "server_side_090"},
		{0.92, "server_side_092"},
		{0.78, "server_side_078"},
	}
	for _, c := range cases {
		if got := dedupTypeLabel(c.threshold); got != c.want {
			t.Errorf("dedupTypeLabel(%v) = %q, want %q", c.threshold, got, c.want)
		}
	}

	// The whole point: a non-0.92 threshold must not be mislabelled as 0.92.
	if dedupTypeLabel(0.90) == dedupTypeLabel(0.92) {
		t.Error("dedupTypeLabel(0.90) must differ from dedupTypeLabel(0.92)")
	}

	// Bounded cardinality: fixed to 2 decimals (integer percent), so nearby
	// floats collapse to the same 3-digit label rather than exploding it.
	if dedupTypeLabel(0.921) != "server_side_092" {
		t.Errorf("dedupTypeLabel(0.921) = %q, want server_side_092", dedupTypeLabel(0.921))
	}
}

// TestDedupResultThresholdLabel exercises the label through DedupResult.Threshold,
// the value carried out of checkDedup to both call sites.
func TestDedupResultThresholdLabel(t *testing.T) {
	r := &DedupResult{DupFound: true, Threshold: 0.90}
	if got := dedupTypeLabel(r.Threshold); got != "server_side_090" {
		t.Errorf("label from DedupResult.Threshold = %q, want server_side_090", got)
	}
}

// TestCheckDedupThresholdCarried drives the real checkDedup path with an A-MAC
// per-type threshold (directive = 0.90, != the legacy 0.92) and asserts the
// resolved threshold is carried out on DedupResult.Threshold. This covers the
// plumbing the pure-formatter test misses: it fails if checkDedup stops setting
// Threshold or re-resolves a different value.
func TestCheckDedupThresholdCarried(t *testing.T) {
	srv, store := newAMACServer()
	ctx := context.Background()

	content := "always confirm before deleting production data"
	vec, err := srv.embedder.Embed(ctx, content)
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	mem := memory.New(content)
	mem.Collection = CollectionFromContext(ctx)
	if err := store.Insert(ctx, mem, vec); err != nil {
		t.Fatalf("insert: %v", err)
	}

	dr, err := srv.checkDedup(ctx, vec, content, "user_input", memory.TypeDirective)
	if err != nil {
		t.Fatalf("checkDedup: %v", err)
	}
	if !dr.DupFound {
		t.Fatal("expected DupFound=true for identical content above 0.90 threshold")
	}
	if dr.Threshold != 0.90 {
		t.Fatalf("DedupResult.Threshold = %v, want 0.90 (directive A-MAC threshold)", dr.Threshold)
	}
	if got := dedupTypeLabel(dr.Threshold); got != "server_side_090" {
		t.Errorf("label from carried threshold = %q, want server_side_090", got)
	}
}

// TestMemoryAddEmitsThresholdLabel drives the real memory_add handler (the MCP
// call site) so the dedup-hit metric is emitted by production wiring, then
// enumerates the DedupHits series. A directive add hits the 0.90 A-MAC
// threshold, so the emitted dedup_type MUST be "server_side_090" and NOT the
// old hardcoded "server_side_092". This assertion fails under the old code.
func TestMemoryAddEmitsThresholdLabel(t *testing.T) {
	srv, _ := newAMACServer()
	m := engrammetrics.New(nil, nil)
	srv.SetMetrics(m)

	// First add: admitted. Second identical add: dedup_rejected -> emits metric.
	for i := 0; i < 2; i++ {
		if _, err := callTool(srv, "memory_add", map[string]any{
			"content":     "never overwrite the production database without a backup",
			"type":        "directive",
			"source_type": "user_input",
		}); err != nil {
			t.Fatalf("add %d: %v", i, err)
		}
	}

	labels := collectDedupTypeLabels(t, m.DedupHits)
	if len(labels) != 1 {
		t.Fatalf("expected exactly one emitted dedup_type series, got %v", labels)
	}
	if got := labels["server_side_090"]; got != 1 {
		t.Errorf("dedup_hits{dedup_type=server_side_090} = %v, want 1", got)
	}
	if _, ok := labels["server_side_092"]; ok {
		t.Errorf("old hardcoded label server_side_092 must NOT be emitted for a 0.90 threshold; got series %v", labels)
	}
}

// collectDedupTypeLabels drains a CounterVec and returns dedup_type -> value for
// every child series that actually exists (WithLabelValues children only appear
// once incremented).
func collectDedupTypeLabels(t *testing.T, cv *prometheus.CounterVec) map[string]float64 {
	t.Helper()
	ch := make(chan prometheus.Metric, 32)
	cv.Collect(ch)
	close(ch)
	out := make(map[string]float64)
	for metric := range ch {
		var dm dto.Metric
		if err := metric.Write(&dm); err != nil {
			t.Fatalf("write metric: %v", err)
		}
		for _, lp := range dm.GetLabel() {
			if lp.GetName() == "dedup_type" {
				out[lp.GetValue()] = dm.GetCounter().GetValue()
			}
		}
	}
	return out
}
