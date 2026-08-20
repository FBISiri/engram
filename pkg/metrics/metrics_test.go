package metrics

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// fakeEmbedCache is a hermetic stand-in for memory.EmbedCache with fixed stats.
type fakeEmbedCache struct {
	hits, misses int64
}

func (f *fakeEmbedCache) Get(string) ([]float32, bool) { return nil, false }
func (f *fakeEmbedCache) Put(string, []float32)        {}
func (f *fakeEmbedCache) Stats() (int64, int64)        { return f.hits, f.misses }

// collect drains a Collector into a slice of dto.Metric for value inspection,
// avoiding prometheus/testutil (whose godebug dep is not in go.mod).
func collect(t *testing.T, c prometheus.Collector) []*dto.Metric {
	t.Helper()
	ch := make(chan prometheus.Metric, 32)
	c.Collect(ch)
	close(ch)
	var out []*dto.Metric
	for m := range ch {
		var dm dto.Metric
		if err := m.Write(&dm); err != nil {
			t.Fatalf("metric.Write: %v", err)
		}
		out = append(out, &dm)
	}
	return out
}

func descCount(c prometheus.Collector) int {
	ch := make(chan *prometheus.Desc, 32)
	c.Describe(ch)
	close(ch)
	n := 0
	for range ch {
		n++
	}
	return n
}

func TestNew_RegistersBaseHistograms(t *testing.T) {
	m := New(nil, nil)
	if m == nil || m.Registry == nil {
		t.Fatal("New returned nil registry")
	}
	if m.SearchDuration == nil || m.EmbedDuration == nil {
		t.Fatal("histograms not initialized")
	}
	m.SearchDuration.Observe(0.01)
	m.EmbedDuration.Observe(0.02)

	names := gatherNames(t, m.Registry)
	for _, want := range []string{"engram_search_duration_seconds", "engram_embed_duration_seconds"} {
		if !names[want] {
			t.Errorf("missing metric family %q", want)
		}
	}
	// CounterVecs emit no family via Gather until a child series exists, so
	// prove registration by asserting the fields are wired and that a duplicate
	// registration is rejected with AlreadyRegisteredError.
	if m.ReflectionRuns == nil || m.ReflectionInsightsCreated == nil {
		t.Fatal("reflection counters not initialized")
	}
	for _, c := range []prometheus.Collector{m.ReflectionRuns, m.ReflectionInsightsCreated} {
		err := m.Registry.Register(c)
		if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
			t.Errorf("expected AlreadyRegisteredError re-registering reflection counter, got %v", err)
		}
	}
	// Without an embed cache or stats fn, the optional families must be absent.
	if names["engram_embed_cache_hit_total"] {
		t.Error("embed cache counters registered without a cache")
	}
	if names["engram_memory_count"] {
		t.Error("memory count gauge registered without a stats fn")
	}
}

func gatherNames(t *testing.T, reg *prometheus.Registry) map[string]bool {
	t.Helper()
	fams, err := reg.Gather()
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	names := map[string]bool{}
	for _, f := range fams {
		names[f.GetName()] = true
	}
	return names
}

func TestEmbedCacheCollector(t *testing.T) {
	c := newEmbedCacheCollector(&fakeEmbedCache{hits: 7, misses: 3})

	if n := descCount(c); n != 2 {
		t.Fatalf("Describe emitted %d descs, want 2", n)
	}

	metrics := collect(t, c)
	if len(metrics) != 2 {
		t.Fatalf("Collect emitted %d metrics, want 2", len(metrics))
	}
	// Order is Describe order: hits then misses.
	if got := metrics[0].GetCounter().GetValue(); got != 7 {
		t.Errorf("hit counter = %v, want 7", got)
	}
	if got := metrics[1].GetCounter().GetValue(); got != 3 {
		t.Errorf("miss counter = %v, want 3", got)
	}
}

func TestMemoryCountCollector(t *testing.T) {
	fn := func(context.Context) map[string]uint64 {
		return map[string]uint64{"mem_default": 42, "mem_scratch": 5}
	}
	c := newMemoryCountCollector(fn)

	if n := descCount(c); n != 1 {
		t.Fatalf("Describe emitted %d descs, want 1", n)
	}

	metrics := collect(t, c)
	if len(metrics) != 2 {
		t.Fatalf("Collect emitted %d metrics, want 2", len(metrics))
	}
	// Map keyed by collection label → gauge value (Collect order is map-random).
	got := map[string]float64{}
	for _, m := range metrics {
		var col string
		for _, lp := range m.GetLabel() {
			if lp.GetName() == "collection" {
				col = lp.GetValue()
			}
		}
		got[col] = m.GetGauge().GetValue()
	}
	if got["mem_default"] != 42 {
		t.Errorf("mem_default = %v, want 42", got["mem_default"])
	}
	if got["mem_scratch"] != 5 {
		t.Errorf("mem_scratch = %v, want 5", got["mem_scratch"])
	}
}

func TestReflectionCounters(t *testing.T) {
	m := New(nil, nil)
	if m.ReflectionRuns == nil || m.ReflectionInsightsCreated == nil {
		t.Fatal("reflection counters not initialized")
	}

	m.ReflectionRuns.WithLabelValues("v1-flat", "default").Inc()
	m.ReflectionInsightsCreated.WithLabelValues("v1-flat", "high").Add(3)

	var runs dto.Metric
	if err := m.ReflectionRuns.WithLabelValues("v1-flat", "default").Write(&runs); err != nil {
		t.Fatalf("write runs: %v", err)
	}
	if got := runs.GetCounter().GetValue(); got != 1 {
		t.Errorf("reflection_runs = %v, want 1", got)
	}

	var ins dto.Metric
	if err := m.ReflectionInsightsCreated.WithLabelValues("v1-flat", "high").Write(&ins); err != nil {
		t.Fatalf("write insights: %v", err)
	}
	if got := ins.GetCounter().GetValue(); got != 3 {
		t.Errorf("reflection_insights_created(high) = %v, want 3", got)
	}
}

func TestMemoryOpsAndDedupHitsCounters(t *testing.T) {
	m := New(nil, nil)
	if m.MemoryOps == nil || m.DedupHits == nil {
		t.Fatal("memory ops / dedup hits counters not initialized")
	}

	m.MemoryOps.WithLabelValues("add", "mem_default", "user_input").Inc()
	m.MemoryOps.WithLabelValues("add", "mem_default", "user_input").Inc()
	m.MemoryOps.WithLabelValues("update", "mem_default", "reflection").Inc()
	m.MemoryOps.WithLabelValues("delete", "mem_default", "unknown").Inc()
	m.DedupHits.WithLabelValues("mem_default", "server_side_092").Add(4)

	var add dto.Metric
	if err := m.MemoryOps.WithLabelValues("add", "mem_default", "user_input").Write(&add); err != nil {
		t.Fatalf("write add: %v", err)
	}
	if got := add.GetCounter().GetValue(); got != 2 {
		t.Errorf("memory_ops(add) = %v, want 2", got)
	}

	var upd dto.Metric
	if err := m.MemoryOps.WithLabelValues("update", "mem_default", "reflection").Write(&upd); err != nil {
		t.Fatalf("write update: %v", err)
	}
	if got := upd.GetCounter().GetValue(); got != 1 {
		t.Errorf("memory_ops(update) = %v, want 1", got)
	}

	var del dto.Metric
	if err := m.MemoryOps.WithLabelValues("delete", "mem_default", "unknown").Write(&del); err != nil {
		t.Fatalf("write delete: %v", err)
	}
	if got := del.GetCounter().GetValue(); got != 1 {
		t.Errorf("memory_ops(delete) = %v, want 1", got)
	}

	var dedup dto.Metric
	if err := m.DedupHits.WithLabelValues("mem_default", "server_side_092").Write(&dedup); err != nil {
		t.Fatalf("write dedup: %v", err)
	}
	if got := dedup.GetCounter().GetValue(); got != 4 {
		t.Errorf("dedup_hits(server_side_092) = %v, want 4", got)
	}

	// Registration proof: re-registering must be rejected.
	for _, c := range []prometheus.Collector{m.MemoryOps, m.DedupHits} {
		err := m.Registry.Register(c)
		if _, ok := err.(prometheus.AlreadyRegisteredError); !ok {
			t.Errorf("expected AlreadyRegisteredError re-registering counter, got %v", err)
		}
	}
}

func TestNew_RegistersOptionalCollectors(t *testing.T) {
	fn := func(context.Context) map[string]uint64 { return map[string]uint64{"c": 1} }
	m := New(&fakeEmbedCache{hits: 1, misses: 2}, fn)

	names := gatherNames(t, m.Registry)
	for _, want := range []string{
		"engram_embed_cache_hit_total",
		"engram_embed_cache_miss_total",
		"engram_memory_count",
	} {
		if !names[want] {
			t.Errorf("optional collector metric %q not registered", want)
		}
	}
}

func TestMemoryCountCollector_EmptyStats(t *testing.T) {
	// A stats fn returning no collections must not emit any metric or panic.
	c := newMemoryCountCollector(func(context.Context) map[string]uint64 { return nil })
	if metrics := collect(t, c); len(metrics) != 0 {
		t.Errorf("expected 0 metrics for empty stats, got %d", len(metrics))
	}
}
