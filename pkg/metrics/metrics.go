// Package metrics defines the Prometheus metrics registry and metric objects
// for the Engram HTTP server. Metrics are registered into an isolated
// prometheus.Registry (not the global default) to keep tests hermetic.
package metrics

import (
	"context"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/prometheus/client_golang/prometheus"
)

// Metrics holds the prometheus Registry and pre-registered metric objects for
// hot-path observations in server handlers.
type Metrics struct {
	Registry       *prometheus.Registry
	SearchDuration prometheus.Histogram // engram_search_duration_seconds
	EmbedDuration  prometheus.Histogram // engram_embed_duration_seconds
}

// New creates a Metrics instance and registers all metrics into a fresh Registry.
//   - embedCache, if non-nil, registers embed cache hit/miss counters via a Collector.
//   - collectionStatsFn, if non-nil, registers a per-collection memory count Gauge
//     Collector that calls the function at scrape time.
func New(embedCache memory.EmbedCache, collectionStatsFn func(context.Context) map[string]uint64) *Metrics {
	reg := prometheus.NewRegistry()

	searchDur := prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "engram_search_duration_seconds",
		Help:    "Duration of memory_search operations from request to response.",
		Buckets: prometheus.DefBuckets,
	})
	embedDur := prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "engram_embed_duration_seconds",
		Help:    "Duration of text embedding API calls.",
		Buckets: prometheus.DefBuckets,
	})
	reg.MustRegister(searchDur, embedDur)

	if embedCache != nil {
		reg.MustRegister(newEmbedCacheCollector(embedCache))
	}
	if collectionStatsFn != nil {
		reg.MustRegister(newMemoryCountCollector(collectionStatsFn))
	}

	return &Metrics{
		Registry:       reg,
		SearchDuration: searchDur,
		EmbedDuration:  embedDur,
	}
}

// ─────────────────────────────────────────────────────────────
// embedCacheCollector — reads atomic counters from EmbedCache.Stats()
// ─────────────────────────────────────────────────────────────

type embedCacheCollector struct {
	cache      memory.EmbedCache
	hitsDesc   *prometheus.Desc
	missesDesc *prometheus.Desc
}

func newEmbedCacheCollector(c memory.EmbedCache) *embedCacheCollector {
	return &embedCacheCollector{
		cache: c,
		hitsDesc: prometheus.NewDesc(
			"engram_embed_cache_hit_total",
			"Total embed cache hits.",
			nil, nil,
		),
		missesDesc: prometheus.NewDesc(
			"engram_embed_cache_miss_total",
			"Total embed cache misses.",
			nil, nil,
		),
	}
}

func (c *embedCacheCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.hitsDesc
	ch <- c.missesDesc
}

func (c *embedCacheCollector) Collect(ch chan<- prometheus.Metric) {
	hits, misses := c.cache.Stats()
	ch <- prometheus.MustNewConstMetric(c.hitsDesc, prometheus.CounterValue, float64(hits))
	ch <- prometheus.MustNewConstMetric(c.missesDesc, prometheus.CounterValue, float64(misses))
}

// ─────────────────────────────────────────────────────────────
// memoryCountCollector — queries per-collection point counts at scrape time
// ─────────────────────────────────────────────────────────────

type memoryCountCollector struct {
	statsFn func(context.Context) map[string]uint64
	desc    *prometheus.Desc
}

func newMemoryCountCollector(fn func(context.Context) map[string]uint64) *memoryCountCollector {
	return &memoryCountCollector{
		statsFn: fn,
		desc: prometheus.NewDesc(
			"engram_memory_count",
			"Number of memories per Qdrant collection.",
			[]string{"collection"}, nil,
		),
	}
}

func (c *memoryCountCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- c.desc
}

func (c *memoryCountCollector) Collect(ch chan<- prometheus.Metric) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for col, count := range c.statsFn(ctx) {
		ch <- prometheus.MustNewConstMetric(c.desc, prometheus.GaugeValue, float64(count), col)
	}
}
