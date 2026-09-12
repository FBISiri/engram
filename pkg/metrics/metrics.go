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
	// ReflectionRuns counts reflection engine runs that actually triggered,
	// labelled by mode ("v1-flat"|"v2-focal") and collection.
	ReflectionRuns *prometheus.CounterVec // engram_reflection_runs_total
	// ReflectionInsightsCreated counts insights produced by reflection runs,
	// labelled by mode and confidence tier ("high"|"mid"|"low").
	ReflectionInsightsCreated *prometheus.CounterVec // engram_reflection_insights_created_total
	// MemoryOps counts memory_add/update/delete operations, labelled by
	// operation ("add"|"update"|"delete"), collection, and source_type.
	MemoryOps *prometheus.CounterVec // engram_memory_ops_total
	// DedupHits counts deduplication hits, labelled by collection and
	// dedup_type ("server_side_092"|"client_side_078").
	DedupHits *prometheus.CounterVec // engram_dedup_hits_total
	// EvaporationSweepTotal counts evaporation sweep executions.
	EvaporationSweepTotal prometheus.Counter // engram_evaporation_sweep_total
	// EvaporationDeprecatedTotal counts memories deprecated by evaporation, by type.
	EvaporationDeprecatedTotal *prometheus.CounterVec // engram_evaporation_deprecated_total
	// EvaporationSweepDuration observes sweep durations.
	EvaporationSweepDuration prometheus.Histogram // engram_evaporation_sweep_duration_seconds
	// EvaporationEffectiveImportance samples query-time effective_importance, by type.
	EvaporationEffectiveImportance *prometheus.HistogramVec // engram_evaporation_effective_importance
	// EvaporationScannedTotal counts active memories scanned by the sweep.
	EvaporationScannedTotal prometheus.Counter // engram_evaporation_scanned_total
	// EvaporationExemptedTotal counts memories exempted from evaporation, by type and protection rule (P1..P8).
	EvaporationExemptedTotal *prometheus.CounterVec // engram_evaporation_exempted_total
	// EvaporationDryRunCandidatesTotal counts below-threshold, non-exempt candidates identified by the sweep, by type.
	EvaporationDryRunCandidatesTotal *prometheus.CounterVec // engram_evaporation_dry_run_candidates_total
	// EvaporationHardDeletedTotal counts memories hard-deleted via the expiry path, by type.
	EvaporationHardDeletedTotal *prometheus.CounterVec // engram_evaporation_hard_deleted_total
	// ImportanceMean/ImportanceP90 export the CP2 importance-monitor rolling
	// statistics per collection.
	ImportanceMean *prometheus.GaugeVec // engram_importance_mean{collection}
	ImportanceP90  *prometheus.GaugeVec // engram_importance_p90{collection}
	// SearchTopScore samples the top similarity score of vector searches, by collection.
	SearchTopScore *prometheus.HistogramVec // engram_search_top_score{collection}
	// AdmissionTotal counts admission-flow outcomes for memory writes, labelled
	// by memory type and admission decision
	// (admitted|dedup_rejected|rate_limited|error). Mirrors
	// trajectory.Record.AdmissionDecision.
	AdmissionTotal *prometheus.CounterVec // engram_admission_total{type,decision}
	// CheckpointTotal counts write-discipline checkpoint outcomes, labelled by
	// checkpoint kind (cp1|cp2) and outcome bucket (advisory|clean).
	CheckpointTotal *prometheus.CounterVec // engram_checkpoint_total{kind,bucket}
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

	reflectionRuns := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_reflection_runs_total",
		Help: "Total reflection engine runs.",
	}, []string{"mode", "collection"})
	reflectionInsightsCreated := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_reflection_insights_created_total",
		Help: "Total insights created by reflection engine.",
	}, []string{"mode", "confidence"})
	reg.MustRegister(reflectionRuns, reflectionInsightsCreated)

	memoryOps := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_memory_ops_total",
		Help: "Total memory operations by operation, collection, and source_type.",
	}, []string{"operation", "collection", "source_type"})
	dedupHits := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_dedup_hits_total",
		Help: "Total deduplication hits by collection and dedup_type (server_side_092|client_side_078).",
	}, []string{"collection", "dedup_type"})
	reg.MustRegister(memoryOps, dedupHits)

	evaporationSweepTotal := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "engram_evaporation_sweep_total",
		Help: "Total evaporation sweep executions.",
	})
	evaporationDeprecatedTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_evaporation_deprecated_total",
		Help: "Total memories deprecated by evaporation, by type.",
	}, []string{"type"})
	evaporationSweepDuration := prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "engram_evaporation_sweep_duration_seconds",
		Help:    "Duration of evaporation sweep runs.",
		Buckets: prometheus.DefBuckets,
	})
	evaporationEffectiveImportance := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "engram_evaporation_effective_importance",
		Help:    "Distribution of query-time effective_importance, by type.",
		Buckets: prometheus.LinearBuckets(0, 1, 11),
	}, []string{"type"})
	reg.MustRegister(evaporationSweepTotal, evaporationDeprecatedTotal, evaporationSweepDuration, evaporationEffectiveImportance)

	evaporationScannedTotal := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "engram_evaporation_scanned_total",
		Help: "Total active memories scanned by the evaporation sweep.",
	})
	evaporationExemptedTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_evaporation_exempted_total",
		Help: "Total memories exempted from evaporation, by type and protection rule (P1..P8).",
	}, []string{"type", "rule"})
	evaporationDryRunCandidatesTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_evaporation_dry_run_candidates_total",
		Help: "Total below-threshold, non-exempt evaporation candidates identified by the sweep, by type.",
	}, []string{"type"})
	evaporationHardDeletedTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_evaporation_hard_deleted_total",
		Help: "Total memories hard-deleted via the expiry path, by type.",
	}, []string{"type"})
	reg.MustRegister(evaporationScannedTotal, evaporationExemptedTotal, evaporationDryRunCandidatesTotal, evaporationHardDeletedTotal)

	// Pre-declare evaporation counter series at 0 so they appear on /metrics
	// from process start (before any sweep/expiry has run). Types are read from
	// memory.ValidTypes (not hardcoded). exemptRules mirrors the inline P1..P8
	// rule IDs returned by EvaporationExempt in pkg/memory/evaporation.go.
	exemptRules := []string{"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
	for t := range memory.ValidTypes {
		evaporationDeprecatedTotal.WithLabelValues(string(t)).Add(0)
		evaporationDryRunCandidatesTotal.WithLabelValues(string(t)).Add(0)
		evaporationHardDeletedTotal.WithLabelValues(string(t)).Add(0)
		for _, rule := range exemptRules {
			evaporationExemptedTotal.WithLabelValues(string(t), rule).Add(0)
		}
	}

	importanceMean := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "engram_importance_mean",
		Help: "Rolling mean importance of recent writes per collection (CP2 monitor).",
	}, []string{"collection"})
	importanceP90 := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "engram_importance_p90",
		Help: "Rolling p90 importance of recent writes per collection (CP2 monitor).",
	}, []string{"collection"})
	reg.MustRegister(importanceMean, importanceP90)

	searchTopScore := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "engram_search_top_score",
		Help:    "top similarity score of vector search results, by collection (observed on memory_search and dedup paths)",
		Buckets: prometheus.LinearBuckets(0.50, 0.05, 11),
	}, []string{"collection"})
	reg.MustRegister(searchTopScore)

	admissionTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_admission_total",
		Help: "Total memory-write admission outcomes by type and decision (admitted|dedup_rejected|rate_limited|error).",
	}, []string{"type", "decision"})
	checkpointTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "engram_checkpoint_total",
		Help: "Total write-discipline checkpoint outcomes by kind (cp1|cp2) and outcome bucket (advisory|clean).",
	}, []string{"kind", "bucket"})
	reg.MustRegister(admissionTotal, checkpointTotal)

	if embedCache != nil {
		reg.MustRegister(newEmbedCacheCollector(embedCache))
	}
	if collectionStatsFn != nil {
		reg.MustRegister(newMemoryCountCollector(collectionStatsFn))
	}

	return &Metrics{
		Registry:                         reg,
		SearchDuration:                   searchDur,
		EmbedDuration:                    embedDur,
		ReflectionRuns:                   reflectionRuns,
		ReflectionInsightsCreated:        reflectionInsightsCreated,
		MemoryOps:                        memoryOps,
		DedupHits:                        dedupHits,
		EvaporationSweepTotal:            evaporationSweepTotal,
		EvaporationDeprecatedTotal:       evaporationDeprecatedTotal,
		EvaporationSweepDuration:         evaporationSweepDuration,
		EvaporationEffectiveImportance:   evaporationEffectiveImportance,
		EvaporationScannedTotal:          evaporationScannedTotal,
		EvaporationExemptedTotal:         evaporationExemptedTotal,
		EvaporationDryRunCandidatesTotal: evaporationDryRunCandidatesTotal,
		EvaporationHardDeletedTotal:      evaporationHardDeletedTotal,
		ImportanceMean:                   importanceMean,
		ImportanceP90:                    importanceP90,
		SearchTopScore:                   searchTopScore,
		AdmissionTotal:                   admissionTotal,
		CheckpointTotal:                  checkpointTotal,
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
