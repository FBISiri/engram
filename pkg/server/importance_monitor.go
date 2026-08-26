// importance_monitor.go — CP2: importance distribution monitor (spec v1 §2.2).
//
// Maintains a per-collection sliding-window ring buffer of recent importance
// values. When the rolling mean exceeds the threshold (and enough samples have
// accumulated), Record returns an `importance_inflation` advisory. This tracks
// the *trend* — complementary to A-MAC's per-write clamp, which only bounds
// individual values.
package server

import (
	"fmt"
	"math"
	"sort"
	"sync"

	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
)

// importanceRing is a fixed-size circular buffer of importance values.
type importanceRing struct {
	values []float64
	pos    int
	count  int
}

// ImportanceMonitor tracks recent importance values per collection.
type ImportanceMonitor struct {
	mu        sync.Mutex
	window    int
	threshold float64
	entries   map[string]*importanceRing
	metrics   *engrammetrics.Metrics
}

// NewImportanceMonitor builds a monitor with the given ring size and mean
// threshold. A non-positive window falls back to 50.
func NewImportanceMonitor(window int, threshold float64) *ImportanceMonitor {
	if window <= 0 {
		window = 50
	}
	if threshold <= 0 {
		threshold = 7.0
	}
	return &ImportanceMonitor{
		window:    window,
		threshold: threshold,
		entries:   map[string]*importanceRing{},
	}
}

// SetMetrics registers the Prometheus metrics used to export the gauges.
func (m *ImportanceMonitor) SetMetrics(mt *engrammetrics.Metrics) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.metrics = mt
}

// Record writes importance into the collection's ring, refreshes the Prometheus
// gauges, and returns an advisory when the rolling mean exceeds the threshold.
// The advisory only triggers once at least window/2 samples have accumulated.
func (m *ImportanceMonitor) Record(collection string, importance float64) *Advisory {
	m.mu.Lock()
	defer m.mu.Unlock()

	r := m.entries[collection]
	if r == nil {
		r = &importanceRing{values: make([]float64, m.window)}
		m.entries[collection] = r
	}
	r.values[r.pos] = importance
	r.pos = (r.pos + 1) % m.window
	if r.count < m.window {
		r.count++
	}

	mean, p90 := ringStats(r)

	if m.metrics != nil {
		m.metrics.ImportanceMean.WithLabelValues(collection).Set(mean)
		m.metrics.ImportanceP90.WithLabelValues(collection).Set(p90)
	}

	// Require at least window/2 samples for statistical significance.
	if r.count < m.window/2 {
		return nil
	}
	if mean > m.threshold {
		return &Advisory{
			Type:     "importance_inflation",
			Severity: "warning",
			Message: fmt.Sprintf(
				"Recent %d writes have mean importance %.1f (threshold: %.1f). Consider using lower importance for routine memories.",
				r.count, mean, m.threshold),
			Data: map[string]any{
				"recent_mean":  mean,
				"threshold":    m.threshold,
				"window_size":  m.window,
				"sample_count": r.count,
			},
		}
	}
	return nil
}

// ringStats returns the mean and p90 of the filled values in the ring.
func ringStats(r *importanceRing) (mean, p90 float64) {
	if r.count == 0 {
		return 0, 0
	}
	vals := make([]float64, r.count)
	copy(vals, r.values[:r.count])
	var sum float64
	for _, v := range vals {
		sum += v
	}
	mean = sum / float64(r.count)

	sort.Float64s(vals)
	idx := int(math.Ceil(0.9*float64(r.count))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= r.count {
		idx = r.count - 1
	}
	p90 = vals[idx]
	return mean, p90
}
