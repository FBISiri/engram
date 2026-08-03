// Package replay implements the Engram Replay Harness MVP: it captures real
// production memory-search interactions from trajectory JSONL logs, replays
// them against a (possibly reconfigured) running Engram server, and measures
// retrieval regression for A/B config testing.
//
// See design: engram-replay-harness-mvp-design-2026-08-02.md (§4.2).
package replay

import (
	"time"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// -----------------------------------------------------------------------------
// A. Trace Loader types
// -----------------------------------------------------------------------------

// ReplayCase is a single retrieve operation extracted from trajectory JSONL.
type ReplayCase struct {
	Timestamp time.Time `json:"timestamp"`
	Query     string    `json:"query"`
	Strategy  string    `json:"strategy"`
	Caller    string    `json:"caller"`

	// Ground truth: what the production system actually returned.
	RecordedResults []trajectory.ResultItem `json:"recorded_results"`
	RecordedLatency int64                   `json:"recorded_latency"`
}

// -----------------------------------------------------------------------------
// B. Replay Engine types
// -----------------------------------------------------------------------------

// ReplayResult holds the live results produced when replaying a ReplayCase.
type ReplayResult struct {
	Case          *ReplayCase             `json:"-"`
	LiveResults   []trajectory.ResultItem `json:"live_results"`
	LiveLatency   int64                   `json:"live_latency"`
	ConfigApplied MemoryConfig            `json:"config_applied"`
}

// MemoryConfig is a local mirror of pkg/server's config-override JSON shape.
// It is duplicated here deliberately to avoid importing pkg/server (R9: no
// import cycle). The JSON tags match memory_apply_config's expected input.
type MemoryConfig struct {
	RetrieveConfig RetrieveConfig `json:"retrieve_config"`
	UpdateConfig   UpdateConfig   `json:"update_config"`
}

// RetrieveConfig holds hot-reloadable retrieval settings.
type RetrieveConfig struct {
	RecencyWeight      float64 `json:"recency_weight,omitempty"`
	ScoreThreshold     float64 `json:"score_threshold,omitempty"`
	TopK               int     `json:"top_k,omitempty"`
	QueryRewritePrompt string  `json:"query_rewrite_prompt,omitempty"`
}

// UpdateConfig holds hot-reloadable write settings.
type UpdateConfig struct {
	DedupeThreshold float64 `json:"dedupe_threshold,omitempty"`
	MaxEntries      int     `json:"max_entries,omitempty"`
	EvictionPolicy  string  `json:"eviction_policy,omitempty"`
}

// -----------------------------------------------------------------------------
// C. Comparator types
// -----------------------------------------------------------------------------

// CaseComparison holds per-case retrieval metrics comparing recorded vs live.
type CaseComparison struct {
	Query           string   `json:"query"`
	RecallAtK       float64  `json:"recall_at_k"`
	PrecisionAtK    float64  `json:"precision_at_k"`
	RankCorrelation float64  `json:"rank_correlation"`         // Kendall's tau on overlapping IDs
	ScoreShift      float64  `json:"score_shift"`              // mean score delta for overlapping IDs
	NewIDs          []string `json:"new_ids"`                  // in live but not recorded
	DroppedIDs      []string `json:"dropped_ids"`              // in recorded but not live
	RecordedEmpty   bool     `json:"recorded_empty,omitempty"` // recorded set empty, vacuously satisfied
	OverlapCount    int      `json:"overlap_count"`            // #IDs in both recorded and live (definedness of tau/shift)
	RecordedLatency int64    `json:"recorded_latency"`
	LiveLatency     int64    `json:"live_latency"`
}

// AggregateReport summarizes CaseComparisons across a full replay run.
type AggregateReport struct {
	TotalCases       int     `json:"total_cases"`
	MeanRecall       float64 `json:"mean_recall"`
	MeanPrecision    float64 `json:"mean_precision"`
	MeanRankCorr     float64 `json:"mean_rank_corr"`
	MeanScoreShift   float64 `json:"mean_score_shift"`
	RegressionCount  int     `json:"regression_count"`  // cases where recall dropped > threshold
	ImprovementCount int     `json:"improvement_count"` // cases where recall stayed full but new IDs appeared

	// Distribution
	RecallHistogram [10]int `json:"recall_histogram"` // 0-10%, 10-20%, ..., 90-100%
	LatencyP50Delta int64   `json:"latency_p50_delta"`
	LatencyP99Delta int64   `json:"latency_p99_delta"`

	Verdict Verdict `json:"verdict"`
}

// Verdict is the PASS/WARN/FAIL outcome of a replay run.
type Verdict string

const (
	VerdictPass Verdict = "PASS"
	VerdictWarn Verdict = "WARN"
	VerdictFail Verdict = "FAIL"
)

// Thresholds configure the pass/fail decision. See design §4.2C.
type Thresholds struct {
	MeanRecallPass float64 // PASS requires MeanRecall >= this (default 0.80)
	MeanRecallWarn float64 // FAIL if MeanRecall < this (default 0.70)
	RegressionPct  float64 // PASS requires RegressionCount/Total <= this (default 0.05)
	RegressionWarn float64 // FAIL if RegressionCount/Total > this (default 0.10)
	RegressionDrop float64 // a case is a regression if (1-recall) > this (default 0.20)
}

// DefaultThresholds returns the design-doc starting thresholds.
func DefaultThresholds() Thresholds {
	return Thresholds{
		MeanRecallPass: 0.80,
		MeanRecallWarn: 0.70,
		RegressionPct:  0.05,
		RegressionWarn: 0.10,
		RegressionDrop: 0.20,
	}
}

// Normalize keeps the hardcoded WARN floors from contradicting operator-set PASS
// thresholds (--mean-recall / --regression-pct). A run at or above the PASS
// threshold must never FAIL, so when the operator relaxes a PASS bound past the
// default WARN floor, the floor is relaxed with it.
//
// assumed: no separate WARN band is desired below an operator-declared PASS
// line; the WARN floor is simply clamped to the PASS threshold (WARN collapses
// to the PASS bound when the operator makes PASS more lenient than the default).
func (t Thresholds) Normalize() Thresholds {
	if t.MeanRecallWarn > t.MeanRecallPass {
		t.MeanRecallWarn = t.MeanRecallPass
	}
	if t.RegressionWarn < t.RegressionPct {
		t.RegressionWarn = t.RegressionPct
	}
	return t
}

// Report is the full replay report emitted by the reporter (JSON + Markdown).
type Report struct {
	GeneratedAt time.Time        `json:"generated_at"`
	Trace       string           `json:"trace"`
	Collection  string           `json:"collection"`
	ConfigDiff  MemoryConfig     `json:"config_diff"`
	Aggregate   AggregateReport  `json:"aggregate"`
	Comparisons []CaseComparison `json:"comparisons"`
	Thresholds  Thresholds       `json:"thresholds"`
}

// AggregateDelta holds candidate-minus-baseline deltas for a multi-day compare.
type AggregateDelta struct {
	MeanRecallDelta    float64 `json:"mean_recall_delta"`
	MeanPrecisionDelta float64 `json:"mean_precision_delta"`
	MeanRankCorrDelta  float64 `json:"mean_rank_corr_delta"`
	RegressionDelta    int     `json:"regression_delta"`
	LatencyP50Delta    int64   `json:"latency_p50_delta"`
	LatencyP99Delta    int64   `json:"latency_p99_delta"`
}

// MultiDayReport compares a baseline trace day against a candidate trace day.
type MultiDayReport struct {
	GeneratedAt time.Time      `json:"generated_at"`
	Baseline    Report         `json:"baseline"`
	Candidate   Report         `json:"candidate"`
	Delta       AggregateDelta `json:"delta"`
}
