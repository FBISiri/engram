package replay

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// BuildComparisons compares each replay result against its recorded case,
// populating query + latency fields for reporting.
func BuildComparisons(results []ReplayResult) []CaseComparison {
	cmps := make([]CaseComparison, 0, len(results))
	for i := range results {
		r := &results[i]
		c := Compare(r.Case.RecordedResults, r.LiveResults)
		c.Query = r.Case.Query
		c.RecordedLatency = r.Case.RecordedLatency
		c.LiveLatency = r.LiveLatency
		cmps = append(cmps, c)
	}
	return cmps
}

// BuildReport assembles the full report from a completed replay run.
func BuildReport(trace, collection string, cfg MemoryConfig, results []ReplayResult, th Thresholds) Report {
	cmps := BuildComparisons(results)
	return Report{
		GeneratedAt: time.Now().UTC(),
		Trace:       trace,
		Collection:  collection,
		ConfigDiff:  cfg,
		Aggregate:   Aggregate(cmps, th),
		Comparisons: cmps,
		Thresholds:  th,
	}
}

// RenderJSON returns the machine-readable JSON report.
func RenderJSON(r Report) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RenderMarkdown returns a human-readable Markdown report for Obsidian archival.
func RenderMarkdown(r Report) string {
	a := r.Aggregate
	var b strings.Builder
	fmt.Fprintf(&b, "# Replay Report\n\n")
	fmt.Fprintf(&b, "> Generated: %s | Trace: `%s` | Collection: `%s`\n\n",
		r.GeneratedAt.Format(time.RFC3339), r.Trace, r.Collection)
	fmt.Fprintf(&b, "**Verdict: %s**\n\n", a.Verdict)

	fmt.Fprintf(&b, "## Config Diff\n\n")
	cfgJSON, _ := json.MarshalIndent(r.ConfigDiff, "", "  ")
	fmt.Fprintf(&b, "```json\n%s\n```\n\n", string(cfgJSON))

	fmt.Fprintf(&b, "## Aggregate Metrics\n\n")
	fmt.Fprintf(&b, "| Metric | Value |\n|---|---|\n")
	fmt.Fprintf(&b, "| Total cases | %d |\n", a.TotalCases)
	fmt.Fprintf(&b, "| Mean Recall@K | %.4f |\n", a.MeanRecall)
	fmt.Fprintf(&b, "| Mean Precision@K | %.4f |\n", a.MeanPrecision)
	fmt.Fprintf(&b, "| Mean Rank Corr (τ) | %.4f |\n", a.MeanRankCorr)
	fmt.Fprintf(&b, "| Mean Score Shift | %.4f |\n", a.MeanScoreShift)
	fmt.Fprintf(&b, "| Regressions | %d |\n", a.RegressionCount)
	fmt.Fprintf(&b, "| Improvements | %d |\n", a.ImprovementCount)
	b.WriteString("\n")

	fmt.Fprintf(&b, "## Recall Histogram\n\n")
	fmt.Fprintf(&b, "| Bucket | Count |\n|---|---|\n")
	for i, c := range a.RecallHistogram {
		fmt.Fprintf(&b, "| %d0-%d0%% | %d |\n", i, i+1, c)
	}
	b.WriteString("\n")

	fmt.Fprintf(&b, "## Latency Comparison\n\n")
	fmt.Fprintf(&b, "| Percentile | Delta (ms) |\n|---|---|\n")
	fmt.Fprintf(&b, "| p50 | %+d |\n", a.LatencyP50Delta)
	fmt.Fprintf(&b, "| p99 | %+d |\n\n", a.LatencyP99Delta)

	fmt.Fprintf(&b, "## Regressions\n\n")
	any := false
	for _, c := range r.Comparisons {
		if (1 - c.RecallAtK) <= r.Thresholds.RegressionDrop {
			continue
		}
		any = true
		fmt.Fprintf(&b, "- **recall=%.2f** dropped=%d query=%q\n",
			c.RecallAtK, len(c.DroppedIDs), truncate(c.Query, 80))
	}
	if !any {
		fmt.Fprintf(&b, "_No regressions above threshold._\n")
	}
	b.WriteString("\n")
	b.WriteString(knownLimitationsMarkdown())
	return b.String()
}

// knownLimitationsMarkdown documents inherent server-side behaviors that bias
// replay results and cannot be fixed from the harness (they'd need server
// changes). Surfaced in every report so A/B conclusions are read with caveats.
func knownLimitationsMarkdown() string {
	return "## Known Limitations\n\n" +
		"- **Recency contamination:** `/memories/search` fires an async server goroutine that bumps `access_count` / `last_accessed_at` on every matched memory, so replay perturbs the very recency signal a recency-weight A/B measures.\n" +
		"- **Caller isolation not reproduced:** recorded results from an isolated (collection-scoped) caller are replayed as a non-isolated global fan-out, so scoped-caller trajectories are compared apples-to-oranges. Reproducing it would require the caller's principal API key, which the harness does not hold.\n" +
		"- **Config apply is set-only:** the server honors only `recency_weight`, `top_k`, `dedupe_threshold`; it cannot clear a field (hence a mandatory non-zero `--baseline-config`).\n" +
		"- **`--snapshot` is global:** `memory_reset` rewrites the ENTIRE store, not just the eval collection (data-loss window if killed mid-run).\n"
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// BuildMultiDayReport assembles a baseline-vs-candidate multi-day report.
func BuildMultiDayReport(baseline, candidate Report) MultiDayReport {
	return MultiDayReport{
		GeneratedAt: time.Now().UTC(),
		Baseline:    baseline,
		Candidate:   candidate,
		Delta:       CompareAggregates(baseline.Aggregate, candidate.Aggregate),
	}
}

// RenderMultiDayJSON returns the machine-readable multi-day JSON report.
func RenderMultiDayJSON(r MultiDayReport) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// RenderMultiDayMarkdown returns a human-readable multi-day comparison report.
func RenderMultiDayMarkdown(r MultiDayReport) string {
	d := r.Delta
	var b strings.Builder
	fmt.Fprintf(&b, "# Multi-Day Replay Comparison\n\n")
	fmt.Fprintf(&b, "> Generated: %s\n\n", r.GeneratedAt.Format(time.RFC3339))
	fmt.Fprintf(&b, "| Trace | Cases | Mean Recall | Mean Precision | Regressions | Verdict |\n|---|---|---|---|---|---|\n")
	fmt.Fprintf(&b, "| baseline `%s` | %d | %.4f | %.4f | %d | %s |\n",
		r.Baseline.Trace, r.Baseline.Aggregate.TotalCases, r.Baseline.Aggregate.MeanRecall,
		r.Baseline.Aggregate.MeanPrecision, r.Baseline.Aggregate.RegressionCount, r.Baseline.Aggregate.Verdict)
	fmt.Fprintf(&b, "| candidate `%s` | %d | %.4f | %.4f | %d | %s |\n\n",
		r.Candidate.Trace, r.Candidate.Aggregate.TotalCases, r.Candidate.Aggregate.MeanRecall,
		r.Candidate.Aggregate.MeanPrecision, r.Candidate.Aggregate.RegressionCount, r.Candidate.Aggregate.Verdict)

	fmt.Fprintf(&b, "## Delta (candidate − baseline)\n\n")
	fmt.Fprintf(&b, "| Metric | Delta |\n|---|---|\n")
	fmt.Fprintf(&b, "| Mean Recall | %+.4f |\n", d.MeanRecallDelta)
	fmt.Fprintf(&b, "| Mean Precision | %+.4f |\n", d.MeanPrecisionDelta)
	fmt.Fprintf(&b, "| Mean Rank Corr | %+.4f |\n", d.MeanRankCorrDelta)
	fmt.Fprintf(&b, "| Regressions | %+d |\n", d.RegressionDelta)
	fmt.Fprintf(&b, "| Latency p50 (ms) | %+d |\n", d.LatencyP50Delta)
	fmt.Fprintf(&b, "| Latency p99 (ms) | %+d |\n\n", d.LatencyP99Delta)
	b.WriteString(knownLimitationsMarkdown())
	return b.String()
}
