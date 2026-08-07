package replay

import (
	"sort"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// Compare computes per-case retrieval metrics comparing the recorded results
// (ground truth) against the live results produced by the replay engine.
func Compare(rec []trajectory.ResultItem, live []trajectory.ResultItem) CaseComparison {
	recIDs := ids(rec)
	liveIDs := ids(live)
	recSet := toSet(recIDs)
	liveSet := toSet(liveIDs)

	overlap := 0
	for id := range recSet {
		if liveSet[id] {
			overlap++
		}
	}

	// A recorded-empty case is vacuously satisfied: there were no ground-truth
	// IDs to recall, so recall is 1.0 and it must NOT count as a total-loss.
	recordedEmpty := len(recIDs) == 0
	var recall, precision float64
	if recordedEmpty {
		recall = 1.0
	} else {
		recall = float64(overlap) / float64(len(recIDs))
	}
	if len(liveIDs) > 0 {
		precision = float64(overlap) / float64(len(liveIDs))
	} else if recordedEmpty {
		precision = 1.0 // both empty → vacuously precise
	}

	var newIDs, dropped []string
	for _, id := range liveIDs {
		if !recSet[id] {
			newIDs = append(newIDs, id)
		}
	}
	for _, id := range recIDs {
		if !liveSet[id] {
			dropped = append(dropped, id)
		}
	}

	return CaseComparison{
		RecallAtK:       recall,
		PrecisionAtK:    precision,
		RankCorrelation: kendallTau(recIDs, liveIDs),
		ScoreShift:      scoreShift(rec, live),
		NewIDs:          newIDs,
		DroppedIDs:      dropped,
		RecordedEmpty:   recordedEmpty,
		OverlapCount:    overlap,
	}
}

// kendallTau computes Kendall's tau-a over the IDs present in BOTH rankings,
// using each ID's rank (position) in the recorded and live orderings. Pure Go.
// Returns 0 when fewer than two IDs overlap (undefined correlation).
func kendallTau(recIDs, liveIDs []string) float64 {
	recRank := ranks(recIDs)
	liveRank := ranks(liveIDs)

	var common []string
	for _, id := range recIDs {
		if _, ok := liveRank[id]; ok {
			common = append(common, id)
		}
	}
	n := len(common)
	if n < 2 {
		return 0
	}

	var concordant, discordant int
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a, b := common[i], common[j]
			dr := recRank[a] - recRank[b]
			dl := liveRank[a] - liveRank[b]
			prod := dr * dl
			if prod > 0 {
				concordant++
			} else if prod < 0 {
				discordant++
			}
		}
	}
	pairs := n * (n - 1) / 2
	return float64(concordant-discordant) / float64(pairs)
}

// scoreShift is the mean (live - recorded) score delta over overlapping IDs.
func scoreShift(rec, live []trajectory.ResultItem) float64 {
	liveScore := make(map[string]float64, len(live))
	for _, r := range live {
		liveScore[r.ID] = r.Score
	}
	var sum float64
	var n int
	for _, r := range rec {
		if s, ok := liveScore[r.ID]; ok {
			sum += s - r.Score
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// Aggregate rolls per-case comparisons and live/recorded latencies into an
// AggregateReport, applying the pass/fail thresholds.
func Aggregate(cmps []CaseComparison, th Thresholds, records ...trajectory.Record) AggregateReport {
	rep := AggregateReport{TotalCases: len(cmps)}
	// Memory Worth is joined from the raw trajectory records (task_id + outcome),
	// independent of the recorded-vs-live comparison. Nil when no task_id data.
	rep.MW = ComputeMW(records)
	if len(cmps) == 0 {
		rep.Verdict = VerdictPass
		return rep
	}

	var sumRecall, sumPrec, sumCorr, sumShift float64
	var recallN, corrN, shiftN int
	totalLoss := false
	recLat := make([]int64, 0, len(cmps))
	liveLat := make([]int64, 0, len(cmps))
	for _, c := range cmps {
		// RankCorrelation is only defined with >=2 overlapping IDs; ScoreShift
		// with >=1. Average each only over cases where it is defined so
		// undefined-zero cases don't bias the means toward 0.
		if c.OverlapCount >= 2 {
			sumCorr += c.RankCorrelation
			corrN++
		}
		if c.OverlapCount >= 1 {
			sumShift += c.ScoreShift
			shiftN++
		}

		b := int(c.RecallAtK * 10)
		if b > 9 {
			b = 9
		}
		if b < 0 {
			b = 0
		}
		rep.RecallHistogram[b]++

		// Recorded-empty cases are vacuously satisfied: excluded from BOTH the
		// recall and precision means (symmetric) and never a regression /
		// total-loss.
		if !c.RecordedEmpty {
			sumRecall += c.RecallAtK
			sumPrec += c.PrecisionAtK
			recallN++
			if (1 - c.RecallAtK) > th.RegressionDrop {
				rep.RegressionCount++
			}
			if c.RecallAtK == 0 {
				totalLoss = true
			}
		}
		if c.RecallAtK >= 1.0 && len(c.NewIDs) > 0 {
			rep.ImprovementCount++
		}
		recLat = append(recLat, c.RecordedLatency)
		liveLat = append(liveLat, c.LiveLatency)
	}

	rep.MeanRecall = 1.0
	if recallN > 0 {
		rep.MeanRecall = sumRecall / float64(recallN)
	}
	rep.MeanPrecision = 1.0
	if recallN > 0 {
		rep.MeanPrecision = sumPrec / float64(recallN)
	}
	if corrN > 0 {
		rep.MeanRankCorr = sumCorr / float64(corrN)
	}
	if shiftN > 0 {
		rep.MeanScoreShift = sumShift / float64(shiftN)
	}
	rep.LatencyP50Delta = percentile(liveLat, 0.50) - percentile(recLat, 0.50)
	rep.LatencyP99Delta = percentile(liveLat, 0.99) - percentile(recLat, 0.99)
	rep.Verdict = verdict(rep, th, totalLoss, recallN)
	return rep
}

func verdict(rep AggregateReport, th Thresholds, totalLoss bool, recallN int) Verdict {
	// Regression% uses the SAME non-empty denominator as MeanRecall so vacuous
	// recorded-empty cases can't dilute/mask a FAIL.
	var regPct float64
	if recallN > 0 {
		regPct = float64(rep.RegressionCount) / float64(recallN)
	}
	if rep.MeanRecall < th.MeanRecallWarn || regPct > th.RegressionWarn || totalLoss {
		return VerdictFail
	}
	// PASS conditions.
	if rep.MeanRecall >= th.MeanRecallPass && regPct <= th.RegressionPct {
		return VerdictPass
	}
	return VerdictWarn
}

// percentile returns the value at fraction p (0..1) using nearest-rank on a
// copy of the input. Returns 0 for an empty slice.
func percentile(vals []int64, p float64) int64 {
	if len(vals) == 0 {
		return 0
	}
	cp := make([]int64, len(vals))
	copy(cp, vals)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	idx := int(p * float64(len(cp)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}

func ids(items []trajectory.ResultItem) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.ID
	}
	return out
}

func ranks(order []string) map[string]int {
	m := make(map[string]int, len(order))
	for i, id := range order {
		if _, seen := m[id]; !seen {
			m[id] = i
		}
	}
	return m
}

func toSet(xs []string) map[string]bool {
	m := make(map[string]bool, len(xs))
	for _, x := range xs {
		m[x] = true
	}
	return m
}

// CompareAggregates returns candidate-minus-baseline deltas for multi-day runs.
func CompareAggregates(baseline, candidate AggregateReport) AggregateDelta {
	return AggregateDelta{
		MeanRecallDelta:    candidate.MeanRecall - baseline.MeanRecall,
		MeanPrecisionDelta: candidate.MeanPrecision - baseline.MeanPrecision,
		MeanRankCorrDelta:  candidate.MeanRankCorr - baseline.MeanRankCorr,
		RegressionDelta:    candidate.RegressionCount - baseline.RegressionCount,
		LatencyP50Delta:    candidate.LatencyP50Delta - baseline.LatencyP50Delta,
		LatencyP99Delta:    candidate.LatencyP99Delta - baseline.LatencyP99Delta,
	}
}
