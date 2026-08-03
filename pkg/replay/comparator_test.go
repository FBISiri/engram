package replay

import (
	"math"
	"testing"

	"github.com/FBISiri/engram/pkg/trajectory"
)

func items(pairs ...any) []trajectory.ResultItem {
	// pairs: id, score, id, score, ...
	var out []trajectory.ResultItem
	for i := 0; i < len(pairs); i += 2 {
		out = append(out, trajectory.ResultItem{
			ID:    pairs[i].(string),
			Score: pairs[i+1].(float64),
		})
	}
	return out
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestCompare_RecallPrecision(t *testing.T) {
	rec := items("a", 1.0, "b", 0.9, "c", 0.8, "d", 0.7)
	live := items("a", 1.0, "b", 0.9, "x", 0.5) // overlap {a,b}=2

	c := Compare(rec, live)
	// Recall = overlap/|rec| = 2/4 = 0.5
	if !approx(c.RecallAtK, 0.5) {
		t.Errorf("recall = %v, want 0.5", c.RecallAtK)
	}
	// Precision = overlap/|live| = 2/3
	if !approx(c.PrecisionAtK, 2.0/3.0) {
		t.Errorf("precision = %v, want 0.6667", c.PrecisionAtK)
	}
	if len(c.NewIDs) != 1 || c.NewIDs[0] != "x" {
		t.Errorf("new ids = %v, want [x]", c.NewIDs)
	}
	if len(c.DroppedIDs) != 2 {
		t.Errorf("dropped ids = %v, want 2 (c,d)", c.DroppedIDs)
	}
}

func TestKendallTau_PerfectAgreement(t *testing.T) {
	rec := []string{"a", "b", "c", "d"}
	live := []string{"a", "b", "c", "d"}
	if got := kendallTau(rec, live); !approx(got, 1.0) {
		t.Errorf("tau = %v, want 1.0", got)
	}
}

func TestKendallTau_PerfectDisagreement(t *testing.T) {
	rec := []string{"a", "b", "c", "d"}
	live := []string{"d", "c", "b", "a"}
	if got := kendallTau(rec, live); !approx(got, -1.0) {
		t.Errorf("tau = %v, want -1.0", got)
	}
}

func TestKendallTau_OneSwap(t *testing.T) {
	// recorded order a,b,c ; live swaps b and c → a,c,b
	// pairs: (a,b) concordant, (a,c) concordant, (b,c) discordant
	// tau = (2 - 1) / 3 = 1/3
	rec := []string{"a", "b", "c"}
	live := []string{"a", "c", "b"}
	if got := kendallTau(rec, live); !approx(got, 1.0/3.0) {
		t.Errorf("tau = %v, want 0.3333", got)
	}
}

func TestKendallTau_Overlap(t *testing.T) {
	// Only {a,b,c} overlap; live has extra x. Order in both a<b<c → tau=1.
	rec := []string{"a", "b", "c"}
	live := []string{"x", "a", "b", "c"}
	if got := kendallTau(rec, live); !approx(got, 1.0) {
		t.Errorf("tau = %v, want 1.0", got)
	}
}

func TestScoreShift(t *testing.T) {
	rec := items("a", 1.0, "b", 0.5)
	live := items("a", 1.2, "b", 0.4) // deltas +0.2, -0.1 → mean 0.05
	c := Compare(rec, live)
	if !approx(c.ScoreShift, 0.05) {
		t.Errorf("score shift = %v, want 0.05", c.ScoreShift)
	}
}

func TestAggregate_VerdictPass(t *testing.T) {
	// All perfect recall → PASS.
	cmps := []CaseComparison{
		{RecallAtK: 1.0, PrecisionAtK: 1.0, RankCorrelation: 1.0, RecordedLatency: 10, LiveLatency: 12},
		{RecallAtK: 1.0, PrecisionAtK: 1.0, RankCorrelation: 1.0, RecordedLatency: 20, LiveLatency: 18},
	}
	a := Aggregate(cmps, DefaultThresholds())
	if a.Verdict != VerdictPass {
		t.Errorf("verdict = %s, want PASS", a.Verdict)
	}
	if !approx(a.MeanRecall, 1.0) {
		t.Errorf("mean recall = %v", a.MeanRecall)
	}
	if a.RecallHistogram[9] != 2 {
		t.Errorf("histogram top bucket = %d, want 2", a.RecallHistogram[9])
	}
}

func TestAggregate_VerdictFail_TotalLoss(t *testing.T) {
	cmps := []CaseComparison{
		{RecallAtK: 1.0},
		{RecallAtK: 0.0}, // total loss → FAIL
	}
	a := Aggregate(cmps, DefaultThresholds())
	if a.Verdict != VerdictFail {
		t.Errorf("verdict = %s, want FAIL (total loss)", a.Verdict)
	}
}

func TestAggregate_LatencyDelta(t *testing.T) {
	cmps := []CaseComparison{
		{RecallAtK: 1.0, RecordedLatency: 10, LiveLatency: 30},
		{RecallAtK: 1.0, RecordedLatency: 20, LiveLatency: 40},
	}
	a := Aggregate(cmps, DefaultThresholds())
	// p50 nearest-rank index int(0.5*1)=0 → live[0]=30, rec[0]=10 → +20
	if a.LatencyP50Delta != 20 {
		t.Errorf("p50 delta = %d, want 20", a.LatencyP50Delta)
	}
}

func TestCompare_RecordedEmpty_Vacuous(t *testing.T) {
	// No recorded ground truth → vacuously satisfied: recall 1.0, RecordedEmpty.
	c := Compare(nil, items("x", 0.5))
	if c.RecallAtK != 1.0 {
		t.Errorf("recall = %v, want 1.0 (vacuous)", c.RecallAtK)
	}
	if !c.RecordedEmpty {
		t.Error("RecordedEmpty = false, want true")
	}
}

func TestAggregate_RecordedEmpty_NotTotalLoss(t *testing.T) {
	// One recorded-empty case (recall 1.0, vacuous) + one real perfect case.
	// The recorded-empty case must NOT force FAIL nor drag the recall mean.
	cmps := []CaseComparison{
		{RecallAtK: 1.0, RecordedEmpty: true},
		{RecallAtK: 1.0},
	}
	a := Aggregate(cmps, DefaultThresholds())
	if a.Verdict != VerdictPass {
		t.Errorf("verdict = %s, want PASS (empty-recorded is vacuous)", a.Verdict)
	}
	if !approx(a.MeanRecall, 1.0) {
		t.Errorf("mean recall = %v, want 1.0 (empty excluded)", a.MeanRecall)
	}
	// A recorded-empty case with recall 1.0 must not be counted as total-loss.
	cmps2 := []CaseComparison{{RecallAtK: 1.0, RecordedEmpty: true}}
	if v := Aggregate(cmps2, DefaultThresholds()).Verdict; v != VerdictPass {
		t.Errorf("all-empty verdict = %s, want PASS", v)
	}
}

func TestCompareAggregates_Delta(t *testing.T) {
	base := AggregateReport{MeanRecall: 0.80, MeanPrecision: 0.70, RegressionCount: 5, LatencyP50Delta: 10}
	cand := AggregateReport{MeanRecall: 0.85, MeanPrecision: 0.68, RegressionCount: 3, LatencyP50Delta: 15}
	d := CompareAggregates(base, cand)
	if !approx(d.MeanRecallDelta, 0.05) {
		t.Errorf("recall delta = %v, want 0.05", d.MeanRecallDelta)
	}
	if !approx(d.MeanPrecisionDelta, -0.02) {
		t.Errorf("precision delta = %v, want -0.02", d.MeanPrecisionDelta)
	}
	if d.RegressionDelta != -2 {
		t.Errorf("regression delta = %d, want -2", d.RegressionDelta)
	}
	if d.LatencyP50Delta != 5 {
		t.Errorf("latency p50 delta = %d, want 5", d.LatencyP50Delta)
	}
}

func TestAggregate_RegressionPct_ExcludesEmptyDenominator(t *testing.T) {
	// 1 real regression + 9 vacuous recorded-empty cases. With TotalCases=10
	// denominator regPct=0.1 (WARN boundary); with the correct non-empty
	// denominator (recallN=1) regPct=1.0 → must FAIL.
	cmps := []CaseComparison{{RecallAtK: 0.0}} // real total-loss regression
	for i := 0; i < 9; i++ {
		cmps = append(cmps, CaseComparison{RecallAtK: 1.0, RecordedEmpty: true})
	}
	a := Aggregate(cmps, DefaultThresholds())
	if a.RegressionCount != 1 {
		t.Fatalf("regression count = %d, want 1", a.RegressionCount)
	}
	if a.Verdict != VerdictFail {
		t.Errorf("verdict = %s, want FAIL (regPct must use non-empty denominator)", a.Verdict)
	}
}

func TestAggregate_MeanCorrShift_DefinedOnly(t *testing.T) {
	// Case A: 3 overlapping IDs → corr & shift defined.
	// Case B: 1 overlap → shift defined, corr NOT (needs >=2).
	// Case C: 0 overlap → neither defined.
	cmps := []CaseComparison{
		{RankCorrelation: 1.0, ScoreShift: 0.4, OverlapCount: 3, RecallAtK: 1.0},
		{RankCorrelation: 0.0, ScoreShift: 0.2, OverlapCount: 1, RecallAtK: 0.5},
		{RankCorrelation: 0.0, ScoreShift: 0.0, OverlapCount: 0, RecallAtK: 0.0},
	}
	a := Aggregate(cmps, DefaultThresholds())
	// MeanRankCorr over defined (only case A): 1.0, NOT 1.0/3.
	if !approx(a.MeanRankCorr, 1.0) {
		t.Errorf("mean rank corr = %v, want 1.0 (defined-only)", a.MeanRankCorr)
	}
	// MeanScoreShift over A,B: (0.4+0.2)/2 = 0.3, NOT /3.
	if !approx(a.MeanScoreShift, 0.3) {
		t.Errorf("mean score shift = %v, want 0.3 (defined-only)", a.MeanScoreShift)
	}
}

func TestCompare_SetsOverlapCount(t *testing.T) {
	c := Compare(items("a", 1.0, "b", 0.9, "c", 0.8), items("a", 1.0, "b", 0.9, "x", 0.5))
	if c.OverlapCount != 2 {
		t.Errorf("overlap count = %d, want 2", c.OverlapCount)
	}
}

func TestAggregate_Precision_ExcludesRecordedEmpty(t *testing.T) {
	// One real case (precision 1.0) + one recorded-empty (precision 0 in Compare).
	// Recorded-empty must be excluded from the precision mean, same as recall.
	cmps := []CaseComparison{
		{RecallAtK: 1.0, PrecisionAtK: 1.0, OverlapCount: 2},
		{RecallAtK: 1.0, PrecisionAtK: 0.0, RecordedEmpty: true},
	}
	a := Aggregate(cmps, DefaultThresholds())
	if !approx(a.MeanPrecision, 1.0) {
		t.Errorf("mean precision = %v, want 1.0 (empty-recorded excluded, symmetric with recall)", a.MeanPrecision)
	}
}

func TestThresholds_NormalizeAvoidsWarnContradiction(t *testing.T) {
	// Operator declares 0.60 passing; a 0.65 run must NOT FAIL despite the
	// default 0.70 WARN floor.
	th := DefaultThresholds()
	th.MeanRecallPass = 0.60
	th = th.Normalize()
	if th.MeanRecallWarn > th.MeanRecallPass {
		t.Fatalf("warn floor %v > pass %v after Normalize", th.MeanRecallWarn, th.MeanRecallPass)
	}
	cmps := []CaseComparison{{RecallAtK: 0.65, OverlapCount: 2}}
	th.RegressionDrop = 0.5 // isolate the recall-floor fix (0.35 drop is not a per-case regression here)
	if v := Aggregate(cmps, th).Verdict; v == VerdictFail {
		t.Errorf("verdict = FAIL, want non-FAIL (0.65 >= operator PASS 0.60)")
	}
	// Symmetric: a lenient regression-pct must relax the WARN ceiling too.
	th2 := DefaultThresholds()
	th2.RegressionPct = 0.20
	th2 = th2.Normalize()
	if th2.RegressionWarn < th2.RegressionPct {
		t.Fatalf("regression warn %v < pass %v after Normalize", th2.RegressionWarn, th2.RegressionPct)
	}
}
