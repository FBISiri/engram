package replay

import (
	"os"
	"path/filepath"
	"testing"
)

// candidateFixture writes a JSONL file mixing admitted / dedup_rejected /
// rate_limited / error candidate records, a retrieve record (must be ignored),
// and malformed lines (must be skipped).
func candidateFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "cand.jsonl")
	content := `{"timestamp":"2026-09-05T00:00:00Z","operation":"candidate","content":"admitted insight one","type":"insight","importance":6,"source_type":"reflection","admission_decision":"admitted","latency_ms":10,"caller":"agent-self","task_id":"loop-42"}
{"timestamp":"2026-09-05T00:00:01Z","operation":"candidate","content":"rejected high score","type":"insight","importance":6,"source_type":"reflection","admission_decision":"dedup_rejected","gate_details":"dedup score=0.9500 against id=aaa","latency_ms":11,"caller":"agent-self"}
{"timestamp":"2026-09-05T00:00:02Z","operation":"candidate","content":"rejected low score","type":"insight","importance":7,"source_type":"reflection","admission_decision":"dedup_rejected","gate_details":"dedup score=0.8900 against id=bbb","latency_ms":12,"caller":"agent-self"}
{"timestamp":"2026-09-05T00:00:03Z","operation":"candidate","content":"rate limited event","type":"event","importance":4,"admission_decision":"rate_limited","gate_details":"rate_limited: retry_after=30s","latency_ms":1,"caller":"agent-self"}
{"timestamp":"2026-09-05T00:00:04Z","operation":"candidate","content":"errored event","type":"event","importance":4,"admission_decision":"error","gate_details":"embed_error: boom","latency_ms":2,"caller":"agent-self"}
{"timestamp":"2026-09-05T00:00:05Z","operation":"candidate","content":"rejected no score","type":"directive","importance":7,"admission_decision":"dedup_rejected","gate_details":"","latency_ms":3,"caller":"agent-self"}
{"timestamp":"2026-09-05T00:00:06Z","operation":"retrieve","query":"ignored","strategy":"semantic_search","latency_ms":9,"results":[{"id":"z","content":"c","score":1.0}]}
this is not json
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestLoadCandidates_SkipsRetrieveMalformed(t *testing.T) {
	path := candidateFixture(t)
	recs, err := LoadCandidates(path)
	if err != nil {
		t.Fatalf("LoadCandidates: %v", err)
	}
	if len(recs) != 6 {
		t.Fatalf("expected 6 candidate records, got %d", len(recs))
	}
	// task_id is carried through.
	if recs[0].TaskID != "loop-42" {
		t.Errorf("record 0 task_id = %q, want loop-42", recs[0].TaskID)
	}
	// dedup score parsed from gate_details on rejection.
	if !recs[1].HasDedupScore || recs[1].DedupScore != 0.95 || recs[1].DedupAgainstID != "aaa" {
		t.Errorf("record 1 dedup parse wrong: %+v", recs[1])
	}
	// rejected with empty gate_details → no parseable score.
	if recs[5].HasDedupScore {
		t.Errorf("record 5 should have no dedup score, got %+v", recs[5])
	}
	// admitted record carries no dedup score.
	if recs[0].HasDedupScore {
		t.Errorf("admitted record should not carry a dedup score")
	}
}

func TestLoadCandidates_DoesNotAffectLoadTrace(t *testing.T) {
	// The same file loaded via LoadTrace must yield ONLY the retrieve record
	// (retrieve-only contract preserved).
	path := candidateFixture(t)
	cases, err := LoadTrace(path)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	if len(cases) != 1 {
		t.Fatalf("LoadTrace should yield 1 retrieve case, got %d", len(cases))
	}
	if cases[0].Query != "ignored" {
		t.Errorf("wrong retrieve query: %q", cases[0].Query)
	}
}

func TestBuildCandidateReport_DedupFlip(t *testing.T) {
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	global, hasGlobal, perType, err := ParseDedupOverride("0.90")
	if err != nil {
		t.Fatal(err)
	}
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{
		DedupGlobal: global, HasDedupGlobal: hasGlobal, DedupPerType: perType,
	})

	if rep.TotalCandidates != 6 {
		t.Errorf("total = %d, want 6", rep.TotalCandidates)
	}
	if rep.Admitted != 1 || rep.DedupRejected != 3 || rep.RateLimited != 1 || rep.Errored != 1 {
		t.Errorf("census wrong: %+v", rep)
	}
	// Under threshold 0.90: score 0.95 stays rejected; score 0.89 flips to
	// admitted; the empty-gate rejection is unresolvable. The 1 admitted
	// record is unresolvable (no top score).
	if len(rep.NewlyAdmitted) != 1 {
		t.Fatalf("newly_admitted = %d, want 1", len(rep.NewlyAdmitted))
	}
	if rep.NewlyAdmitted[0].Score != 0.89 || rep.NewlyAdmitted[0].AgainstID != "bbb" {
		t.Errorf("wrong newly_admitted delta: %+v", rep.NewlyAdmitted[0])
	}
	if len(rep.NewlyRejected) != 0 {
		t.Errorf("newly_rejected should be empty, got %d", len(rep.NewlyRejected))
	}
	// unresolvable = 1 admitted + 1 rejected-without-score.
	if rep.UnresolvableDedup != 2 || rep.Unresolvable != 2 {
		t.Errorf("unresolvable = %d (dedup %d), want 2", rep.Unresolvable, rep.UnresolvableDedup)
	}
}

func TestBuildCandidateReport_PerTypeThreshold(t *testing.T) {
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	// Only override insight; directive/event get no dedup override.
	_, _, perType, err := ParseDedupOverride("insight=0.96")
	if err != nil {
		t.Fatal(err)
	}
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{DedupPerType: perType})
	// insight rejections: 0.95 and 0.89 both < 0.96 → both flip. admitted
	// insight is unresolvable. directive rejection (no override) untouched.
	if len(rep.NewlyAdmitted) != 2 {
		t.Fatalf("newly_admitted = %d, want 2", len(rep.NewlyAdmitted))
	}
	// ordering is highest score first.
	if rep.NewlyAdmitted[0].Score < rep.NewlyAdmitted[1].Score {
		t.Errorf("newly_admitted not sorted desc: %+v", rep.NewlyAdmitted)
	}
	if rep.UnresolvableDedup != 1 {
		t.Errorf("unresolvable_dedup = %d, want 1 (the admitted insight)", rep.UnresolvableDedup)
	}
}

func TestBuildCandidateReport_ImportanceBounds(t *testing.T) {
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	// insight recorded importances: 6, 6, 7. New bounds [5,6.5]:
	//   6 in-bounds (unresolvable x2), 7 > 6.5 → reclamp to 6.5.
	bounds, err := ParseImportanceBounds("insight=5:6.5")
	if err != nil {
		t.Fatal(err)
	}
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{ImportanceBounds: bounds})
	if len(rep.ImportanceReclamped) != 1 {
		t.Fatalf("importance_reclamped = %d, want 1", len(rep.ImportanceReclamped))
	}
	if rep.ImportanceReclamped[0].NewImportance != 6.5 || rep.ImportanceReclamped[0].OldImportance != 7 {
		t.Errorf("wrong reclamp: %+v", rep.ImportanceReclamped[0])
	}
	if rep.UnresolvableImportance != 2 {
		t.Errorf("unresolvable_importance = %d, want 2", rep.UnresolvableImportance)
	}
}

func TestBuildCandidateReport_NoOverrideIsCensusOnly(t *testing.T) {
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{})
	if len(rep.NewlyAdmitted) != 0 || rep.Unresolvable != 0 {
		t.Errorf("no override should produce no diff/unresolvable, got %+v", rep)
	}
	if rep.TotalCandidates != 6 {
		t.Errorf("census total = %d, want 6", rep.TotalCandidates)
	}
}

func TestBuildCandidateReport_NoDoubleCountUnresolvable(t *testing.T) {
	// When BOTH dedup and importance overrides are given, a record unresolvable
	// in both dimensions must count ONCE in Unresolvable (distinct records), so
	// Unresolvable never exceeds TotalCandidates.
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	g, hg, pt, _ := ParseDedupOverride("0.90")
	bounds, _ := ParseImportanceBounds("insight=5:6.5")
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{
		DedupGlobal: g, HasDedupGlobal: hg, DedupPerType: pt, ImportanceBounds: bounds,
	})
	if rep.Unresolvable > rep.TotalCandidates {
		t.Fatalf("Unresolvable %d exceeds TotalCandidates %d (double-count)", rep.Unresolvable, rep.TotalCandidates)
	}
	// The admitted insight (imp 6) is unresolvable in BOTH dimensions → counts once.
	// dedup-unresolvable: admitted insight(imp6) + rejected-no-score directive = 2.
	// importance-unresolvable: the two insights with imp 6 (in [5,6.5]) = 2.
	// distinct unresolvable records: admitted-insight(6), rejected-no-score-directive,
	// rejected-insight(0.89, imp7 → reclamped not unres), rejected-insight(0.95, imp6 in-bounds unres).
	if rep.UnresolvableDedup != 2 || rep.UnresolvableImportance != 2 {
		t.Errorf("dim counts: dedup=%d imp=%d, want 2/2", rep.UnresolvableDedup, rep.UnresolvableImportance)
	}
	if rep.Unresolvable != 3 {
		t.Errorf("distinct Unresolvable = %d, want 3", rep.Unresolvable)
	}
}

func TestParseDedupOverride(t *testing.T) {
	// global
	g, hg, pt, err := ParseDedupOverride("0.88")
	if err != nil || !hg || g != 0.88 || pt != nil {
		t.Fatalf("global parse: g=%v hg=%v pt=%v err=%v", g, hg, pt, err)
	}
	// per-type
	_, hg, pt, err = ParseDedupOverride("insight=0.90,directive=0.92")
	if err != nil || hg || pt["insight"] != 0.90 || pt["directive"] != 0.92 {
		t.Fatalf("per-type parse: hg=%v pt=%v err=%v", hg, pt, err)
	}
	// empty
	_, hg, _, err = ParseDedupOverride("")
	if err != nil || hg {
		t.Fatalf("empty parse: hg=%v err=%v", hg, err)
	}
	// invalid cases
	for _, bad := range []string{"abc", "2.0", "insight=xyz", "bogus=0.9", "insight=1.5"} {
		if _, _, _, err := ParseDedupOverride(bad); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}
}

func TestParseImportanceBounds(t *testing.T) {
	b, err := ParseImportanceBounds("insight=5:8,directive=6:10")
	if err != nil {
		t.Fatal(err)
	}
	if b["insight"] != [2]float64{5, 8} || b["directive"] != [2]float64{6, 10} {
		t.Fatalf("wrong bounds: %+v", b)
	}
	if got, _ := ParseImportanceBounds(""); got != nil {
		t.Errorf("empty should be nil, got %+v", got)
	}
	for _, bad := range []string{"insight=8:5", "insight=5", "bogus=5:8", "insight=a:b", "insight"} {
		if _, err := ParseImportanceBounds(bad); err == nil {
			t.Errorf("expected error for %q", bad)
		}
	}
}

func TestRenderCandidate_Roundtrips(t *testing.T) {
	path := candidateFixture(t)
	recs, _ := LoadCandidates(path)
	g, hg, pt, _ := ParseDedupOverride("0.90")
	rep := BuildCandidateReport([]string{path}, recs, CandidateOptions{DedupGlobal: g, HasDedupGlobal: hg, DedupPerType: pt})
	if _, err := RenderCandidateJSON(rep); err != nil {
		t.Fatalf("RenderCandidateJSON: %v", err)
	}
	md := RenderCandidateMarkdown(rep)
	if len(md) == 0 {
		t.Fatal("markdown empty")
	}
}
