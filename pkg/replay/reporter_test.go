package replay

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/trajectory"
)

func sampleReport() Report {
	results := []ReplayResult{
		{
			Case: &ReplayCase{
				Query:           "hello",
				RecordedResults: []trajectory.ResultItem{{ID: "a", Score: 1.0}, {ID: "b", Score: 0.9}},
				RecordedLatency: 10,
			},
			LiveResults: []trajectory.ResultItem{{ID: "a", Score: 1.0}},
			LiveLatency: 12,
		},
	}
	cfg := MemoryConfig{RetrieveConfig: RetrieveConfig{RecencyWeight: 0.25}}
	return BuildReport("trajectories/x.jsonl", "engram_eval_replay", cfg, results, DefaultThresholds())
}

func TestRenderJSON_RoundTrips(t *testing.T) {
	r := sampleReport()
	data, err := RenderJSON(r)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var back Report
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("unmarshal JSON report: %v", err)
	}
	if back.Collection != "engram_eval_replay" {
		t.Errorf("collection lost: %q", back.Collection)
	}
	if back.Aggregate.TotalCases != 1 {
		t.Errorf("total cases = %d, want 1", back.Aggregate.TotalCases)
	}
	if len(back.Comparisons) != 1 || back.Comparisons[0].Query != "hello" {
		t.Errorf("comparisons lost: %+v", back.Comparisons)
	}
}

func TestRenderMarkdown_HasSections(t *testing.T) {
	md := RenderMarkdown(sampleReport())
	for _, want := range []string{
		"# Replay Report",
		"## Config Diff",
		"## Aggregate Metrics",
		"## Recall Histogram",
		"## Latency Comparison",
		"## Regressions",
		"## Known Limitations",
		"Verdict:",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown missing section %q\n---\n%s", want, md)
		}
	}
}
