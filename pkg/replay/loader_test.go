package replay

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadTrace_RealTrajectory(t *testing.T) {
	path := filepath.Join("..", "..", "trajectories", "2026-08-02.jsonl")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("trajectory fixture not in repo")
	}
	cases, err := LoadTrace(path)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	if len(cases) == 0 {
		t.Fatal("expected at least one retrieve case")
	}
	for i, c := range cases {
		if c.Query == "" {
			t.Errorf("case %d has empty query", i)
		}
		if c.Strategy != "semantic_search" {
			t.Errorf("case %d strategy = %q, want semantic_search", i, c.Strategy)
		}
	}
	// First recorded case in the file is a known retrieve with 6 results.
	first := cases[0]
	if len(first.RecordedResults) == 0 {
		t.Error("first case has no recorded results")
	}
	if first.RecordedResults[0].ID == "" || first.RecordedResults[0].Score == 0 {
		t.Errorf("first result missing id/score: %+v", first.RecordedResults[0])
	}
	if first.RecordedLatency <= 0 {
		t.Errorf("first case latency = %d, want > 0", first.RecordedLatency)
	}
}

func TestLoadTrace_SkipsMalformedAndNonRetrieve(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mixed.jsonl")
	content := `{"timestamp":"2026-08-02T00:00:00Z","operation":"retrieve","query":"q1","strategy":"semantic_search","latency_ms":10,"results":[{"id":"a","content":"c","score":1.0}]}
this is not json
{"timestamp":"2026-08-02T00:00:01Z","operation":"update","content":"noise"}
{"timestamp":"2026-08-02T00:00:02Z","operation":"retrieve","query":"q2","latency_ms":20}
`
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	cases, err := LoadTrace(path)
	if err != nil {
		t.Fatalf("LoadTrace: %v", err)
	}
	if len(cases) != 2 {
		t.Fatalf("expected 2 retrieve cases, got %d", len(cases))
	}
	if cases[0].Query != "q1" || cases[1].Query != "q2" {
		t.Errorf("wrong queries: %q %q", cases[0].Query, cases[1].Query)
	}
}
