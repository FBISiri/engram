package reflection

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/llm"
)

// P1 regression: the real production dump must now parse. The fixture holds a
// malformed placeholder object (with the structural `";` defect) followed by a
// prose separator and the actual complete JSON object; first-valid-block
// extraction must recover the real object.
func TestParseDialecticResponse_FixtureRecovers(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "dialectic-q2-217584911.txt"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}

	pq := PerQuestionEvidence{
		Question: "q2",
		Evidence: makeEvidence(
			"37bd6188-df4", "f7d12119-a45", "ac40d532-a8a", "53d5823e-1c3",
			"dd3aef6d-a0c", "bd6736d4-9ec", "180972c5-725", "ffd06c12-e03",
		),
	}

	insight, err := parseDialecticResponse(string(data), pq, llm.Meta{}, "dialectic-q2")
	if err != nil {
		t.Fatalf("expected fixture to parse, got error: %v", err)
	}
	if !strings.HasPrefix(insight.Content, "All three symptom lines") {
		t.Errorf("unexpected content: %.60q", insight.Content)
	}
	if insight.Importance != 9 {
		t.Errorf("expected importance=9, got %d", insight.Importance)
	}
	wantTags := []string{"architecture", "blackboard-pattern", "concurrency", "root-cause", "engram"}
	if len(insight.Tags) != len(wantTags) {
		t.Fatalf("expected %d tags, got %v", len(wantTags), insight.Tags)
	}
	for i, tag := range wantTags {
		if insight.Tags[i] != tag {
			t.Errorf("tag[%d]=%q, want %q", i, insight.Tags[i], tag)
		}
	}
	if len(insight.SourceIDs) != 8 {
		t.Errorf("expected 8 source_ids, got %d", len(insight.SourceIDs))
	}
}

// A legitimate `";` sequence INSIDE a string value must NOT be corrupted by the
// delimiter-repair stage. This input is already valid JSON, so it takes the
// strict fast path AND the repair function itself must be a no-op on it.
func TestParseDialecticResponse_LegitSemicolonNotCorrupted(t *testing.T) {
	const content = `he said "run"; then stopped`
	resp := `{"content":"he said \"run\"; then stopped",` +
		`"tensions":["t1"],"source_ids":["e1","e2"],` +
		`"confidence":0.5,"importance":4,"tags":["x"]}`

	// Repair must be a no-op: the `";` lives inside a string literal.
	if got := repairStructuralSemicolons(resp); got != resp {
		t.Errorf("repair corrupted in-string semicolon:\n got=%q\nwant=%q", got, resp)
	}

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	insight, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if insight.Content != content {
		t.Errorf("content mangled: got %q want %q", insight.Content, content)
	}
}

// Prose prologue/epilogue around a valid object must be tolerated via
// first-valid-block extraction.
func TestParseDialecticResponse_ProseWrapped(t *testing.T) {
	resp := "Sure, here is the synthesis you asked for:\n" +
		`{"content":"prose-wrapped insight","tensions":[],` +
		`"source_ids":["e1","e2"],"confidence":0.7,"importance":5,"tags":["p"]}` +
		"\nLet me know if you need anything else."

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	insight, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if insight.Content != "prose-wrapped insight" {
		t.Errorf("unexpected content: %q", insight.Content)
	}
}

// Genuine garbage must still fail cleanly (dump path). HOME guard keeps the
// dump inside t.TempDir() so we never pollute the production dir.
func TestParseDialecticResponse_GarbageStillFails(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	if _, err := parseDialecticResponse(`{broken json!!!`, pq, llm.Meta{}, "dialectic-q1"); err == nil {
		t.Fatal("expected garbage to fail parsing")
	}
	if _, err := parseDialecticResponse(`not valid json at all`, pq, llm.Meta{}, "dialectic-q1"); err == nil {
		t.Fatal("expected garbage to fail parsing")
	}
}

// The delimiter-repair stage must recover an object whose sole defect is a
// structural `";` (quote-terminated value followed by `;` before the next key),
// and lenient parsing must NOT weaken the source_ids injection defense.
func TestParseDialecticResponse_DelimiterRepair(t *testing.T) {
	resp := `{"content":"repaired value";` +
		`"tensions":["t"],"source_ids":["e1","e2"],` +
		`"confidence":0.6,"importance":3,"tags":["r"]}`

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	insight, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected delimiter repair to recover, got: %v", err)
	}
	if insight.Content != "repaired value" {
		t.Errorf("unexpected content: %q", insight.Content)
	}

	// Injection defense still runs on recovered objects.
	bad := `{"content":"x";"tensions":["t"],"source_ids":["e1","INJECTED"],` +
		`"confidence":0.6,"importance":3,"tags":["r"]}`
	if _, err := parseDialecticResponse(bad, pq, llm.Meta{}, "dialectic-q1"); err == nil ||
		!strings.Contains(err.Error(), "prompt injection") {
		t.Errorf("expected prompt injection error after repair, got: %v", err)
	}
}
