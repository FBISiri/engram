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

	insight, _, err := parseDialecticResponse(string(data), pq, llm.Meta{}, "dialectic-q2")
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
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
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
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
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
	if _, _, err := parseDialecticResponse(`{broken json!!!`, pq, llm.Meta{}, "dialectic-q1"); err == nil {
		t.Fatal("expected garbage to fail parsing")
	}
	if _, _, err := parseDialecticResponse(`not valid json at all`, pq, llm.Meta{}, "dialectic-q1"); err == nil {
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
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected delimiter repair to recover, got: %v", err)
	}
	if insight.Content != "repaired value" {
		t.Errorf("unexpected content: %q", insight.Content)
	}

	// Injection defense still runs on recovered objects. Under the new
	// degrade-gracefully contract the injected id is DROPPED (never trusted),
	// leaving only 1 valid id (< 2), so the question is discarded with an error
	// while the warnings observe the drop. The injected id must never enter output.
	bad := `{"content":"x";"tensions":["t"],"source_ids":["e1","INJECTED"],` +
		`"confidence":0.6,"importance":3,"tags":["r"]}`
	badInsight, warnings, err := parseDialecticResponse(bad, pq, llm.Meta{}, "dialectic-q1")
	if badInsight != nil || err == nil {
		t.Fatalf("expected nil insight + error for injected id, got insight=%v err=%v", badInsight, err)
	}
	var mentioned bool
	for _, w := range warnings {
		if strings.Contains(w, "INJECTED") && strings.Contains(w, "prompt injection") {
			mentioned = true
		}
	}
	if !mentioned {
		t.Errorf("expected a warning naming INJECTED as dropped for prompt injection, got: %v", warnings)
	}
}

// R5 MANDATORY regression: ONE garbled/truncated id among two good ids must NOT
// void the whole insight. The bad id is dropped (with a warning naming it) and
// the insight keeps exactly the two good ids.
func TestParseDialecticResponse_OneBadIDDropped(t *testing.T) {
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["37bd6188-df4","7b8cfbf259-e9b","f7d12119-a45"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"37bd6188-df4", "f7d12119-a45", "ac40d532-a8a",
	)}
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q3")
	if err != nil {
		t.Fatalf("expected parse to succeed after dropping one bad id, got: %v", err)
	}
	want := []string{"37bd6188-df4", "f7d12119-a45"}
	if len(insight.SourceIDs) != len(want) {
		t.Fatalf("expected %v source_ids, got %v", want, insight.SourceIDs)
	}
	for i, id := range want {
		if insight.SourceIDs[i] != id {
			t.Errorf("source_id[%d]=%q, want %q", i, insight.SourceIDs[i], id)
		}
	}
	var named bool
	for _, w := range warnings {
		if strings.Contains(w, "7b8cfbf259-e9b") {
			named = true
		}
	}
	if !named {
		t.Errorf("expected a warning naming the dropped id 7b8cfbf259-e9b, got: %v", warnings)
	}
}

// All ids invalid -> nil insight + error + warnings naming each bad id.
func TestParseDialecticResponse_AllIDsInvalid(t *testing.T) {
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["bogus-one","bogus-two"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if insight != nil || err == nil {
		t.Fatalf("expected nil insight + error, got insight=%v err=%v", insight, err)
	}
	joined := strings.Join(warnings, " ")
	if !strings.Contains(joined, "bogus-one") || !strings.Contains(joined, "bogus-two") {
		t.Errorf("expected warnings naming both bad ids, got: %v", warnings)
	}
}

// Missing tensions degrades gracefully: parses to an insight with an empty
// (non-nil) tensions slice and no error.
func TestParseDialecticResponse_MissingTensionsDegrades(t *testing.T) {
	resp := `{"content":"synthesis","source_ids":["e1","e2"],` +
		`"confidence":0.5,"importance":4,"tags":["x"]}`

	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence("e1", "e2")}
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected missing tensions to degrade, got: %v", err)
	}
	if insight.Tensions == nil {
		t.Error("expected non-nil empty tensions slice")
	}
	if len(insight.Tensions) != 0 {
		t.Errorf("expected empty tensions, got %v", insight.Tensions)
	}
}

// Prefix repair: a truncated-but-unique prefix (>= 8 chars) resolves to the
// canonical full evidence id; an ambiguous prefix is dropped, not trusted.
func TestParseDialecticResponse_PrefixRepair(t *testing.T) {
	// "abcdef01-longtail-1" and "abcdef01-longtail-2" share the "abcdef01" prefix.
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"abcdef0123456789-unique", "99998888-second", "77776666-third",
	)}

	// Unique truncated prefix repairs to the full id.
	good := `{"content":"c","tensions":["t"],` +
		`"source_ids":["abcdef01234","99998888-second"],` +
		`"confidence":0.5,"importance":4,"tags":["x"]}`
	insight, warnings, err := parseDialecticResponse(good, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected prefix repair to succeed, got: %v (warnings=%v)", err, warnings)
	}
	if insight.SourceIDs[0] != "abcdef0123456789-unique" {
		t.Errorf("expected prefix repaired to full id, got %q", insight.SourceIDs[0])
	}

	// Ambiguous prefix (matches >1 full id) must be dropped, not trusted.
	amb := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"sharedpre-aaa-1", "sharedpre-bbb-2", "cleanid-3",
	)}
	resp := `{"content":"c","tensions":["t"],` +
		`"source_ids":["sharedpre","cleanid-3","sharedpre-aaa-1"],` +
		`"confidence":0.5,"importance":4,"tags":["x"]}`
	ins2, warn2, err2 := parseDialecticResponse(resp, amb, llm.Meta{}, "dialectic-q1")
	if err2 != nil {
		t.Fatalf("expected 2 valid ids after dropping ambiguous prefix, got: %v", err2)
	}
	for _, id := range ins2.SourceIDs {
		if id == "sharedpre" {
			t.Error("ambiguous prefix must not be trusted")
		}
	}
	if !strings.Contains(strings.Join(warn2, " "), "sharedpre") {
		t.Errorf("expected warning naming dropped ambiguous prefix, got: %v", warn2)
	}
}
