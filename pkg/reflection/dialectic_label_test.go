package reflection

import (
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/llm"
)

// Ordinal labels [E1]..[En] map back to the full evidence UUIDs by 1-based
// position in this question's evidence set. Accepted forms: "E1" and "[E1]".
func TestParseDialecticResponse_LabelMapping(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"37bd6188-df4-full", "f7d12119-a45-full", "ac40d532-a8a-full",
	)}
	// Mix bracketed and bare forms; both are accepted.
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["[E1]","E3"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected labels to resolve, got: %v (warnings=%v)", err, warnings)
	}
	want := []string{"37bd6188-df4-full", "ac40d532-a8a-full"}
	if len(insight.SourceIDs) != len(want) {
		t.Fatalf("expected %v, got %v", want, insight.SourceIDs)
	}
	for i, id := range want {
		if insight.SourceIDs[i] != id {
			t.Errorf("source_id[%d]=%q, want %q", i, insight.SourceIDs[i], id)
		}
	}
}

// Out-of-range labels (E0, E99) are dropped, never invented. With only 1 valid
// label surviving, the >=2 rule discards the question.
func TestParseDialecticResponse_LabelOutOfRangeDropped(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"id-one", "id-two", "id-three",
	)}
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["[E0]","[E99]","E1"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if insight != nil || err == nil {
		t.Fatalf("expected nil insight + error (only 1 valid), got insight=%v err=%v", insight, err)
	}
	joined := strings.Join(warnings, " ")
	if !strings.Contains(joined, "E0") || !strings.Contains(joined, "E99") {
		t.Errorf("expected warnings naming dropped out-of-range labels, got: %v", warnings)
	}
}

// Out-of-range labels dropped but >= 2 in-range labels survive -> question kept.
func TestParseDialecticResponse_LabelOutOfRangeKeepsRest(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"id-one", "id-two", "id-three",
	)}
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["E99","E1","[E2]"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected question kept (2 valid labels), got: %v", err)
	}
	want := []string{"id-one", "id-two"}
	if len(insight.SourceIDs) != len(want) {
		t.Fatalf("expected %v, got %v", want, insight.SourceIDs)
	}
	for i, id := range want {
		if insight.SourceIDs[i] != id {
			t.Errorf("source_id[%d]=%q, want %q", i, insight.SourceIDs[i], id)
		}
	}
}

// A non-label, non-evidence token is dropped (prompt injection defense);
// with < 2 valid ids the question is discarded.
func TestParseDialecticResponse_UnknownLabelDropped(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"id-one", "id-two", "id-three",
	)}
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["E1","INJECTED_ID"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if insight != nil || err == nil {
		t.Fatalf("expected nil insight + error, got insight=%v err=%v", insight, err)
	}
	if !strings.Contains(strings.Join(warnings, " "), "INJECTED_ID") {
		t.Errorf("expected warning naming dropped INJECTED_ID, got: %v", warnings)
	}
}

// Duplicate labels collapse to a single resolved id: ["E1","E1"] alone is
// therefore dropped by the >=2 corroboration rule (a single real source must
// not fake corroboration).
func TestParseDialecticResponse_DuplicateLabelsCollapse(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"id-one", "id-two", "id-three",
	)}
	// Two references to the same evidence must not satisfy the >=2 gate.
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["E1","E1"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, _, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if insight != nil || err == nil {
		t.Fatalf("expected nil insight + error (dup collapses to 1 < 2), got insight=%v err=%v", insight, err)
	}

	// Distinct labels pointing at distinct evidence survive, and a trailing
	// duplicate collapses (no fake third source).
	resp2 := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["E1","[E1]","E2"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	ins2, _, err2 := parseDialecticResponse(resp2, pq, llm.Meta{}, "dialectic-q1")
	if err2 != nil {
		t.Fatalf("expected question kept (2 distinct ids), got: %v", err2)
	}
	want := []string{"id-one", "id-two"}
	if len(ins2.SourceIDs) != len(want) {
		t.Fatalf("expected deduped %v, got %v", want, ins2.SourceIDs)
	}
	for i, id := range want {
		if ins2.SourceIDs[i] != id {
			t.Errorf("source_id[%d]=%q, want %q", i, ins2.SourceIDs[i], id)
		}
	}
}

// Backward compat: an exact full UUID and an exact 12-char promptIDForm id are
// still accepted (older prompts echoed hex ids).
func TestParseDialecticResponse_BackwardCompatExactIDs(t *testing.T) {
	full := "abcdef0123456789-tail-unique"
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		full, "99998888-second-full",
	)}
	// full id verbatim + the 12-char prompt id form of the second evidence.
	short := promptIDForm("99998888-second-full") // "99998888-sec"
	resp := `{"content":"synthesis","tensions":["t"],` +
		`"source_ids":["` + full + `","` + short + `"],` +
		`"confidence":0.7,"importance":5,"tags":["x"]}`
	insight, warnings, err := parseDialecticResponse(resp, pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected exact full + 12-char ids to resolve, got: %v (warnings=%v)", err, warnings)
	}
	want := []string{full, short}
	if len(insight.SourceIDs) != len(want) {
		t.Fatalf("expected %v, got %v", want, insight.SourceIDs)
	}
	for i, id := range want {
		if insight.SourceIDs[i] != id {
			t.Errorf("source_id[%d]=%q, want %q", i, insight.SourceIDs[i], id)
		}
	}
}

// evidenceLabelIndex accepts "E<n>" and "[E<n>]" (uppercase E) and rejects
// everything else (lowercase e, non-numeric, bare hex ids).
func TestEvidenceLabelIndex(t *testing.T) {
	cases := []struct {
		in string
		n  int
		ok bool
	}{
		{"E1", 1, true},
		{"[E1]", 1, true},
		{" [E12] ", 12, true},
		{"E0", 0, true},
		{"e1", 0, false},
		{"E", 0, false},
		{"Ex", 0, false},
		{"37bd6188-df4", 0, false},
		{"INJECTED", 0, false},
	}
	for _, c := range cases {
		n, ok := evidenceLabelIndex(c.in)
		if ok != c.ok || (ok && n != c.n) {
			t.Errorf("evidenceLabelIndex(%q)=(%d,%v), want (%d,%v)", c.in, n, ok, c.n, c.ok)
		}
	}
}

// The updated prompt renders ordinal labels and instructs the LLM to return
// them, and no longer leaks [id=...] hex prefixes.
func TestBuildDialecticPrompt_UsesLabels(t *testing.T) {
	pq := PerQuestionEvidence{Question: "q", Evidence: makeEvidence(
		"37bd6188-df4-full", "f7d12119-a45-full",
	)}
	p := buildDialecticPrompt(pq)
	if !strings.Contains(p, "[E1]") || !strings.Contains(p, "[E2]") {
		t.Errorf("expected [E1]/[E2] labels in prompt, got:\n%s", p)
	}
	if strings.Contains(p, "[id=") {
		t.Errorf("prompt must not emit [id=...] hex prefixes:\n%s", p)
	}
	if !strings.Contains(p, "source_ids") || !strings.Contains(p, "LABELS") {
		t.Errorf("prompt must instruct returning evidence labels:\n%s", p)
	}
}
