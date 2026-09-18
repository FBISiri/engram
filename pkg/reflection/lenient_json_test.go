package reflection

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/llm"
)

func readFixture(t *testing.T, name string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return string(data)
}

// TestParseLenientJSON_Variants is the R5 table-driven regression across the
// whole defect family. Each real production dump (copied into testdata) must be
// recovered by the shared ladder; variant (3) is exercised via an inline case
// mirroring dialectic-q2-217584911.txt (also covered by dialectic_lenient_test).
func TestParseLenientJSON_Variants(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantStage string
	}{
		{
			// variant (1): bare unescaped quotes inside string values, FOCAL ARRAY.
			// This is today's `invalid character 'å' after array element` failure.
			name:      "variant1_bare_quote_focal_array",
			input:     readFixture(t, "focal-3644203320.txt"),
			wantStage: "bare-quote escaping",
		},
		{
			// variant (2): bare unescaped newline inside a string value.
			name:      "variant2_bare_newline_object",
			input:     readFixture(t, "dialectic-q1-2643121867.txt"),
			wantStage: "control-char sanitization",
		},
		{
			// variant (3): the `";` structural-separator typo.
			name:      "variant3_structural_semicolon",
			input:     `{"content":"repaired value";"source_ids":["e1","e2"]}`,
			wantStage: `delimiter repair (";"->",")`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if strings.HasPrefix(tc.input, "[") {
				var out []string
				stage, err := parseLenientJSON(tc.input, &out)
				if err != nil {
					t.Fatalf("expected recovery, got error: %v", err)
				}
				if stage != tc.wantStage {
					t.Errorf("stage=%q, want %q", stage, tc.wantStage)
				}
				if len(out) == 0 {
					t.Error("expected non-empty array")
				}
			} else {
				var out dialecticLLMResponse
				stage, err := parseLenientJSON(tc.input, &out)
				if err != nil {
					t.Fatalf("expected recovery, got error: %v", err)
				}
				if stage != tc.wantStage {
					t.Errorf("stage=%q, want %q", stage, tc.wantStage)
				}
				if out.Content == "" {
					t.Error("expected non-empty content")
				}
			}
		})
	}
}

// escapeBareQuotes must be a no-op on JSON whose quotes are already correctly
// escaped (the `\"run\"; then stopped` fixture from dialectic_lenient_test).
func TestEscapeBareQuotes_NoOpOnValid(t *testing.T) {
	valid := `{"content":"he said \"run\"; then stopped","source_ids":["e1","e2"]}`
	if got := escapeBareQuotes(valid); got != valid {
		t.Errorf("escapeBareQuotes corrupted valid input:\n got=%q\nwant=%q", got, valid)
	}
}

// variant (1) end to end through the FOCAL path: the array of question strings
// with bare inner quotes must be recovered rather than dropping the batch.
func TestParseFocalResponse_BareQuoteFixture(t *testing.T) {
	insight, err := parseFocalResponse(readFixture(t, "focal-3644203320.txt"), 3, llm.Meta{})
	if err != nil {
		t.Fatalf("expected focal fixture to parse after repair, got: %v", err)
	}
	if len(insight) != 3 {
		t.Fatalf("expected 3 questions, got %d", len(insight))
	}
	for i, q := range insight {
		if strings.TrimSpace(q) == "" {
			t.Errorf("question[%d] is empty", i)
		}
	}
}

// variant (1) end to end through the DIALECTIC path: junk method-chain after the
// content value plus a truncated source_ids array. Structural salvage recovers
// content + tensions + the two complete ids; the truncated id is dropped by the
// injection defense, leaving >= 2 valid ids.
func TestParseDialecticResponse_SalvageFixture(t *testing.T) {
	pq := PerQuestionEvidence{
		Question: "q3",
		Evidence: makeEvidence("56c4ffe3-69d", "0651ca7c-7c0"),
	}
	insight, warnings, err := parseDialecticResponse(
		readFixture(t, "dialectic-q3-669518171.txt"), pq, llm.Meta{}, "dialectic-q3")
	if err != nil {
		t.Fatalf("expected q3 fixture to parse after repair, got: %v (warnings=%v)", err, warnings)
	}
	if !strings.HasPrefix(insight.Content, "三类看似独立") {
		t.Errorf("unexpected content prefix: %.30q", insight.Content)
	}
	if len(insight.SourceIDs) != 2 {
		t.Errorf("expected 2 valid source_ids, got %v", insight.SourceIDs)
	}
}

// variant (2) end to end through the DIALECTIC path: a bare newline inside the
// content value; control-char sanitization recovers the whole object.
func TestParseDialecticResponse_BareNewlineFixture(t *testing.T) {
	pq := PerQuestionEvidence{
		Question: "q1",
		Evidence: makeEvidence(
			"6f15d1be-ef1", "77a31ab1-2cf", "505a0cca-e21",
			"35b45faf-ac4", "dc0aafb1-971", "fa5b569a-040",
		),
	}
	insight, _, err := parseDialecticResponse(
		readFixture(t, "dialectic-q1-2643121867.txt"), pq, llm.Meta{}, "dialectic-q1")
	if err != nil {
		t.Fatalf("expected q1 newline fixture to parse after repair, got: %v", err)
	}
	if !strings.HasPrefix(insight.Content, "Every 'false green'") {
		t.Errorf("unexpected content prefix: %.30q", insight.Content)
	}
	if len(insight.SourceIDs) != 6 {
		t.Errorf("expected 6 source_ids, got %d", len(insight.SourceIDs))
	}
}

// Genuine garbage through the FOCAL path must still fail cleanly, and the
// returned error MUST include the raw dump path (R4). HOME guard keeps the dump
// inside t.TempDir() so we never pollute the production dir.
func TestParseFocalResponse_GarbageFailsWithDumpPath(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	_, err := parseFocalResponse(`not valid json at all`, 3, llm.Meta{})
	if err == nil {
		t.Fatal("expected garbage to fail parsing")
	}
	if !strings.Contains(err.Error(), "raw dump:") {
		t.Errorf("focal error must include raw dump path, got: %v", err)
	}
}
