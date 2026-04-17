package reflection

import (
	"context"
	"strings"
	"testing"
)

// TestRunSingleEvent_StructuralSmoke verifies the event-driven control-flow
// path wires end-to-end: preconditions pass, Haiku is invoked, the result
// is well-formed. Actual insight content / insertion path depends on prod
// credentials and embedder availability, so this is a structural check only.
//
// Accepted terminal states:
//   - InsightsCreated >= 1 (prod path with embedder)
//   - DraftsWritten >= 1  (low-confidence diversion)
//   - len(Errors) >= 1    (no embedder / no haiku creds / parse failure)
//
// What MUST NOT happen:
//   - hard error return (means upstream bug)
//   - LLMCalls != 1 (means we didn't reach Haiku)
//   - precondition-style error leaked to soft errors
func TestRunSingleEvent_StructuralSmoke(t *testing.T) {
	eng := NewEngine(nil, nil, DefaultConfig())
	result, err := eng.RunSingleEvent(context.Background(), SingleEventInput{
		Cause:   TriggerTaskFailure,
		Summary: "smoke: event-driven path wired end-to-end",
	})
	if err != nil {
		t.Fatalf("RunSingleEvent returned unexpected hard error: %v", err)
	}
	if !result.Triggered {
		t.Error("event-driven run should always set Triggered=true")
	}
	if result.LLMCalls != 1 {
		t.Errorf("expected exactly 1 LLM call, got %d", result.LLMCalls)
	}
	// Sanity: at least one terminal signal.
	terminalSignals := result.InsightsCreated + result.DraftsWritten + len(result.Errors)
	if terminalSignals == 0 {
		t.Error("expected at least one of insights/drafts/errors, got none")
	}
	for _, e := range result.Errors {
		if strings.Contains(e, "Cause is required") || strings.Contains(e, "Summary is required") {
			t.Errorf("precondition error leaked into run: %v", e)
		}
	}
}
