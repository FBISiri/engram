package reflection

import (
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── W17 v1.1 batch 2 — Evidence grounding enforcement ──────────────────────

// TestValidateEvidenceGrounding covers the len=0, 1, 2, 3 boundary cases for
// the minEvidenceCount floor (set to 2). Batches with < 2 memories MUST be
// rejected; batches with >= 2 MUST pass.
func TestValidateEvidenceGrounding_Len0(t *testing.T) {
	err := validateEvidenceGrounding(nil)
	if err == nil {
		t.Fatal("expected error for empty batch (len=0), got nil")
	}
	if !strings.Contains(err.Error(), "insufficient source memories") {
		t.Errorf("unexpected error text: %q", err.Error())
	}
}

func TestValidateEvidenceGrounding_Len1(t *testing.T) {
	batch := []memory.Memory{
		{ID: "a", Content: "only source", Importance: 5},
	}
	err := validateEvidenceGrounding(batch)
	if err == nil {
		t.Fatal("expected error for single-source batch (len=1), got nil")
	}
	if !strings.Contains(err.Error(), "1 < 2") {
		t.Errorf("error should mention 1 < 2 floor, got %q", err.Error())
	}
}

func TestValidateEvidenceGrounding_Len2(t *testing.T) {
	batch := []memory.Memory{
		{ID: "a", Content: "first", Importance: 5},
		{ID: "b", Content: "second", Importance: 6},
	}
	if err := validateEvidenceGrounding(batch); err != nil {
		t.Errorf("len=2 should pass (boundary), got error: %v", err)
	}
}

func TestValidateEvidenceGrounding_Len3(t *testing.T) {
	batch := []memory.Memory{
		{ID: "a", Content: "first", Importance: 5},
		{ID: "b", Content: "second", Importance: 6},
		{ID: "c", Content: "third", Importance: 7},
	}
	if err := validateEvidenceGrounding(batch); err != nil {
		t.Errorf("len=3 should pass, got error: %v", err)
	}
}

// ── W17 v1.1 batch 2 — Single-event prompt construction ────────────────────

func TestBuildSingleEventPrompt_ContainsCauseAndSummary(t *testing.T) {
	in := SingleEventInput{
		Cause:   TriggerTaskFailure,
		Summary: "event-loop retry exhausted for task X",
	}
	p := buildSingleEventPrompt(in)
	if !strings.Contains(p, string(TriggerTaskFailure)) {
		t.Error("prompt should include the trigger cause")
	}
	if !strings.Contains(p, "event-loop retry exhausted") {
		t.Error("prompt should include the event summary verbatim")
	}
	if !strings.Contains(p, "EXACTLY ONE insight") {
		t.Error("prompt should request exactly one insight")
	}
}

func TestBuildSingleEventPrompt_IncludesEvidenceWhenPresent(t *testing.T) {
	in := SingleEventInput{
		Cause:       TriggerUserCorrection,
		Summary:     "Frank corrected calendar-event creation pattern",
		EvidenceIDs: []string{"mem-a", "mem-b"},
	}
	p := buildSingleEventPrompt(in)
	if !strings.Contains(p, "mem-a") || !strings.Contains(p, "mem-b") {
		t.Error("prompt should include provided evidence IDs")
	}
}

func TestBuildSingleEventPrompt_OmitsEvidenceHeaderWhenEmpty(t *testing.T) {
	in := SingleEventInput{
		Cause:   TriggerExternalEvent,
		Summary: "external event with no grounding memories",
	}
	p := buildSingleEventPrompt(in)
	if strings.Contains(p, "Related memory IDs") {
		t.Error("prompt should omit evidence header when EvidenceIDs is empty")
	}
}

// ── W17 v1.1 batch 2 — RunSingleEvent precondition checks ──────────────────

func TestRunSingleEvent_RejectsEmptyCause(t *testing.T) {
	eng := NewEngine(nil, nil, DefaultConfig())
	_, err := eng.RunSingleEvent(nil, SingleEventInput{
		Summary: "missing cause",
	})
	if err == nil {
		t.Fatal("expected error for empty Cause, got nil")
	}
	if !strings.Contains(err.Error(), "Cause is required") {
		t.Errorf("unexpected error text: %q", err.Error())
	}
}

func TestRunSingleEvent_RejectsEmptySummary(t *testing.T) {
	eng := NewEngine(nil, nil, DefaultConfig())
	_, err := eng.RunSingleEvent(nil, SingleEventInput{
		Cause:   TriggerTaskFailure,
		Summary: "   ",
	})
	if err == nil {
		t.Fatal("expected error for blank Summary, got nil")
	}
	if !strings.Contains(err.Error(), "Summary is required") {
		t.Errorf("unexpected error text: %q", err.Error())
	}
}
