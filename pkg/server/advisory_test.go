package server

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestAdvisoryJSONSerialization verifies field names and Data serialization.
func TestAdvisoryJSONSerialization(t *testing.T) {
	a := Advisory{
		Type:     "similar_memory",
		Severity: "info",
		Message:  "hello",
		Data:     map[string]any{"score": 0.78},
	}
	b, err := json.Marshal(a)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	s := string(b)
	for _, want := range []string{`"type":"similar_memory"`, `"severity":"info"`, `"message":"hello"`, `"data":`, `"score":0.78`} {
		if !strings.Contains(s, want) {
			t.Errorf("expected %q in %s", want, s)
		}
	}
}

// TestAdvisoryDataOmitempty verifies Data is omitted when nil.
func TestAdvisoryDataOmitempty(t *testing.T) {
	a := Advisory{Type: "tags_missing", Severity: "info", Message: "no tags"}
	b, _ := json.Marshal(a)
	if strings.Contains(string(b), "data") {
		t.Errorf("expected no data key, got %s", b)
	}
}

// TestAdvisoriesOmitemptyInResult verifies the addResult-style wrapper omits the
// advisories key when empty (backward compat).
func TestAdvisoriesOmitemptyInResult(t *testing.T) {
	type wrapper struct {
		Status     string     `json:"status"`
		Advisories []Advisory `json:"advisories,omitempty"`
	}
	empty, _ := json.Marshal(wrapper{Status: "created"})
	if strings.Contains(string(empty), "advisories") {
		t.Errorf("empty advisories must be omitted, got %s", empty)
	}
	full, _ := json.Marshal(wrapper{Status: "created", Advisories: []Advisory{{Type: "x", Severity: "info", Message: "m"}}})
	if !strings.Contains(string(full), "advisories") {
		t.Errorf("non-empty advisories must be present, got %s", full)
	}
}
