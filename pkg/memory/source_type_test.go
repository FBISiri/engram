package memory

import "testing"

func TestIsValidSourceType(t *testing.T) {
	valid := []string{"tool_output", "reflection", "web_search", "user_input", "calendar", "document"}
	for _, s := range valid {
		if !IsValidSourceType(s) {
			t.Errorf("IsValidSourceType(%q) = false, want true", s)
		}
	}

	invalid := []string{"", "tool", "TOOL_OUTPUT", "web", "unknown", "user"}
	for _, s := range invalid {
		if IsValidSourceType(s) {
			t.Errorf("IsValidSourceType(%q) = true, want false", s)
		}
	}
}

func TestValidateSourceType(t *testing.T) {
	for st := range ValidSourceTypes {
		if err := ValidateSourceType(string(st)); err != nil {
			t.Errorf("ValidateSourceType(%q) returned error: %v", st, err)
		}
	}

	for _, s := range []string{"", "nope", "Reflection"} {
		if err := ValidateSourceType(s); err == nil {
			t.Errorf("ValidateSourceType(%q) = nil, want error", s)
		}
	}
}

func TestValidSourceTypesComplete(t *testing.T) {
	// Guard against accidental enum drift.
	if len(ValidSourceTypes) != 6 {
		t.Errorf("expected 6 valid source types, got %d", len(ValidSourceTypes))
	}
}
