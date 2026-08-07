package memory

import "testing"

func TestHighestTrustSource(t *testing.T) {
	// Each source_type in isolation resolves to itself (it is the only/highest).
	cases := []struct {
		sources []string
		want    string
	}{
		{[]string{"user_input"}, "user_input"},
		{[]string{"tool_output"}, "tool_output"},
		{[]string{"web_search"}, "web_search"},
		{[]string{"document"}, "document"},
		{[]string{"calendar"}, "calendar"},
		{[]string{"reflection"}, "reflection"},
		{[]string{"unknown"}, "unknown"},
		// Empty list falls back to "unknown".
		{[]string{}, "unknown"},
		{nil, "unknown"},
		// Mixed: highest trust (lowest rank) wins.
		{[]string{"reflection", "tool_output", "user_input"}, "user_input"},
		{[]string{"calendar", "document"}, "document"},
		{[]string{"reflection", "web_search"}, "web_search"},
		// Unrecognized values are ignored.
		{[]string{"bogus", "reflection"}, "reflection"},
		{[]string{"bogus"}, "unknown"},
	}
	for _, c := range cases {
		if got := HighestTrustSource(c.sources); got != c.want {
			t.Errorf("HighestTrustSource(%v) = %q, want %q", c.sources, got, c.want)
		}
	}
}

func TestHasSourceType(t *testing.T) {
	history := []ProvenanceEntry{
		{SourceType: "tool_output", MergedAt: 1, ContentScore: 0.95},
		{SourceType: "web_search", MergedAt: 2, ContentScore: 0.93},
	}
	if !HasSourceType(history, "tool_output") {
		t.Error("expected HasSourceType to find tool_output")
	}
	if !HasSourceType(history, "web_search") {
		t.Error("expected HasSourceType to find web_search")
	}
	if HasSourceType(history, "user_input") {
		t.Error("did not expect HasSourceType to find user_input")
	}
	if HasSourceType(nil, "tool_output") {
		t.Error("did not expect HasSourceType to find anything in nil history")
	}
}
