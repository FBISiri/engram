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

// TestPerTypeDedupThreshold verifies the A-MAC per-type dedup thresholds:
// a candidate at cosine 0.91 is a duplicate for directive (thr 0.90) but not
// for insight/event (thr 0.92) nor identity (thr 0.95).
func TestPerTypeDedupThreshold(t *testing.T) {
	cand := []ScoredMemory{{Score: 0.91}}

	if IsDuplicateForType(cand, TypeDirective) == nil {
		t.Errorf("directive: score 0.91 >= 0.90 should be a duplicate")
	}
	if IsDuplicateForType(cand, TypeInsight) != nil {
		t.Errorf("insight: score 0.91 < 0.92 should NOT be a duplicate")
	}
	if IsDuplicateForType(cand, TypeEvent) != nil {
		t.Errorf("event: score 0.91 < 0.92 should NOT be a duplicate")
	}
	if IsDuplicateForType(cand, TypeIdentity) != nil {
		t.Errorf("identity: score 0.91 < 0.95 should NOT be a duplicate")
	}

	// DedupThresholdForType falls back to the global default for unknown types.
	if got := DedupThresholdForType(MemoryType("bogus")); got != DefaultDedupThreshold {
		t.Errorf("unknown type threshold = %v, want %v", got, DefaultDedupThreshold)
	}
}

// TestDedupThresholdForType verifies the A-MAC MVP per-type threshold values.
func TestDedupThresholdForType(t *testing.T) {
	cases := []struct {
		t    MemoryType
		want float64
	}{
		{TypeIdentity, 0.95},
		{TypeDirective, 0.90},
		{TypeInsight, 0.92},
		{TypeEvent, 0.92},
	}
	for _, c := range cases {
		if got := DedupThresholdForType(c.t); got != c.want {
			t.Errorf("DedupThresholdForType(%s) = %v, want %v", c.t, got, c.want)
		}
	}
}

// TestDedupThreshold_MVP verifies the A-MAC MVP boundary behavior per type
// (spec §5.2): a score at the threshold is a duplicate, just below is not.
func TestDedupThreshold_MVP(t *testing.T) {
	cases := []struct {
		t        MemoryType
		below    float64 // must NOT be a duplicate
		atOrOver float64 // must BE a duplicate
	}{
		{TypeDirective, 0.89, 0.90},
		{TypeInsight, 0.91, 0.92},
		{TypeEvent, 0.91, 0.92},
		{TypeIdentity, 0.94, 0.95},
	}
	for _, c := range cases {
		if IsDuplicateForType([]ScoredMemory{{Score: c.below}}, c.t) != nil {
			t.Errorf("%s: score %v should NOT be a duplicate", c.t, c.below)
		}
		if IsDuplicateForType([]ScoredMemory{{Score: c.atOrOver}}, c.t) == nil {
			t.Errorf("%s: score %v should be a duplicate", c.t, c.atOrOver)
		}
	}
}
