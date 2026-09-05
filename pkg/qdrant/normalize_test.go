package qdrant

import (
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// TestNormalizePayload_ProvenanceEntries verifies that normalizePayload plus the
// NewValueMap wrapper accept a payload carrying []memory.ProvenanceEntry both
// nested under "metadata" and as a top-level dotted key, without panicking, and
// convert the struct slices into []interface{} of map[string]interface{}.
func TestNormalizePayload_ProvenanceEntries(t *testing.T) {
	history := []memory.ProvenanceEntry{
		{SourceType: "user_input", MergedAt: 100, ContentScore: 0.91},
		{SourceType: "tool_output", MergedAt: 200, ContentScore: 0.93},
	}
	fields := map[string]any{
		"metadata": map[string]any{
			"source_type":        "user_input",
			"provenance_history": history,
			"tags":               []string{"a", "b"},
		},
		"metadata.provenance_history": history,
		"updated_at":                  float64(123),
	}

	norm := normalizePayload(fields)

	// top-level dotted key
	top, ok := norm["metadata.provenance_history"].([]any)
	if !ok {
		t.Fatalf("top-level provenance_history is %T, want []any", norm["metadata.provenance_history"])
	}
	assertProvSlice(t, top)

	// nested under metadata
	md, ok := norm["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("metadata is %T, want map[string]any", norm["metadata"])
	}
	nested, ok := md["provenance_history"].([]any)
	if !ok {
		t.Fatalf("nested provenance_history is %T, want []any", md["provenance_history"])
	}
	assertProvSlice(t, nested)

	if _, ok := md["tags"].([]any); !ok {
		t.Errorf("nested tags is %T, want []any", md["tags"])
	}

	// NewValueMap wrapper must not panic and must succeed.
	payload, err := safeNewValueMap(norm)
	if err != nil {
		t.Fatalf("safeNewValueMap error: %v", err)
	}
	if payload == nil {
		t.Fatal("expected non-nil payload")
	}
}

func assertProvSlice(t *testing.T, s []any) {
	t.Helper()
	if len(s) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(s))
	}
	for i, e := range s {
		m, ok := e.(map[string]any)
		if !ok {
			t.Fatalf("element %d is %T, want map[string]any", i, e)
		}
		for _, k := range []string{"source_type", "merged_at", "content_score"} {
			if _, ok := m[k]; !ok {
				t.Errorf("element %d missing key %q", i, k)
			}
		}
	}
}

// TestSafeNewValueMap_UnsupportedType ensures the wrapper turns a client panic
// on an unsupported value type into a returned error rather than crashing.
func TestSafeNewValueMap_UnsupportedType(t *testing.T) {
	type weird struct{ X int }
	// A raw struct slice that normalizePayload does not handle stays unsupported.
	if _, err := safeNewValueMap(map[string]any{"bad": []weird{{X: 1}}}); err == nil {
		t.Fatal("expected error for unsupported type, got nil")
	}
}
