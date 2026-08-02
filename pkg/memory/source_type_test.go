package memory

import (
	"strings"
	"sync"
	"testing"
)

func TestIsValidSourceType(t *testing.T) {
	valid := []string{"tool_output", "reflection", "web_search", "user_input", "calendar", "document", "unknown"}
	for _, s := range valid {
		if !IsValidSourceType(s) {
			t.Errorf("IsValidSourceType(%q) = false, want true", s)
		}
	}

	invalid := []string{"", "tool", "TOOL_OUTPUT", "web", "user"}
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
	if len(ValidSourceTypes) != 7 {
		t.Errorf("expected 7 valid source types, got %d", len(ValidSourceTypes))
	}
}

// R2: SourceTypeUnknown is the 7th enum value and is valid.
func TestSourceTypeUnknownValid(t *testing.T) {
	if !IsValidSourceType(string(SourceTypeUnknown)) {
		t.Errorf("IsValidSourceType(%q) = false, want true", SourceTypeUnknown)
	}
	if SourceTypeUnknown != "unknown" {
		t.Errorf("SourceTypeUnknown = %q, want unknown", SourceTypeUnknown)
	}
}

// R2: legacy memories without source_type now default to "unknown".
func TestDefaultSourceTypeIsUnknown(t *testing.T) {
	if DefaultSourceType != SourceTypeUnknown {
		t.Errorf("DefaultSourceType = %q, want %q", DefaultSourceType, SourceTypeUnknown)
	}
}

// R4e: invalid values must yield a clear, actionable error containing the
// offending value and the list of valid types.
func TestValidateSourceTypeErrorMessage(t *testing.T) {
	err := ValidateSourceType("bogus")
	if err == nil {
		t.Fatal("expected error for invalid source_type, got nil")
	}
	msg := err.Error()
	if !strings.Contains(msg, `"bogus"`) {
		t.Errorf("error should quote the offending value, got: %q", msg)
	}
	for _, valid := range []string{"tool_output", "reflection", "web_search", "user_input", "calendar", "document", "unknown"} {
		if !strings.Contains(msg, valid) {
			t.Errorf("error should list valid type %q, got: %q", valid, msg)
		}
	}
}

// R4a: empty string is invalid — callers treat it as "not provided" before
// calling ValidateSourceType, so it must never validate as a real type. This is
// the one assertion not already covered by TestIsValidSourceType /
// TestValidateSourceType (both list "" among invalids but don't pair the two).
func TestValidateSourceTypeEmpty(t *testing.T) {
	if IsValidSourceType("") != (ValidateSourceType("") == nil) {
		t.Fatalf("IsValidSourceType and ValidateSourceType disagree on empty string")
	}
	if ValidateSourceType("") == nil {
		t.Error("ValidateSourceType(\"\") = nil, want error")
	}
}

// R6a: ValidateSourceType on the happy path must not allocate (map lookup +
// string->SourceType conversion only; fmt.Errorf runs only on the error path).
// This asserts the contract; BenchmarkValidateSourceType reports the ns/op.
func TestValidateSourceTypeZeroAlloc(t *testing.T) {
	if n := testing.AllocsPerRun(100, func() { _ = ValidateSourceType("reflection") }); n != 0 {
		t.Errorf("ValidateSourceType happy path allocates %v/op, want 0", n)
	}
}

// R6a: ValidateSourceType on the happy path must not allocate (map lookup +
// string->SourceType conversion only; fmt.Errorf runs only on the error path).
func BenchmarkValidateSourceType(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = ValidateSourceType("reflection")
	}
}

// R4c: ValidSourceTypes is a package-level map read by validation. The len()
// check below only catches size-changing writes (added/removed keys); it does
// NOT catch value replacement or concurrent access hazards. Running this test
// under -race is what actually guards against a future write racing the reads.
func TestValidSourceTypesConcurrentReadSafe(t *testing.T) {
	before := len(ValidSourceTypes)
	inputs := []string{"tool_output", "reflection", "", "bogus", "REFLECTION", "document"}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, s := range inputs {
				_ = IsValidSourceType(s)
				_ = ValidateSourceType(s)
			}
		}()
	}
	wg.Wait()
	if after := len(ValidSourceTypes); after != before {
		t.Errorf("ValidSourceTypes mutated by validation: len %d -> %d", before, after)
	}
}
