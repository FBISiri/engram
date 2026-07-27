package memory

import "fmt"

// SourceType is a fine-grained provenance classifier describing what kind of
// input produced a memory. It is stored in Memory.Metadata["source_type"]
// (NOT as a top-level struct field) and complements the coarse Memory.Source
// field ("user", "agent", "system"). C1 feature (D4 productization).
type SourceType string

const (
	// SourceTypeToolOutput is a memory derived from a tool/function call result.
	SourceTypeToolOutput SourceType = "tool_output"
	// SourceTypeReflection is a memory synthesized by the Reflection Engine.
	SourceTypeReflection SourceType = "reflection"
	// SourceTypeWebSearch is a memory derived from a web search result.
	SourceTypeWebSearch SourceType = "web_search"
	// SourceTypeUserInput is a memory derived directly from user input.
	SourceTypeUserInput SourceType = "user_input"
	// SourceTypeCalendar is a memory derived from calendar data.
	SourceTypeCalendar SourceType = "calendar"
	// SourceTypeDocument is a memory derived from a document.
	SourceTypeDocument SourceType = "document"
)

// ValidSourceTypes is the set of all valid source types.
var ValidSourceTypes = map[SourceType]bool{
	SourceTypeToolOutput: true,
	SourceTypeReflection: true,
	SourceTypeWebSearch:  true,
	SourceTypeUserInput:  true,
	SourceTypeCalendar:   true,
	SourceTypeDocument:   true,
}

// IsValidSourceType reports whether s is a recognized source type.
func IsValidSourceType(s string) bool {
	return ValidSourceTypes[SourceType(s)]
}

// ValidateSourceType returns an error if s is not a recognized source type.
// An empty string is treated as invalid; callers that soft-require source_type
// should check for emptiness before calling this.
func ValidateSourceType(s string) error {
	if !IsValidSourceType(s) {
		return fmt.Errorf("invalid source_type: %q (valid: tool_output, reflection, web_search, user_input, calendar, document)", s)
	}
	return nil
}
