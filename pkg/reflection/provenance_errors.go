package reflection

import "errors"

// Provenance enforcement errors for reflection write-back (Phase 2 §4.3).
var (
	// ErrMissingProvenance is returned when a write-back memory lacks a
	// source_type value while the provenance filter is in "block" mode.
	ErrMissingProvenance = errors.New("reflection: write-back missing provenance (source_type) in block mode")

	// ErrInvalidProvenance is returned when a write-back memory carries a
	// source_type that is not in the configured AllowedProvenances whitelist
	// while the provenance filter is in "block" mode.
	ErrInvalidProvenance = errors.New("reflection: write-back source_type not in allowed provenances")
)
