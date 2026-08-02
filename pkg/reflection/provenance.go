package reflection

import (
	"fmt"
	"log"

	"github.com/FBISiri/engram/pkg/memory"
)

// ProvenanceFilterMode selects how the provenance (source_type) filter behaves
// during evidence retrieval and write-back (Phase 2 §4.1).
type ProvenanceFilterMode string

const (
	// ProvenanceModeWarn logs a warning but does NOT filter evidence.
	ProvenanceModeWarn ProvenanceFilterMode = "warn"
	// ProvenanceModeDefault filters evidence with an OpIn whitelist on
	// metadata.source_type (backward-compatible with the legacy behavior).
	ProvenanceModeDefault ProvenanceFilterMode = "default"
	// ProvenanceModeBlock filters with OpIn AND additionally emits an IsNull
	// filter so that memories lacking a source_type are explicitly excluded.
	ProvenanceModeBlock ProvenanceFilterMode = "block"
)

const provenanceSourceField = "metadata.source_type"

// ProvenanceFilterConfig configures provenance-based evidence filtering and
// write-back enforcement (Phase 2 §4.1).
type ProvenanceFilterConfig struct {
	Enabled            bool                 `json:"enabled"`
	Mode               ProvenanceFilterMode `json:"mode"`
	AllowedProvenances []string             `json:"allowed_provenances"`
}

// BuildEvidenceFilters translates a ProvenanceFilterConfig into store filters
// for evidence retrieval (Phase 2 §4.2).
//
//   - disabled / empty whitelist → no filters
//   - warn    → no filters (logs a warning)
//   - default → OpIn on metadata.source_type
//   - block   → OpIn on metadata.source_type + OpIsNull exclusion
func BuildEvidenceFilters(cfg ProvenanceFilterConfig) []memory.Filter {
	if !cfg.Enabled || len(cfg.AllowedProvenances) == 0 {
		return nil
	}

	switch cfg.Mode {
	case ProvenanceModeWarn:
		log.Printf("[reflection] provenance filter in WARN mode: evidence NOT filtered (allowed=%v)", cfg.AllowedProvenances)
		return nil
	case ProvenanceModeBlock:
		return []memory.Filter{
			{Field: provenanceSourceField, Op: memory.OpIn, Value: cfg.AllowedProvenances},
			{Field: provenanceSourceField, Op: memory.OpIsNull, Value: nil},
		}
	default: // ProvenanceModeDefault (and unknown modes fall back to default)
		return []memory.Filter{
			{Field: provenanceSourceField, Op: memory.OpIn, Value: cfg.AllowedProvenances},
		}
	}
}

// EnforceWriteBackProvenance validates/normalizes the source_type provenance on
// a write-back memory's metadata according to the provenance filter mode
// (Phase 2 §4.3). It may mutate metadata (default mode auto-sets source_type).
//
//   - disabled → no-op
//   - block    → reject missing source_type (ErrMissingProvenance) and reject
//     source_type outside the whitelist (ErrInvalidProvenance)
//   - warn     → log missing source_type, proceed
//   - default  → auto-set missing source_type to "reflection"
func EnforceWriteBackProvenance(cfg ProvenanceFilterConfig, metadata map[string]any) error {
	if !cfg.Enabled {
		return nil
	}

	st, _ := metadata["source_type"].(string)
	hasSourceType := st != ""

	// Validate an explicit value against the whitelist (block mode only).
	if hasSourceType && cfg.Mode == ProvenanceModeBlock && len(cfg.AllowedProvenances) > 0 {
		if !containsProvenance(cfg.AllowedProvenances, st) {
			return fmt.Errorf("%w: %q not in %v", ErrInvalidProvenance, st, cfg.AllowedProvenances)
		}
	}

	if hasSourceType {
		return nil
	}

	switch cfg.Mode {
	case ProvenanceModeBlock:
		return ErrMissingProvenance
	case ProvenanceModeWarn:
		log.Printf("[reflection] write-back missing source_type (warn mode); proceeding")
		return nil
	default: // default mode: auto-label as reflection provenance
		metadata["source_type"] = "reflection"
		return nil
	}
}

func containsProvenance(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}
