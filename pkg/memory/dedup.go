package memory

// DefaultDedupThreshold is the cosine similarity above which a new memory
// is considered a duplicate of an existing one and will be skipped.
const DefaultDedupThreshold = 0.92

// TypeDedupThresholds holds the A-MAC per-type dedup thresholds (A-MAC MVP).
// Higher threshold = fewer merges = more memories coexist. identity stays
// protective at 0.95 so similar-but-distinct identity facets coexist; directive
// is aggressive at 0.90 to merge near-duplicate directives and prevent the
// sycophancy flywheel; insight and event use the standard 0.92 to balance
// coexistence against redundancy (0.92 matches the global default).
var TypeDedupThresholds = map[MemoryType]float64{
	TypeIdentity:  0.95, // protective: only near-identical merge
	TypeDirective: 0.90, // aggressive: prevent sycophancy flywheel
	TypeInsight:   0.92, // standard: balance coexistence vs redundancy
	TypeEvent:     0.92, // standard: align with insight
}

// DedupThresholdForType returns the A-MAC per-type dedup threshold, falling back
// to the global DefaultDedupThreshold for unrecognized types.
func DedupThresholdForType(t MemoryType) float64 {
	if v, ok := TypeDedupThresholds[t]; ok {
		return v
	}
	return DefaultDedupThreshold
}

// IsDuplicateForType is IsDuplicate using the A-MAC per-type threshold for t.
func IsDuplicateForType(candidates []ScoredMemory, t MemoryType) *ScoredMemory {
	return IsDuplicate(candidates, DedupThresholdForType(t))
}

// ProvenanceEntry records a single provenance merge event: when a content
// duplicate arrives with a different source_type, the additional source is
// appended to the existing memory's provenance_history.
type ProvenanceEntry struct {
	SourceType   string  `json:"source_type"`
	MergedAt     int64   `json:"merged_at"`
	ContentScore float64 `json:"content_score"`
}

// sourceTypeTrust returns the trust rank of a source_type (lower = more
// trusted). Follows the MemIR evidence hierarchy: external/human-authoritative
// sources outrank agent-synthesized ones.
var sourceTypeTrust = map[string]int{
	"user_input":  1,
	"tool_output": 2,
	"web_search":  3,
	"document":    4,
	"calendar":    5,
	"reflection":  6,
	"unknown":     7,
}

// HighestTrustSource returns the source_type with the highest trust (lowest
// rank) from a list. Unrecognized values are ignored; if none are recognized,
// it returns "unknown".
func HighestTrustSource(sources []string) string {
	best := "unknown"
	bestRank := 999
	for _, s := range sources {
		if rank, ok := sourceTypeTrust[s]; ok && rank < bestRank {
			best = s
			bestRank = rank
		}
	}
	return best
}

// HasSourceType reports whether a source_type already exists in a provenance
// history (used for idempotent merges).
func HasSourceType(history []ProvenanceEntry, st string) bool {
	for _, h := range history {
		if h.SourceType == st {
			return true
		}
	}
	return false
}

// MaxProvenanceHistory caps the number of provenance merge entries stored on a
// single memory (safety bound against unbounded growth).
const MaxProvenanceHistory = 10

// IsDuplicate checks if any of the scored candidates are above the dedup threshold.
// Returns the best matching memory if duplicate, nil otherwise.
func IsDuplicate(candidates []ScoredMemory, threshold float64) *ScoredMemory {
	if len(candidates) == 0 {
		return nil
	}
	// Candidates are expected to be sorted by raw cosine similarity (descending).
	best := &candidates[0]
	if best.Score >= threshold {
		return best
	}
	return nil
}
