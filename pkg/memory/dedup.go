package memory

// DefaultDedupThreshold is the cosine similarity above which a new memory
// is considered a duplicate of an existing one and will be skipped.
const DefaultDedupThreshold = 0.92

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
