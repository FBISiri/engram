package memory

// DefaultDedupThreshold is the cosine similarity above which a new memory
// is considered a duplicate of an existing one and will be skipped.
const DefaultDedupThreshold = 0.92

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
