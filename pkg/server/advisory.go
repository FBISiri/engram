// advisory.go — Write Discipline Checkpoints (spec v1): non-blocking write
// feedback signals attached to memory_add/memory_update responses.
//
// Unlike A-MAC hard limits (which reject/skip writes), advisories let the write
// succeed while signaling quality concerns to the calling agent. All checkpoint
// behavior is gated behind ENGRAM_WRITE_CHECKPOINTS_ENABLED (default false); when
// disabled there is zero overhead and the response shape is unchanged.
package server

import (
	"fmt"
	"log"

	"github.com/FBISiri/engram/pkg/memory"
)

// Advisory is a non-blocking write feedback signal.
type Advisory struct {
	Type     string         `json:"type"`           // machine-readable: "similar_memory", "importance_inflation", "rate_limit_warning", "content_too_short", "content_too_long", "tags_missing", "source_type_missing"
	Severity string         `json:"severity"`       // "info" | "warning"
	Message  string         `json:"message"`        // human-readable explanation
	Data     map[string]any `json:"data,omitempty"` // structured context for programmatic consumption
}

// logAdvisory emits the R12 audit log line for a generated advisory.
func logAdvisory(a Advisory) {
	log.Printf("[INFO] engram write_checkpoint: %s — %s", a.Type, a.Message)
}

// writeCheckpointsEnabled reports whether the write-discipline checkpoints are
// active (master switch). When false, no checkpoint code runs.
func (s *Server) writeCheckpointsEnabled() bool {
	return s.cfg != nil && s.cfg.WriteCheckpointsEnabled
}

// cp1DedupAdvisory (CP1) emits a `similar_memory` advisory when a dedup-search
// candidate falls in the near-duplicate band [min_score, type_threshold). It
// reuses the candidates already returned by checkDedup — zero extra embedding or
// search cost.
func (s *Server) cp1DedupAdvisory(candidates []memory.ScoredMemory, memType memory.MemoryType) *Advisory {
	if !s.cfg.CPDedupAdvisoryEnabled {
		return nil
	}
	threshold := s.resolveDedupThreshold(memType)
	minScore := s.cfg.CPDedupAdvisoryMinScore
	var sims []map[string]any
	topScore := 0.0
	for _, c := range candidates {
		if c.Score >= minScore && c.Score < threshold {
			content := c.Content
			if r := []rune(content); len(r) > 200 {
				content = string(r[:200])
			}
			sims = append(sims, map[string]any{"id": c.ID, "content": content, "score": c.Score})
			if c.Score > topScore {
				topScore = c.Score
			}
		}
	}
	if len(sims) == 0 {
		return nil
	}
	return &Advisory{
		Type:     "similar_memory",
		Severity: "info",
		Message:  fmt.Sprintf("Similar memory exists (score %.2f). Consider using memory_update instead of memory_add.", topScore),
		Data:     map[string]any{"similar_memories": sims},
	}
}

// cp2Importance (CP2) records the importance into the per-collection monitor and
// returns an `importance_inflation` advisory when the rolling mean exceeds the
// threshold.
func (s *Server) cp2Importance(collection string, importance float64) *Advisory {
	if s.importanceMonitor == nil || !s.cfg.CPImportanceMonitorEnabled {
		return nil
	}
	return s.importanceMonitor.Record(collection, importance)
}

// cp3RateLimitAdvisory (CP3) emits a `rate_limit_warning` when utilization for
// (collection, memType) reaches the configured fraction. Requires A-MAC enabled
// (the RateLimiter is only populated then); otherwise it silently skips.
func (s *Server) cp3RateLimitAdvisory(collection string, memType memory.MemoryType) *Advisory {
	if !s.cfg.CPRateLimitWarningEnabled || !s.amacEnabled() {
		return nil
	}
	count, limit, frac := s.rateLimiter.Utilization(collection, memType)
	if frac < s.cfg.CPRateLimitWarningFraction {
		return nil
	}
	return &Advisory{
		Type:     "rate_limit_warning",
		Severity: "warning",
		Message:  fmt.Sprintf("Approaching rate limit: %d/%d writes for type %s this hour (%.0f%%).", count, limit, memType, frac*100),
		Data: map[string]any{
			"current_count": count,
			"limit":         limit,
			"utilization":   frac,
			"type":          string(memType),
		},
	}
}

// cp4Content (CP4) runs the lightweight content-quality checks. The source_type
// check does not overlap provenance strict mode: strict mode rejects a missing
// source_type before this success-path check is ever reached.
func (s *Server) cp4Content(content string, tags []string, sourceType string) []Advisory {
	if s.contentChecker == nil || !s.cfg.CPContentCheckEnabled {
		return nil
	}
	return s.contentChecker.Check(content, tags, sourceType)
}
