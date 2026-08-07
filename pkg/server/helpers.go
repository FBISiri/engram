package server

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ─────────────────────────────────────────────────────────────
// Shared search reranking (F8)
// ─────────────────────────────────────────────────────────────

// rerankResults applies the 3-component scoring to raw store results, sorts by
// final score descending, then reranks with MMR for relevance + diversity. When
// vectors are missing it falls back to simple truncation. The slice is mutated
// in place and trimmed to at most limit. Shared by the MCP and REST search paths.
func rerankResults(results []memory.ScoredMemory, weights memory.ScoringWeights, decay memory.DecayConfig, mmrLambda float64, limit int) []memory.ScoredMemory {
	for i := range results {
		results[i].Score = memory.Score(&results[i].Memory, results[i].Score, weights, decay)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })

	vectors := make([][]float32, len(results))
	hasVectors := false
	for i, r := range results {
		if len(r.Vector) > 0 {
			vectors[i] = r.Vector
			hasVectors = true
		}
	}
	if hasVectors && len(results) > limit {
		return memory.MMR(results, vectors, limit, mmrLambda)
	}
	if len(results) > limit {
		return results[:limit]
	}
	return results
}

// ─────────────────────────────────────────────────────────────
// Shared async access bookkeeping (F3)
// ─────────────────────────────────────────────────────────────

// accessUpdate carries the per-result data needed to bump access counters.
type accessUpdate struct {
	ID          string
	AccessCount int64
	Collection  string // used only when a targetedUpdater is supplied
}

// asyncAccessCountCap bounds concurrent access-count updates so a large result
// set does not stampede the store. Above this size the goroutine caps in-flight
// updates with a semaphore; at or below it, updates run sequentially.
const asyncAccessCountCap = 50

// asyncUpdateAccessCounts bumps access_count and last_accessed_at (plus
// last_accessed_source when callerType is non-empty and updated_at when
// setUpdatedAt is true) for the given results in a background goroutine. Unlike
// the previous inline copies, update errors are logged rather than silently
// discarded. When tu is non-nil, per-collection targeted updates are used;
// otherwise the plain store.Update path is taken.
func asyncUpdateAccessCounts(store memory.Store, tu targetedUpdater, items []accessUpdate, callerType string, setUpdatedAt bool) {
	if len(items) == 0 {
		return
	}
	go func() {
		now := float64(time.Now().Unix())
		base := map[string]any{"last_accessed_at": now}
		if setUpdatedAt {
			base["updated_at"] = now
		}
		if callerType != "" {
			base["last_accessed_source"] = callerType
		}

		apply := func(it accessUpdate) {
			fields := make(map[string]any, len(base)+1)
			for k, v := range base {
				fields[k] = v
			}
			fields["access_count"] = it.AccessCount + 1
			var err error
			if tu != nil {
				err = tu.UpdateInCollection(context.Background(), it.ID, fields, it.Collection)
			} else {
				err = store.Update(context.Background(), it.ID, fields)
			}
			if err != nil {
				log.Printf("[WARN] engram: async access-count update failed for %s: %v", it.ID, err)
			}
		}

		// Small result sets: sequential (also the ultimate concurrency cap).
		if len(items) <= asyncAccessCountCap {
			for _, it := range items {
				apply(it)
			}
			return
		}
		// Large result sets: bounded concurrency to avoid a thundering herd.
		sem := make(chan struct{}, asyncAccessCountCap)
		var wg sync.WaitGroup
		for _, it := range items {
			it := it
			wg.Add(1)
			sem <- struct{}{}
			go func() {
				defer wg.Done()
				defer func() { <-sem }()
				apply(it)
			}()
		}
		wg.Wait()
	}()
}

// ─────────────────────────────────────────────────────────────
// Shared importance clamping (F8)
// ─────────────────────────────────────────────────────────────

// clampImportance constrains an importance value to the valid [0, 10] range.
func clampImportance(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 10 {
		return 10
	}
	return v
}

// ─────────────────────────────────────────────────────────────
// Shared provenance validation + assignment (F7)
// ─────────────────────────────────────────────────────────────

// Sentinel provenance errors. Callers use errors.Is to map them to the correct
// HTTP status (REST) or tool error (MCP).
var (
	errProvenanceInvalid  = errors.New("invalid source_type")       // present but not a valid enum value → 400
	errProvenanceRejected = errors.New("source_type not permitted") // strict mode forbids this explicit value → 422
	errProvenanceRequired = errors.New("source_type required")      // strict mode and source_type omitted → 422
)

// validateProvenance checks an explicit source_type against the enum and the
// strict-mode allow-list. Returns errProvenanceInvalid if not a valid enum
// value, errProvenanceRejected if strict mode forbids it, else nil.
func (s *Server) validateProvenance(sourceType string) error {
	if err := memory.ValidateSourceType(sourceType); err != nil {
		return fmt.Errorf("%w: %v", errProvenanceInvalid, err)
	}
	if s.strictProvenanceRejects(sourceType) {
		return errProvenanceRejected
	}
	return nil
}

// applyProvenance validates and stamps source_type into metadata following the
// shared C1 provenance rules used by every write path. `provided` indicates
// whether the caller supplied a source_type. On the omitted path it either
// rejects (strict mode) or stamps the default. metadata must be non-nil; on
// error metadata is left unmodified for the rejection cases.
func (s *Server) applyProvenance(metadata map[string]any, sourceType string, provided bool, op string) error {
	if provided {
		if err := s.validateProvenance(sourceType); err != nil {
			if errors.Is(err, errProvenanceRejected) {
				log.Printf("[WARN] engram %s: source_type %q not permitted, rejecting (strict mode)", op, sourceType)
			}
			return err
		}
		metadata["source_type"] = sourceType
		return nil
	}
	if s.cfg != nil && s.cfg.ProvenanceMode == "strict" {
		log.Printf("[WARN] engram %s: source_type not provided, rejecting (strict mode)", op)
		return errProvenanceRequired
	}
	metadata["source_type"] = string(memory.DefaultSourceType)
	log.Printf("[WARN] engram %s: source_type not provided, defaulting to 'unknown'", op)
	return nil
}

// extractMetaSourceType pulls source_type out of a metadata map. Returns the
// string value and whether the key was present. A present-but-non-string value
// yields an error (caller → 400).
func extractMetaSourceType(metadata map[string]any) (string, bool, error) {
	raw, ok := metadata["source_type"]
	if !ok {
		return "", false, nil
	}
	s, isStr := raw.(string)
	if !isStr {
		return "", true, fmt.Errorf("invalid source_type: %v", raw)
	}
	return s, true, nil
}

// provenanceStatus maps a sentinel provenance error to an HTTP status + message
// for the REST handlers.
func provenanceStatus(err error) (int, string) {
	switch {
	case errors.Is(err, errProvenanceInvalid):
		return http.StatusBadRequest, err.Error()
	case errors.Is(err, errProvenanceRejected):
		return http.StatusUnprocessableEntity, strictProvenanceRejectMsg
	case errors.Is(err, errProvenanceRequired):
		return http.StatusUnprocessableEntity, strictProvenanceMsg
	default:
		return http.StatusBadRequest, err.Error()
	}
}

// provenanceMCPMsg maps a sentinel provenance error to an MCP tool-error message.
func provenanceMCPMsg(err error) string {
	switch {
	case errors.Is(err, errProvenanceRejected):
		return strictProvenanceRejectMsg
	case errors.Is(err, errProvenanceRequired):
		return strictProvenanceMsg
	default:
		return err.Error()
	}
}
