// cross_search.go — POST /memories/cross-search (Phase 4: physical isolation).
//
// Strict-mode cross-collection search. The request body MUST include a
// non-empty `collections` array — defaulting to "all collections" was
// rejected at Day1 review (BMO, 2026-05-05) because it makes recall
// surface implicit and hard to reason about (e.g. reflection insights
// silently leaking into user recall).
//
// Each collection in the list must be registered. Unknown name → 400.
// The handler searches each physical Qdrant collection independently,
// merges results, re-sorts by score, and applies the limit.
// Dedup by ID is no longer needed since physical isolation guarantees
// the same ID cannot exist in multiple collections.
package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
)

// crossSearchRequest is the JSON body for POST /memories/cross-search.
type crossSearchRequest struct {
	Query           string   `json:"query"`
	Collections     []string `json:"collections"`
	Limit           int      `json:"limit"`
	IncludeArchived bool     `json:"include_archived"`
	Types           []string `json:"types"`
	Tags            []string `json:"tags"`
	SourceType      []string `json:"source_type,omitempty"`
}

type crossSearchHit struct {
	memory.Memory
	Score      float64 `json:"score"`
	Collection string  `json:"collection"`
	SourceType string  `json:"source_type,omitempty"`
}

func (h *HTTPServer) handleCrossSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	start := time.Now()
	if h.srv.metrics != nil {
		defer func() { h.srv.metrics.SearchDuration.Observe(time.Since(start).Seconds()) }()
	}

	var req crossSearchRequest
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body too large"})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if req.Query == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query is required"})
		return
	}
	// STRICT MODE: collections list must be present and non-empty.
	if len(req.Collections) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "collections is required and must be non-empty (strict mode — implicit all-collection search is not supported)",
		})
		return
	}
	// All listed collections must be registered.
	for _, name := range req.Collections {
		if _, ok := collection.DefaultRegistry.Get(name); !ok {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error":      "unknown collection: " + name,
				"collection": name,
			})
			return
		}
	}

	limit := req.Limit
	if limit <= 0 {
		limit = 10
	}
	if limit > 100 {
		limit = 100
	}

	var filters []memory.Filter
	if len(req.Types) > 0 {
		filters = append(filters, memory.Filter{Field: "type", Op: memory.OpIn, Value: req.Types})
	}
	if len(req.Tags) > 0 {
		filters = append(filters, memory.Filter{Field: "tags", Op: memory.OpIn, Value: req.Tags})
	}
	if len(req.SourceType) > 0 {
		for _, v := range req.SourceType {
			if err := memory.ValidateSourceType(v); err != nil {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
				return
			}
		}
		filters = append(filters, memory.Filter{Field: "metadata.source_type", Op: memory.OpIn, Value: req.SourceType})
	}

	embedStart := time.Now()
	vec, err := h.srv.embedder.Embed(r.Context(), req.Query)
	if h.srv.metrics != nil {
		h.srv.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embed error: %v", err)})
		return
	}

	// Per-collection fetchLimit. With single-collection Store today this
	// is mostly bookkeeping; once Store is per-collection, the math
	// stays the same.
	perColLimit := limit * 3
	if perColLimit < 10 {
		perColLimit = 10
	}

	// Phase 4: each collection is a separate physical Qdrant collection, so the
	// same ID cannot appear across collections. Dedup is no longer needed.
	all := make([]crossSearchHit, 0, perColLimit*len(req.Collections))
	for _, colName := range req.Collections {
		colFilter := append(append([]memory.Filter(nil), filters...), memory.Filter{Field: "collection", Op: memory.OpIn, Value: []string{colName}})
		results, err := h.srv.store.Search(r.Context(), vec, memory.SearchOptions{
			Limit:           perColLimit,
			Filters:         colFilter,
			ExcludeArchived: !req.IncludeArchived,
		})
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{
				"error":      fmt.Sprintf("search error in %s: %v", colName, err),
				"collection": colName,
			})
			return
		}
		for i := range results {
			results[i].Score = memory.Score(&results[i].Memory, results[i].Score, h.srv.weights, h.srv.decay)
			all = append(all, crossSearchHit{
				Memory:     results[i].Memory,
				Score:      results[i].Score,
				Collection: colName,
				SourceType: sourceTypeFromMetadata(results[i].Metadata),
			})
		}
	}

	sort.Slice(all, func(i, j int) bool { return all[i].Score > all[j].Score })
	if len(all) > limit {
		all = all[:limit]
	}

	// Async access bookkeeping — same pattern as handleSearchMemories.
	callerType := r.Header.Get("X-Caller-Type")
	items := make([]accessUpdate, len(all))
	for i, hit := range all {
		items[i] = accessUpdate{ID: hit.ID, AccessCount: hit.AccessCount}
	}
	asyncUpdateAccessCounts(h.srv.store, nil, items, callerType, true)

	writeJSON(w, http.StatusOK, all)
}
