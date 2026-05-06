// cross_search.go — POST /memories/cross-search (W20 Day2 Phase 2).
//
// Strict-mode cross-collection search. The request body MUST include a
// non-empty `collections` array — defaulting to "all collections" was
// rejected at Day1 review (BMO, 2026-05-05) because it makes recall
// surface implicit and hard to reason about (e.g. reflection insights
// silently leaking into user recall).
//
// Each collection in the list must be registered. Unknown name → 400.
// The handler iterates the collections, runs the existing search per
// collection, merges results, re-sorts by score, and applies the limit.
//
// Phase 2 caveat: underlying Store is still single Qdrant collection,
// so all listed collections currently return the same point set. This
// is intentional — the API surface is what BMO and the reflection
// engine need to start migrating against. Phase 4 will plumb collection
// name into the Store interface and provide real physical isolation.
package server

import (
	"context"
	"encoding/json"
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
}

type crossSearchHit struct {
	memory.Memory
	Score      float64 `json:"score"`
	Collection string  `json:"collection"`
}

func (h *HTTPServer) handleCrossSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}

	var req crossSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
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

	vec, err := h.srv.embedder.Embed(r.Context(), req.Query)
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

	all := make([]crossSearchHit, 0, perColLimit*len(req.Collections))
	seen := make(map[string]struct{})
	for _, colName := range req.Collections {
		results, err := h.srv.store.Search(r.Context(), vec, memory.SearchOptions{
			Limit:           perColLimit,
			Filters:         filters,
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
			if _, dup := seen[results[i].ID]; dup {
				// Phase 2 dedup: same ID across "collections" today is the
				// same row. Phase 4 will give physical isolation and dedup
				// will become a no-op.
				continue
			}
			seen[results[i].ID] = struct{}{}
			all = append(all, crossSearchHit{
				Memory:     results[i].Memory,
				Score:      results[i].Score,
				Collection: colName,
			})
		}
	}

	sort.Slice(all, func(i, j int) bool { return all[i].Score > all[j].Score })
	if len(all) > limit {
		all = all[:limit]
	}

	// Async access bookkeeping — same pattern as handleSearchMemories.
	callerType := r.Header.Get("X-Caller-Type")
	if len(all) > 0 {
		ids := make([]string, len(all))
		counts := make([]int64, len(all))
		for i, hit := range all {
			ids[i] = hit.ID
			counts[i] = hit.AccessCount
		}
		go func() {
			now := float64(time.Now().Unix())
			updates := map[string]any{
				"last_accessed_at": now,
				"updated_at":       now,
			}
			if callerType != "" {
				updates["last_accessed_source"] = callerType
			}
			for i, id := range ids {
				u := make(map[string]any, len(updates)+1)
				for k, v := range updates {
					u[k] = v
				}
				u["access_count"] = counts[i] + 1
				_ = h.srv.store.Update(context.Background(), id, u)
			}
		}()
	}

	writeJSON(w, http.StatusOK, all)
}
