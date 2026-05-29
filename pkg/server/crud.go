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

// ─────────────────────────────────────────────────────────────
// POST /memories — create a new memory
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleCreateMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}

	var body struct {
		Type       string         `json:"type"`
		Content    string         `json:"content"`
		Source     string         `json:"source"`
		Importance float64        `json:"importance"`
		Tags       []string       `json:"tags"`
		ValidUntil float64        `json:"valid_until"`
		Metadata   map[string]any `json:"metadata"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if body.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content is required"})
		return
	}

	memType := memory.MemoryType(body.Type)
	if body.Type == "" {
		memType = memory.TypeEvent
	} else if !memory.ValidTypes[memType] {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid type: %s", body.Type)})
		return
	}

	importance := body.Importance
	if importance <= 0 {
		importance = 5.0
	}
	if importance < 1 {
		importance = 1
	}
	if importance > 10 {
		importance = 10
	}

	source := body.Source
	if source == "" {
		source = "agent"
	}

	tags := body.Tags
	if tags == nil {
		tags = []string{}
	}

	ttlCfg := memory.DefaultTTLConfig()
	computedValidUntil := memory.ComputeValidUntil(ttlCfg, memType, importance, tags, body.ValidUntil)

	opts := []memory.Option{
		memory.WithType(memType),
		memory.WithImportance(importance),
		memory.WithSource(source),
		memory.WithTags(tags...),
	}
	if computedValidUntil > 0 {
		opts = append(opts, memory.WithValidUntil(computedValidUntil))
	}
	mem := memory.New(body.Content, opts...)
	if body.Metadata != nil {
		mem.Metadata = body.Metadata
	}

	embedStart := time.Now()
	vec, err := h.srv.embedder.Embed(r.Context(), body.Content)
	if h.srv.metrics != nil {
		h.srv.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embed error: %v", err)})
		return
	}

	if err := h.srv.store.Insert(r.Context(), mem, vec); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("insert error: %v", err)})
		return
	}

	writeJSON(w, http.StatusCreated, mem)
}

// ─────────────────────────────────────────────────────────────
// GET /memories/{id}
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleGetMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use GET"})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"})
		return
	}

	mems, err := h.srv.store.SearchByIDs(r.Context(), []string{id})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("store error: %v", err)})
		return
	}
	if len(mems) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	writeJSON(w, http.StatusOK, mems[0])
}

// ─────────────────────────────────────────────────────────────
// PATCH /memories/{id} — partial update (content forbidden)
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handlePatchMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPatch {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use PATCH"})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"})
		return
	}

	// Decode into a raw map to detect forbidden fields.
	var raw map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if _, hasContent := raw["content"]; hasContent {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content cannot be patched — use PUT to replace content"})
		return
	}

	// Fetch current memory for FSM validation.
	mems, err := h.srv.store.SearchByIDs(r.Context(), []string{id})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("store error: %v", err)})
		return
	}
	if len(mems) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	current := mems[0]

	updates := map[string]any{
		"updated_at": float64(time.Now().Unix()),
	}

	// lifecycle_status FSM validation.
	if rawStatus, ok := raw["lifecycle_status"]; ok {
		var nextStatus string
		if err := json.Unmarshal(rawStatus, &nextStatus); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid lifecycle_status"})
			return
		}
		curStatus := current.LifecycleStatus
		if curStatus == "" {
			curStatus = memory.LifecycleActive
		}
		if !isValidLifecycleTransition(curStatus, nextStatus) {
			writeJSON(w, http.StatusConflict, map[string]string{
				"error": fmt.Sprintf("lifecycle transition %s→%s is not allowed", curStatus, nextStatus),
			})
			return
		}
		updates["lifecycle_status"] = nextStatus
		if nextStatus == memory.LifecycleArchived && curStatus != memory.LifecycleArchived {
			updates["archived_at"] = float64(time.Now().Unix())
		}
	}

	// Other patchable fields.
	if rawTags, ok := raw["tags"]; ok {
		var tags []string
		if err := json.Unmarshal(rawTags, &tags); err == nil {
			tagsAny := make([]any, len(tags))
			for i, t := range tags {
				tagsAny[i] = t
			}
			updates["tags"] = tagsAny
		}
	}
	if rawImp, ok := raw["importance"]; ok {
		var imp float64
		if err := json.Unmarshal(rawImp, &imp); err == nil {
			updates["importance"] = imp
		}
	}
	if rawSrc, ok := raw["source"]; ok {
		var src string
		if err := json.Unmarshal(rawSrc, &src); err == nil {
			updates["source"] = src
		}
	}
	if rawMeta, ok := raw["metadata"]; ok {
		var meta map[string]any
		if err := json.Unmarshal(rawMeta, &meta); err == nil {
			updates["metadata"] = meta
		}
	}

	if err := h.srv.store.Update(r.Context(), id, updates); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("update error: %v", err)})
		return
	}

	// Return updated memory.
	mems, _ = h.srv.store.SearchByIDs(r.Context(), []string{id})
	if len(mems) > 0 {
		writeJSON(w, http.StatusOK, mems[0])
	} else {
		writeJSON(w, http.StatusOK, map[string]string{"id": id})
	}
}

// ─────────────────────────────────────────────────────────────
// PUT /memories/{id} — full replacement (content allowed, re-embeds)
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handlePutMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use PUT"})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"})
		return
	}

	// Fetch existing to preserve lifecycle_status, CreatedAt, etc.
	existing, err := h.srv.store.SearchByIDs(r.Context(), []string{id})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("store error: %v", err)})
		return
	}
	if len(existing) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	prev := existing[0]

	var body struct {
		Type       string         `json:"type"`
		Content    string         `json:"content"`
		Source     string         `json:"source"`
		Importance float64        `json:"importance"`
		Tags       []string       `json:"tags"`
		ValidUntil float64        `json:"valid_until"`
		Metadata   map[string]any `json:"metadata"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if body.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content is required"})
		return
	}

	memType := memory.MemoryType(body.Type)
	if body.Type == "" {
		memType = prev.Type
	} else if !memory.ValidTypes[memType] {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid type: %s", body.Type)})
		return
	}

	importance := body.Importance
	if importance <= 0 {
		importance = prev.Importance
	}

	source := body.Source
	if source == "" {
		source = prev.Source
	}

	tags := body.Tags
	if tags == nil {
		tags = prev.Tags
	}

	now := float64(time.Now().Unix())
	mem := &memory.Memory{
		ID:                 id,
		Type:               memType,
		Content:            body.Content,
		Source:             source,
		Importance:         importance,
		Tags:               tags,
		CreatedAt:          prev.CreatedAt,
		UpdatedAt:          now,
		Metadata:           body.Metadata,
		ValidUntil:         body.ValidUntil,
		LifecycleStatus:    prev.LifecycleStatus,
		AccessCount:        prev.AccessCount,
		LastAccessedAt:     prev.LastAccessedAt,
		LastAccessedSource: prev.LastAccessedSource,
		ReflectedAt:        prev.ReflectedAt,
		Confidence:         prev.Confidence,
	}
	if mem.Metadata == nil {
		mem.Metadata = map[string]any{}
	}
	if mem.Tags == nil {
		mem.Tags = []string{}
	}
	if mem.LifecycleStatus == "" {
		mem.LifecycleStatus = memory.LifecycleActive
	}

	embedStart := time.Now()
	vec, err := h.srv.embedder.Embed(r.Context(), body.Content)
	if h.srv.metrics != nil {
		h.srv.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embed error: %v", err)})
		return
	}

	if err := h.srv.store.Insert(r.Context(), mem, vec); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("upsert error: %v", err)})
		return
	}

	writeJSON(w, http.StatusOK, mem)
}

// ─────────────────────────────────────────────────────────────
// DELETE /memories/{id} — soft delete (→ archived)
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleDeleteMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use DELETE"})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"})
		return
	}

	mems, err := h.srv.store.SearchByIDs(r.Context(), []string{id})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("store error: %v", err)})
		return
	}
	if len(mems) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}

	now := float64(time.Now().Unix())
	updates := map[string]any{
		"lifecycle_status": memory.LifecycleArchived,
		"archived_at":      now,
		"updated_at":       now,
	}
	// Guard: stamp reflected_at so this memory exits the unreflected pool.
	// Without this, archived memories with reflected_at=0 would keep appearing
	// in fetchUnreflected on every reflection run.
	if mems[0].ReflectedAt == 0 {
		updates["reflected_at"] = now
	}
	if err := h.srv.store.Update(r.Context(), id, updates); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("update error: %v", err)})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"id": id, "lifecycle_status": memory.LifecycleArchived})
}

// ─────────────────────────────────────────────────────────────
// POST /memories/{id}/reset — restore archived/deprecated → active
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleResetMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"})
		return
	}

	mems, err := h.srv.store.SearchByIDs(r.Context(), []string{id})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("store error: %v", err)})
		return
	}
	if len(mems) == 0 {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"})
		return
	}
	cur := mems[0]

	curStatus := cur.LifecycleStatus
	if curStatus == "" {
		curStatus = memory.LifecycleActive
	}
	if curStatus == memory.LifecycleActive {
		writeJSON(w, http.StatusConflict, map[string]string{
			"error": "memory is already active — reset is not allowed on active memories",
		})
		return
	}

	updates := map[string]any{
		"lifecycle_status": memory.LifecycleActive,
		"updated_at":       float64(time.Now().Unix()),
	}
	if err := h.srv.store.Update(r.Context(), id, updates); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("update error: %v", err)})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"id": id, "lifecycle_status": memory.LifecycleActive})
}

// ─────────────────────────────────────────────────────────────
// POST /memories/search — vector search with lifecycle filtering
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleSearchMemories(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	start := time.Now()
	if h.srv.metrics != nil {
		defer func() { h.srv.metrics.SearchDuration.Observe(time.Since(start).Seconds()) }()
	}

	var req struct {
		Query           string   `json:"query"`
		Collection      string   `json:"collection"` // W20 Day2 Phase 3: BMO Q3 — explicit field; falls back to ctx-resolution if empty.
		Limit           int      `json:"limit"`
		IncludeArchived bool     `json:"include_archived"`
		Types           []string `json:"types"`
		Tags            []string `json:"tags"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if req.Query == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query is required"})
		return
	}

	// W20 Day2 Phase 3 — legacy /memories/search compatibility layer.
	// BMO Q3 (2026-05-06, thread:19dfad7019babb78): explicit `collection`
	// field in body wins; missing → resolve from X-Caller-Type via ctx.
	// Unknown name → 400 (don't silently route to wrong namespace).
	// We do NOT 30x — old callers see the same response shape, just with
	// a `resolved_collection` annotation for observability. Phase 4 will
	// plumb this into the Store layer for physical isolation; for now the
	// underlying Qdrant collection is still single, so all routes return
	// the same point set. The annotation is here so callers (esp. the
	// reflection engine) can verify routing without log scraping.
	resolvedCollection := req.Collection
	if resolvedCollection == "" {
		resolvedCollection = CollectionFromContext(r.Context())
	} else {
		if _, ok := collection.DefaultRegistry.Get(resolvedCollection); !ok {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error":      "unknown collection: " + resolvedCollection,
				"collection": resolvedCollection,
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

	embedStart := time.Now()
	vec, err := h.srv.embedder.Embed(r.Context(), req.Query)
	if h.srv.metrics != nil {
		h.srv.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embed error: %v", err)})
		return
	}

	fetchLimit := limit * 3
	if fetchLimit < 10 {
		fetchLimit = 10
	}

	results, err := h.srv.store.Search(r.Context(), vec, memory.SearchOptions{
		Limit:           fetchLimit,
		Filters:         filters,
		ExcludeArchived: !req.IncludeArchived,
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("search error: %v", err)})
		return
	}

	// Apply scoring + MMR (same as MCP search).
	for i := range results {
		results[i].Score = memory.Score(&results[i].Memory, results[i].Score, h.srv.weights, h.srv.decay)
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
		results = memory.MMR(results, vectors, limit, h.srv.mmrLambda)
	} else if len(results) > limit {
		results = results[:limit]
	}

	// Update access_count and last_accessed_source asynchronously.
	callerType := r.Header.Get("X-Caller-Type")
	if len(results) > 0 {
		toUpdate := make([]string, len(results))
		accessCounts := make([]int64, len(results))
		for i, res := range results {
			toUpdate[i] = res.ID
			accessCounts[i] = res.AccessCount
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
			for i, id := range toUpdate {
				u := make(map[string]any, len(updates)+1)
				for k, v := range updates {
					u[k] = v
				}
				u["access_count"] = accessCounts[i] + 1
				_ = h.srv.store.Update(context.Background(), id, u)
			}
		}()
	}

	type result struct {
		memory.Memory
		Score              float64 `json:"score"`
		ResolvedCollection string  `json:"resolved_collection,omitempty"`
	}
	output := make([]result, len(results))
	for i, r := range results {
		output[i] = result{Memory: r.Memory, Score: r.Score, ResolvedCollection: resolvedCollection}
	}

	writeJSON(w, http.StatusOK, output)
}

// ─────────────────────────────────────────────────────────────
// FSM helpers
// ─────────────────────────────────────────────────────────────

// isValidLifecycleTransition returns true if transitioning from current to next
// is allowed via PATCH. The transition archived→* is always false here;
// use POST /{id}/reset to restore.
func isValidLifecycleTransition(current, next string) bool {
	switch current {
	case memory.LifecycleActive:
		return next == memory.LifecycleDeprecated || next == memory.LifecycleArchived || next == memory.LifecycleActive
	case memory.LifecycleDeprecated:
		return next == memory.LifecycleArchived || next == memory.LifecycleDeprecated
	case memory.LifecycleArchived:
		return false
	}
	return false
}

