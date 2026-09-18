package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/trajectory"
)

// ─────────────────────────────────────────────────────────────
// POST /memories — create a new memory
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleCreateMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	createStart := time.Now()

	var body struct {
		Type       string         `json:"type"`
		Content    string         `json:"content"`
		Source     string         `json:"source"`
		Importance float64        `json:"importance"`
		Tags       []string       `json:"tags"`
		ValidUntil float64        `json:"valid_until"`
		Metadata   map[string]any `json:"metadata"`
		TaskID     string         `json:"task_id"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body too large"})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}
	if body.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content is required"})
		return
	}

	// Optional event-loop task id: JSON body wins, else X-Task-ID header (same
	// semantics as memory_search's task_id). Recorded on the candidate record.
	taskID := body.TaskID
	if taskID == "" {
		taskID = r.Header.Get("X-Task-ID")
	}

	memType := memory.MemoryType(body.Type)
	if body.Type == "" {
		memType = memory.TypeEvent
	} else if !memory.ValidTypes[memType] {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid type: %s", body.Type)})
		return
	}

	var rawSourceType string
	if body.Metadata != nil {
		rawSourceType, _ = body.Metadata["source_type"].(string)
	}

	var importance float64
	var importanceClamped bool
	if h.srv.amacEnabled() {
		importance, importanceClamped = h.srv.amacImportance(memType, body.Importance)
		// R6 governance: a directive from Frank (user_input) gets a +1 boost.
		if memType == memory.TypeDirective && rawSourceType == "user_input" {
			importance++
			if importance > 10 {
				importance = 10
			}
		}
	} else {
		importance = body.Importance
		if importance <= 0 {
			importance = 5.0
		}
		if importance < 1 {
			importance = 1
		}
		if importance > 10 {
			importance = 10
		}
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
	// Route the write to the caller's resolved collection (e.g. a pigo
	// principal key → engram_pigo). Mirrors the MCP add path in server.go;
	// without this stamp MultiStore.Insert falls back to the default store
	// (engram_user) and physical isolation is lost. Defaults to engram_user
	// for header-less legacy callers.
	mem.Collection = CollectionFromContext(r.Context())
	if body.Metadata != nil {
		mem.Metadata = body.Metadata
	}

	// C1 provenance (shared helper).
	if mem.Metadata == nil {
		mem.Metadata = map[string]any{}
	}
	sourceType, provided, perr := extractMetaSourceType(mem.Metadata)
	if perr != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": perr.Error()})
		return
	}
	if err := h.srv.applyProvenance(mem.Metadata, sourceType, provided, "REST POST /memories"); err != nil {
		code, msg := provenanceStatus(err)
		writeJSON(w, code, map[string]string{"error": msg})
		return
	}

	// D4: pre-admission candidate flow recording. Register a single deferred
	// trajectory write that fires once at handler exit, tagged with the
	// admission decision reached by the gates below. Async + non-blocking.
	admissionDecision := "admitted"
	gateDetails := ""
	var dedupTopScore float64
	if h.srv.traj != nil {
		defer func() {
			h.srv.traj.Log(trajectory.Record{
				Timestamp:         time.Now().UTC().Format(time.RFC3339),
				Operation:         "candidate",
				Content:           body.Content,
				Type:              string(memType),
				Importance:        importance,
				SourceType:        sourceType,
				Tags:              tags,
				AdmissionDecision: admissionDecision,
				GateDetails:       gateDetails,
				DedupTopScore:     dedupTopScore,
				LatencyMs:         time.Since(createStart).Milliseconds(),
				Caller:            CallerTypeFromContext(r.Context()),
				TaskID:            taskID,
			})
			if h.srv.metrics != nil {
				h.srv.metrics.AdmissionTotal.WithLabelValues(string(memType), admissionDecision).Inc()
			}
		}()
	}

	// A-MAC (R2/R6): annotate importance clamp + identity governance flag, then
	// enforce the per-type write-frequency limit before spending an embedding.
	if h.srv.amacEnabled() {
		if importanceClamped {
			mem.Metadata["importance_clamped"] = true
		}
		if memType == memory.TypeIdentity && sourceType == "reflection" {
			mem.Metadata["needs_review"] = true
		}
		if err := h.srv.rateLimiter.Allow(mem.Collection, memType); err != nil {
			var rle *RateLimitError
			if errors.As(err, &rle) {
				admissionDecision = "rate_limited"
				gateDetails = fmt.Sprintf("rate_limited: retry_after=%ds", rle.RetryAfterSeconds)
				writeJSON(w, http.StatusTooManyRequests, map[string]any{
					"status":              "rate_limited",
					"type":                string(memType),
					"retry_after_seconds": rle.RetryAfterSeconds,
					"error":               rle.Error(),
				})
				return
			}
		}
	}

	embedStart := time.Now()
	vec, err := h.srv.embedder.Embed(r.Context(), body.Content)
	if h.srv.metrics != nil {
		h.srv.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		admissionDecision = "error"
		gateDetails = fmt.Sprintf("embed_error: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("embed error: %v", err)})
		return
	}

	// Server-side dedup (C1): mirror the MCP add path so REST writers
	// (e.g. the Consolidation Agent) don't bypass dedup. The threshold is
	// resolved per memory type (A-MAC). Fail-open on error.
	dedupResult, dupErr := h.srv.checkDedup(r.Context(), vec, body.Content, sourceType, memType)
	if dupErr != nil {
		log.Printf("[WARN] engram REST POST /memories: dedup check failed, proceeding with insert: %v", dupErr)
		dedupResult = &DedupResult{}
	} else if dedupResult.DupFound {
		if h.srv.metrics != nil {
			h.srv.metrics.DedupHits.WithLabelValues(mem.Collection, dedupTypeLabel(dedupResult.Threshold)).Inc()
			h.srv.metrics.MemoryOps.WithLabelValues("add", mem.Collection, sourceType).Inc()
		}
		var dup struct {
			Existing struct {
				ID    string  `json:"id"`
				Score float64 `json:"score"`
			} `json:"existing"`
		}
		_ = json.Unmarshal(dedupResult.DupData, &dup)
		admissionDecision = "dedup_rejected"
		dedupTopScore = dedupResult.TopScore
		gateDetails = fmt.Sprintf("dedup score=%.4f against id=%s", dup.Existing.Score, dup.Existing.ID)
		writeJSON(w, http.StatusConflict, map[string]any{
			"status":      "duplicate",
			"existing_id": dup.Existing.ID,
			"similarity":  dup.Existing.Score,
		})
		return
	}
	dedupTopScore = dedupResult.TopScore

	if err := h.srv.store.Insert(r.Context(), mem, vec); err != nil {
		admissionDecision = "error"
		gateDetails = fmt.Sprintf("insert_error: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("insert error: %v", err)})
		return
	}

	// Write Discipline Checkpoints (R10): CP1/CP2/CP3/CP4 apply on REST create.
	var advisories []Advisory
	if h.srv.writeCheckpointsEnabled() {
		if a := h.srv.cp1DedupAdvisory(dedupResult.Candidates, memType); a != nil {
			advisories = append(advisories, *a)
			if h.srv.metrics != nil {
				h.srv.metrics.CheckpointTotal.WithLabelValues("cp1", "advisory").Inc()
			}
		} else if h.srv.metrics != nil {
			h.srv.metrics.CheckpointTotal.WithLabelValues("cp1", "clean").Inc()
		}
		if a := h.srv.cp2Importance(mem.Collection, importance); a != nil {
			advisories = append(advisories, *a)
			if h.srv.metrics != nil {
				h.srv.metrics.CheckpointTotal.WithLabelValues("cp2", "advisory").Inc()
			}
		} else if h.srv.metrics != nil {
			h.srv.metrics.CheckpointTotal.WithLabelValues("cp2", "clean").Inc()
		}
		if a := h.srv.cp3RateLimitAdvisory(mem.Collection, memType); a != nil {
			advisories = append(advisories, *a)
		}
		advisories = append(advisories, h.srv.cp4Content(body.Content, tags, sourceType)...)
		for _, a := range advisories {
			logAdvisory(a)
		}
	}

	type createResponse struct {
		memory.Memory
		Advisories []Advisory `json:"advisories,omitempty"`
	}
	writeJSON(w, http.StatusCreated, createResponse{Memory: *mem, Advisories: advisories})
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
	type memoryWithSourceType struct {
		memory.Memory
		SourceType string `json:"source_type,omitempty"`
	}
	writeJSON(w, http.StatusOK, memoryWithSourceType{
		Memory:     mems[0],
		SourceType: sourceTypeFromMetadata(mems[0].Metadata),
	})
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
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(&raw); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body too large"})
			return
		}
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

	// Other patchable fields. A recognized field that is present but fails to
	// unmarshal is a client error (400) rather than a silent skip.
	if rawTags, ok := raw["tags"]; ok {
		var tags []string
		if err := json.Unmarshal(rawTags, &tags); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid tags: expected array of strings"})
			return
		}
		tagsAny := make([]any, len(tags))
		for i, t := range tags {
			tagsAny[i] = t
		}
		updates["tags"] = tagsAny
	}
	if rawImp, ok := raw["importance"]; ok {
		var imp float64
		if err := json.Unmarshal(rawImp, &imp); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid importance: expected number"})
			return
		}
		updates["importance"] = clampImportance(imp)
	}
	if rawSrc, ok := raw["source"]; ok {
		var src string
		if err := json.Unmarshal(rawSrc, &src); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid source: expected string"})
			return
		}
		updates["source"] = src
	}
	if rawMeta, ok := raw["metadata"]; ok {
		var meta map[string]any
		if err := json.Unmarshal(rawMeta, &meta); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid metadata: expected object"})
			return
		}
		if st, ok := meta["source_type"]; ok {
			stStr, isStr := st.(string)
			if !isStr {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid source_type: %v", st)})
				return
			}
			if err := h.srv.validateProvenance(stStr); err != nil {
				code, msg := provenanceStatus(err)
				writeJSON(w, code, map[string]string{"error": msg})
				return
			}
		}
		updates["metadata"] = meta
	}

	// C1 provenance (R4): PATCH is a partial update. Only warn when, after the
	// patch is applied, the memory still has no source_type. If metadata is being
	// replaced, the presence is determined by the new map; otherwise it depends
	// on the current stored memory.
	hasSourceType := false
	if _, ok := raw["metadata"]; ok {
		if metaAny, ok := updates["metadata"].(map[string]any); ok {
			if _, has := metaAny["source_type"]; has {
				hasSourceType = true
			}
		}
	} else if current.Metadata != nil {
		if _, has := current.Metadata["source_type"]; has {
			hasSourceType = true
		}
	}
	if !hasSourceType {
		if h.srv.cfg != nil && h.srv.cfg.ProvenanceMode == "strict" {
			log.Printf("[WARN] engram REST PATCH /memories/%s: source_type not provided, rejecting (strict mode)", id)
			writeJSON(w, http.StatusUnprocessableEntity, map[string]string{"error": strictProvenanceMsg})
			return
		}
		// Inject the default so the memory is not stored without provenance.
		if metaAny, ok := updates["metadata"].(map[string]any); ok {
			metaAny["source_type"] = string(memory.DefaultSourceType)
		} else {
			newMeta := map[string]any{}
			if current.Metadata != nil {
				for k, v := range current.Metadata {
					newMeta[k] = v
				}
			}
			newMeta["source_type"] = string(memory.DefaultSourceType)
			updates["metadata"] = newMeta
		}
		log.Printf("[WARN] engram REST PATCH /memories/%s: source_type not provided, defaulting to 'unknown'", id)
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
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body too large"})
			return
		}
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
	importance = clampImportance(importance)

	source := body.Source
	if source == "" {
		source = prev.Source
	}

	tags := body.Tags
	if tags == nil {
		tags = prev.Tags
	}

	// TTL: compute valid_until from the TTL matrix (or honor an explicit,
	// future-dated value), matching the POST path.
	ttlCfg := memory.DefaultTTLConfig()
	computedValidUntil := memory.ComputeValidUntil(ttlCfg, memType, importance, tags, body.ValidUntil)

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
		ValidUntil:         computedValidUntil,
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

	// Spec §5.5: revive a memory that was deprecated by evaporation when the
	// update lifts its effective importance back to/above the eviction threshold.
	h.reviveIfEvaporated(prev, mem)

	// C1 provenance (shared helper).
	sourceType, provided, perr := extractMetaSourceType(mem.Metadata)
	if perr != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": perr.Error()})
		return
	}
	if err := h.srv.applyProvenance(mem.Metadata, sourceType, provided, "REST PUT /memories"); err != nil {
		code, msg := provenanceStatus(err)
		writeJSON(w, code, map[string]string{"error": msg})
		return
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

	// Write Discipline Checkpoints (R10): PUT is an update, so CP2/CP4 apply
	// (CP1/CP3 do not). Mirrors the POST createResponse wrap.
	mem.Collection = CollectionFromContext(r.Context())
	var advisories []Advisory
	if h.srv.writeCheckpointsEnabled() {
		if a := h.srv.cp2Importance(mem.Collection, importance); a != nil {
			advisories = append(advisories, *a)
		}
		advisories = append(advisories, h.srv.cp4Content(body.Content, tags, sourceType)...)
		for _, a := range advisories {
			logAdvisory(a)
		}
	}

	type putResponse struct {
		memory.Memory
		Advisories []Advisory `json:"advisories,omitempty"`
	}
	writeJSON(w, http.StatusOK, putResponse{Memory: *mem, Advisories: advisories})
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
		SourceType      []string `json:"source_type"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
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
	// resolvedCollection is reported as `resolved_collection` ONLY when the
	// search is genuinely collection-scoped (a collection filter is applied at
	// the store layer). For legacy fan-out callers the search is NOT scoped, so
	// it stays empty and omitempty drops it. An explicit body `collection`
	// field is still validated (unknown → 400) but does not by itself scope the
	// fan-out, hence does not populate resolved_collection.
	var resolvedCollection string
	if req.Collection != "" {
		if _, ok := collection.DefaultRegistry.Get(req.Collection); !ok {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error":      "unknown collection: " + req.Collection,
				"collection": req.Collection,
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

	// READ ISOLATION: an isolated caller-type (e.g. pigo, authenticated via its
	// principal key) may ONLY read its own store — never fan out across all
	// collections. Force the collection filter to the caller's own collection,
	// ignoring any self-declared `collection` body field. Legacy
	// user/reflection/agent-self callers are untouched → they keep the
	// cross-store fan-out that Siri/BMO rely on.
	if ct := CallerTypeFromContext(r.Context()); collection.IsIsolatedCallerType(ct) {
		own := collection.DefaultRegistry.Resolve(ct)
		resolvedCollection = own
		filters = append(filters, memory.Filter{Field: "collection", Op: memory.OpIn, Value: []string{own}})
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
	evapCfg := h.srv.evaporationConfig()
	results = rerankResults(results, h.srv.weights, h.srv.decay, evapCfg, h.srv.mmrLambda, limit)

	// Update access_count and last_accessed_source asynchronously.
	callerType := CallerTypeFromContext(r.Context())
	items := make([]accessUpdate, len(results))
	for i, res := range results {
		items[i] = accessUpdate{ID: res.ID, AccessCount: res.AccessCount}
	}
	asyncUpdateAccessCounts(h.srv.store, nil, items, callerType, true)

	type result struct {
		memory.Memory
		Score               float64 `json:"score"`
		EffectiveImportance float64 `json:"effective_importance"`
		ResolvedCollection  string  `json:"resolved_collection,omitempty"`
		SourceType          string  `json:"source_type,omitempty"`
	}
	output := make([]result, len(results))
	for i, r := range results {
		effImp := memory.EffectiveImportance(&results[i].Memory, evapCfg, time.Now())
		if h.srv.metrics != nil {
			h.srv.metrics.EvaporationEffectiveImportance.WithLabelValues(string(r.Type)).Observe(effImp)
		}
		output[i] = result{Memory: r.Memory, Score: r.Score, EffectiveImportance: effImp, ResolvedCollection: resolvedCollection, SourceType: sourceTypeFromMetadata(r.Metadata)}
	}

	writeJSON(w, http.StatusOK, output)
}

// ─────────────────────────────────────────────────────────────
// POST /memories/list — filter-only time-window listing (no vector search)
//
// Body: {time_start?, time_end?, collections?, limit?}. limit defaults to 50
// and is silently capped to a maximum of 100.
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleListMemories(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}

	var req struct {
		TimeStart   float64  `json:"time_start"`
		TimeEnd     float64  `json:"time_end"`
		Collections []string `json:"collections"`
		Limit       int      `json:"limit"`
	}
	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"error": "request body too large"})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("invalid JSON: %v", err)})
		return
	}

	limit := req.Limit
	if limit <= 0 {
		limit = memory.DefaultListLimit
	}
	if limit > 100 {
		limit = 100
	}

	// Resolve collection scope. READ ISOLATION: an isolated caller may ONLY
	// read its own collection; a self-declared collections field cannot widen
	// scope.
	collections := req.Collections
	if ct := CallerTypeFromContext(r.Context()); collection.IsIsolatedCallerType(ct) {
		collections = []string{collection.DefaultRegistry.Resolve(ct)}
	} else if len(collections) > 0 {
		for _, col := range collections {
			if _, ok := collection.DefaultRegistry.Get(col); !ok {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unknown collection: " + col})
				return
			}
		}
	}

	opts := memory.ListMemoriesOptions{
		TimeStart:   req.TimeStart,
		TimeEnd:     req.TimeEnd,
		Collections: collections,
		Limit:       limit,
	}

	mems, err := memory.ListMemories(r.Context(), h.srv.store, opts)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("scroll error: %v", err)})
		return
	}

	type listResult struct {
		ID               string   `json:"id"`
		Content          string   `json:"content"`
		Type             string   `json:"type"`
		Importance       float64  `json:"importance"`
		CreatedAt        float64  `json:"created_at"`
		Tags             []string `json:"tags"`
		SourceCollection string   `json:"source_collection"`
	}

	output := make([]listResult, len(mems))
	for i, m := range mems {
		output[i] = listResult{
			ID:               m.ID,
			Content:          m.Content,
			Type:             string(m.Type),
			Importance:       m.Importance,
			CreatedAt:        m.CreatedAt,
			Tags:             m.Tags,
			SourceCollection: collectionOrFallback(m.Collection, collection.CollectionUser),
		}
	}

	writeJSON(w, http.StatusOK, output)
}

// handleEvaporationStatus handles GET /memories/evaporation-status. It returns
// the server's effective evaporation config plus current stats: the count of
// evaporation-deprecated memories by type and an estimate of the next sweep
// time.
func (h *HTTPServer) handleEvaporationStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed — use GET"})
		return
	}

	cfg := h.srv.evaporationConfig()

	deprecatedByType := map[string]int{}
	var offset string
	for {
		mems, next, err := h.srv.store.Scroll(r.Context(), memory.ScrollOptions{Limit: 100, Offset: offset})
		if err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("scroll error: %v", err)})
			return
		}
		for i := range mems {
			m := &mems[i]
			if m.LifecycleStatus != memory.LifecycleDeprecated {
				continue
			}
			if reason, _ := m.Metadata["deprecated_reason"].(string); reason == "evaporation" {
				deprecatedByType[string(m.Type)]++
			}
		}
		if next == "" {
			break
		}
		offset = next
	}

	var nextSweep any
	var nextBasis any
	var lastSweepAt any
	if cfg.Enabled && cfg.SweepIntervalH > 0 {
		last, hasSwept, started := evapSweepClock.snapshot()
		interval := time.Duration(cfg.SweepIntervalH) * time.Hour
		if est, basis, ok := nextSweepEstimate(last, hasSwept, started, interval); ok {
			nextSweep = est.UTC().Format(time.RFC3339)
			nextBasis = basis
		}
		if hasSwept {
			lastSweepAt = last.UTC().Format(time.RFC3339)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"config":                   cfg,
		"deprecated_count_by_type": deprecatedByType,
		"next_sweep_estimate":      nextSweep,
		"next_sweep_basis":         nextBasis,
		"last_sweep_at":            lastSweepAt,
	})
}

// reviveIfEvaporated restores m to "active" when prev was deprecated by
// evaporation and m's recomputed effective importance now meets the eviction
// threshold (spec §5.5). It clears the deprecated_* metadata keys on m. Returns
// true if a revival occurred.
func (h *HTTPServer) reviveIfEvaporated(prev memory.Memory, m *memory.Memory) bool {
	if prev.LifecycleStatus != memory.LifecycleDeprecated {
		return false
	}
	if reason, _ := prev.Metadata["deprecated_reason"].(string); reason != "evaporation" {
		return false
	}
	cfg := h.srv.evaporationConfig()
	if memory.EffectiveImportance(m, cfg, time.Now()) < cfg.EvictionThreshold {
		return false
	}
	m.LifecycleStatus = memory.LifecycleActive
	if m.Metadata != nil {
		delete(m.Metadata, "deprecated_reason")
		delete(m.Metadata, "deprecated_at")
		delete(m.Metadata, "effective_importance_at_deprecation")
	}
	return true
}

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
