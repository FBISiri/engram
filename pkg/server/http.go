// Package server — HTTP transport for Engram.
//
// This file adds a standalone HTTP server with a /reflect endpoint that
// triggers the Reflection Engine on demand. It complements the MCP
// (stdio) transport without replacing it.
//
// Endpoints:
//
//	POST /reflect                       — Run one Reflection Engine cycle.
//	                                      Optional JSON body: {"dry_run": true}
//	                                      Returns RunResult JSON.
//
//	GET  /reflect/check                 — Check trigger conditions without running.
//	                                      Returns CheckResult JSON.
//
//	GET  /health                        — Liveness probe. Returns {"status":"ok"}.
//
//	GET  /memories/expiry-candidates    — List memories eligible for policy-based deletion.
//	                                      Returns [{id, type, importance, age_days,
//	                                      content_preview, tags}], capped at 50.
//
//	DELETE /memories/expired            — Execute policy-based cleanup.
//	                                      ?dry_run=true  (default) — report without deleting.
//	                                      ?dry_run=false&confirm=true — delete + snapshot.
//	                                      Returns {deleted_count, skipped_count, snapshot_path}.
//
// v0.2 CRUD routes:
//
//	POST   /memories                    — Create a new memory. Returns 201 + Memory JSON.
//	GET    /memories/{id}               — Get a memory by ID. Returns Memory JSON.
//	PATCH  /memories/{id}               — Partial update. Forbids content field. Enforces FSM.
//	PUT    /memories/{id}               — Full replace (content re-embedded). Preserves lifecycle.
//	DELETE /memories/{id}               — Soft delete: sets lifecycle_status=archived.
//	POST   /memories/{id}/reset         — Restore archived/deprecated → active.
//	POST   /memories/search             — Vector search. Body: {query, limit, include_archived}.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/reflection"
	dto "github.com/prometheus/client_model/go"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// HTTPServer wraps the Engram Server with an HTTP interface.
type HTTPServer struct {
	srv       *Server
	port      int
	apiKey    string
	// principalKeys maps caller type → dedicated API key. A request that
	// authenticates with one of these keys gets its caller type from the
	// key (header ignored). See config.PrincipalKeys.
	principalKeys map[string]string
	mux       *http.ServeMux
	httpSrv   *http.Server
	startTime time.Time
}

// NewHTTPServer creates an HTTPServer bound to the given port.
// If apiKey is non-empty, every request must carry the header:
//
//	Authorization: Bearer <apiKey>
func NewHTTPServer(s *Server, port int, apiKey string) *HTTPServer {
	// W20 Day2 Phase 1: ensure the three baseline collections (engram_user,
	// engram_agent_self, engram_reflection) are registered before we start
	// serving traffic. Idempotent — safe to call repeatedly.
	collection.DefaultRegistry.Init()

	h := &HTTPServer{
		srv:       s,
		port:      port,
		apiKey:    apiKey,
		mux:       http.NewServeMux(),
		startTime: time.Now(),
	}

	// Initialise prometheus metrics and register them on the server so handlers
	// can record observations.
	m := metrics.New(s.embedCache, s.PerCollectionStats)
	s.SetMetrics(m)

	h.registerRoutes()
	return h
}

// SetPrincipalKeys installs per-principal API keys (caller type → key).
// Call before Start(). A nil/empty map disables principal-key auth.
func (h *HTTPServer) SetPrincipalKeys(keys map[string]string) {
	h.principalKeys = keys
}

// registerRoutes wires all HTTP handlers.
func (h *HTTPServer) registerRoutes() {
	// /health is intentionally NOT behind auth — standard practice for
	// liveness/readiness probes (k8s, Docker HEALTHCHECK, monitoring).
	h.mux.HandleFunc("/health", h.handleHealth)
	h.mux.HandleFunc("/reflect", h.withAuth(h.handleReflect))
	h.mux.HandleFunc("/reflect/check", h.withAuth(h.handleReflectCheck))
	h.mux.HandleFunc("GET /memories/expiry-candidates", h.withAuth(h.handleExpiryCandidates))
	h.mux.HandleFunc("DELETE /memories/expired", h.withAuth(h.handleDeleteExpired))

	// v0.2 CRUD routes
	h.mux.HandleFunc("POST /memories", h.withAuth(h.handleCreateMemory))
	h.mux.HandleFunc("POST /memories/search", h.withAuth(h.handleSearchMemories))
	h.mux.HandleFunc("GET /memories/{id}", h.withAuth(h.handleGetMemory))
	h.mux.HandleFunc("PATCH /memories/{id}", h.withAuth(h.handlePatchMemory))
	h.mux.HandleFunc("PUT /memories/{id}", h.withAuth(h.handlePutMemory))
	h.mux.HandleFunc("DELETE /memories/{id}", h.withAuth(h.handleDeleteMemory))
	h.mux.HandleFunc("POST /memories/{id}/reset", h.withAuth(h.handleResetMemory))

	// W20 Day2 Phase 1: collection admin API skeleton.
	h.mux.HandleFunc("POST /collections", h.withAuth(h.handleCreateCollection))
	h.mux.HandleFunc("GET /collections", h.withAuth(h.handleListCollections))

	// W20 Day2 Phase 2: per-collection CRUD routing.
	// Each route validates that the URL collection matches the caller's
	// resolved collection (X-Caller-Type) before delegating.
	h.mux.HandleFunc("POST /collections/{name}/memories", h.withAuth(h.handleCollectionCreateMemory))
	h.mux.HandleFunc("POST /collections/{name}/memories/search", h.withAuth(h.handleCollectionListMemories))
	h.mux.HandleFunc("GET /collections/{name}/memories/{id}", h.withAuth(h.handleCollectionGetMemory))
	h.mux.HandleFunc("PATCH /collections/{name}/memories/{id}", h.withAuth(h.handleCollectionPatchMemory))
	h.mux.HandleFunc("PUT /collections/{name}/memories/{id}", h.withAuth(h.handleCollectionPutMemory))
	h.mux.HandleFunc("DELETE /collections/{name}/memories/{id}", h.withAuth(h.handleCollectionDeleteMemory))
	h.mux.HandleFunc("POST /collections/{name}/memories/{id}/reset", h.withAuth(h.handleCollectionResetMemory))

	// W20 Day2 Phase 2: cross-collection search (strict mode — collections
	// list is required, no implicit all-collection fallback).
	h.mux.HandleFunc("POST /memories/cross-search", h.withAuth(h.handleCrossSearch))

	// P5-A1: embed cache metrics — no auth required (safe to scrape from monitoring).
	h.mux.HandleFunc("GET /metrics", h.handleMetrics)
}

// Handler returns the underlying http.Handler for use with httptest.Server or
// any other HTTP server. This allows tests to create a test server without
// binding to a real TCP port via ListenAndServe.
func (h *HTTPServer) Handler() http.Handler {
	return CallerTypeMiddleware(h.mux)
}

// ListenAndServe starts the HTTP server. It blocks until ctx is cancelled.
func (h *HTTPServer) ListenAndServe(ctx context.Context) error {
	addr := fmt.Sprintf(":%d", h.port)
	h.httpSrv = &http.Server{
		Addr:         addr,
		Handler:      CallerTypeMiddleware(h.mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Shut down gracefully when ctx is done.
	go func() {
		<-ctx.Done()
		shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := h.httpSrv.Shutdown(shutCtx); err != nil {
			log.Printf("engram http: shutdown error: %v", err)
		}
	}()

	log.Printf("engram http: listening on %s", addr)
	if err := h.httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return fmt.Errorf("http server: %w", err)
	}
	return nil
}

// ─────────────────────────────────────────────────────────────
// Middleware
// ─────────────────────────────────────────────────────────────

// withAuth wraps a handler with optional Bearer-token authentication.
//
// Two kinds of credentials are accepted:
//   - the legacy shared apiKey — caller type stays whatever
//     CallerTypeMiddleware derived from the X-Caller-Type header;
//   - a per-principal key (SetPrincipalKeys) — the caller type is forced
//     to the principal that owns the key, ignoring the header. This is
//     what makes collection ownership enforceable rather than self-declared.
func (h *HTTPServer) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.apiKey == "" && len(h.principalKeys) == 0 {
			next(w, r)
			return
		}
		auth := r.Header.Get("Authorization")
		token, isBearer := strings.CutPrefix(auth, "Bearer ")
		if isBearer {
			// Principal keys win: identity derived from the key itself.
			for ct, key := range h.principalKeys {
				if key != "" && token == key {
					next(w, r.WithContext(WithCallerType(r.Context(), ct)))
					return
				}
			}
			if h.apiKey != "" && token == h.apiKey {
				next(w, r)
				return
			}
		}
		writeJSON(w, http.StatusUnauthorized, map[string]string{
			"error": "unauthorized",
		})
	}
}

// ─────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────

// healthResponse is the JSON body returned by GET /health.
type healthResponse struct {
	Status           string                  `json:"status"`
	Qdrant           string                  `json:"qdrant"`
	PointCount       uint64                  `json:"point_count"`
	Error            string                  `json:"error,omitempty"`
	UptimeSeconds    float64                 `json:"uptime_seconds,omitempty"`
	MemoryCount      map[string]uint64       `json:"memory_count,omitempty"`
	LastReflection   *reflection.CheckResult `json:"last_reflection,omitempty"`
	EmbeddingLatency *embeddingLatency       `json:"embedding_latency,omitempty"`
}

type embeddingLatency struct {
	P50Seconds float64 `json:"p50_seconds"`
	P99Seconds float64 `json:"p99_seconds"`
}

// handleMetrics serves all registered Prometheus metrics via the standard
// exposition format. Scrape this endpoint to collect search/embed latency
// histograms, embed cache counters, and per-collection memory counts.
func (h *HTTPServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	promhttp.HandlerFor(h.srv.metrics.Registry, promhttp.HandlerOpts{}).ServeHTTP(w, r)
}

// handleHealth performs a deep health check by pinging the Qdrant collection
// via store.Stats(). Returns 200 with status="ok" when healthy, or 503 with
// status="degraded" when the backend is unreachable.
func (h *HTTPServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	stats, err := h.srv.store.Stats(ctx)
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, healthResponse{
			Status: "degraded",
			Qdrant: "unreachable",
			Error:  err.Error(),
		})
		return
	}
	resp := healthResponse{
		Status:        "ok",
		Qdrant:        stats.Status,
		PointCount:    stats.PointCount,
		UptimeSeconds: time.Since(h.startTime).Seconds(),
	}

	if perCol := h.srv.PerCollectionStats(ctx); perCol != nil {
		counts := make(map[string]uint64, len(perCol)+1)
		var total uint64
		for col, n := range perCol {
			counts[col] = n
			total += n
		}
		counts["total"] = total
		resp.MemoryCount = counts
	}

	eng := reflection.NewEngine(h.srv.store, h.srv.embedder, reflection.DefaultConfig())
	if checkResult, err := eng.Check(ctx); err == nil {
		resp.LastReflection = checkResult
	}

	if h.srv.metrics != nil {
		if mfs, err := h.srv.metrics.Registry.Gather(); err == nil {
			p50, ok50 := histogramPercentile(mfs, "engram_embed_duration_seconds", 0.50)
			p99, ok99 := histogramPercentile(mfs, "engram_embed_duration_seconds", 0.99)
			if ok50 || ok99 {
				resp.EmbeddingLatency = &embeddingLatency{P50Seconds: p50, P99Seconds: p99}
			}
		}
	}

	writeJSON(w, http.StatusOK, resp)
}

// histogramPercentile estimates the given quantile (0–1) from the named
// Prometheus histogram in mfs using linear interpolation across buckets.
// Returns (value, true) when at least one observation exists, (0, false) otherwise.
func histogramPercentile(mfs []*dto.MetricFamily, name string, q float64) (float64, bool) {
	for _, mf := range mfs {
		if mf.GetName() != name {
			continue
		}
		if len(mf.Metric) == 0 {
			return 0, false
		}
		h := mf.Metric[0].GetHistogram()
		if h == nil || h.GetSampleCount() == 0 {
			return 0, false
		}
		total := float64(h.GetSampleCount())
		target := q * total
		var prevBound, prevCount float64
		for _, b := range h.GetBucket() {
			count := float64(b.GetCumulativeCount())
			bound := b.GetUpperBound()
			if count >= target {
				if count == prevCount {
					return bound, true
				}
				return prevBound + (target-prevCount)/(count-prevCount)*(bound-prevBound), true
			}
			prevBound = bound
			prevCount = count
		}
		// All observations landed in the +Inf bucket; fall back to mean.
		return h.GetSampleSum() / total, true
	}
	return 0, false
}

// reflectRequest is the optional JSON body for POST /reflect.
type reflectRequest struct {
	DryRun bool `json:"dry_run"`
}

// handleReflect runs one Reflection Engine cycle.
//
// Method: POST
// Body (optional): {"dry_run": true}
// Response: RunResult JSON
func (h *HTTPServer) handleReflect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{
			"error": "method not allowed — use POST",
		})
		return
	}

	var req reflectRequest
	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error": fmt.Sprintf("invalid JSON body: %v", err),
			})
			return
		}
	}

	cfg := reflection.DefaultConfig()
	cfg.DryRun = req.DryRun

	eng := reflection.NewEngine(h.srv.store, h.srv.embedder, cfg)
	result, err := eng.Run(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("reflection run error: %v", err),
		})
		return
	}
	writeJSON(w, http.StatusOK, result)
}

// handleReflectCheck evaluates trigger conditions without executing a cycle.
//
// Method: GET
// Response: CheckResult JSON
func (h *HTTPServer) handleReflectCheck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{
			"error": "method not allowed — use GET",
		})
		return
	}

	eng := reflection.NewEngine(h.srv.store, h.srv.embedder, reflection.DefaultConfig())
	result, err := eng.Check(r.Context())
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("reflection check error: %v", err),
		})
		return
	}
	writeJSON(w, http.StatusOK, result)
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

// writeJSON serialises v as JSON and writes it to w with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("engram http: writeJSON encode error: %v", err)
	}
}
