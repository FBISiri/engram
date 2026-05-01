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
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/FBISiri/engram/pkg/reflection"
)

// HTTPServer wraps the Engram Server with an HTTP interface.
type HTTPServer struct {
	srv     *Server
	port    int
	apiKey  string
	mux     *http.ServeMux
	httpSrv *http.Server
}

// NewHTTPServer creates an HTTPServer bound to the given port.
// If apiKey is non-empty, every request must carry the header:
//
//	Authorization: Bearer <apiKey>
func NewHTTPServer(s *Server, port int, apiKey string) *HTTPServer {
	h := &HTTPServer{
		srv:    s,
		port:   port,
		apiKey: apiKey,
		mux:    http.NewServeMux(),
	}
	h.registerRoutes()
	return h
}

// registerRoutes wires all HTTP handlers.
func (h *HTTPServer) registerRoutes() {
	// /health is intentionally NOT behind auth — standard practice for
	// liveness/readiness probes (k8s, Docker HEALTHCHECK, monitoring).
	h.mux.HandleFunc("/health", h.handleHealth)
	h.mux.HandleFunc("/reflect", h.withAuth(h.handleReflect))
	h.mux.HandleFunc("/reflect/check", h.withAuth(h.handleReflectCheck))
	h.mux.HandleFunc("/memories/expiry-candidates", h.withAuth(h.handleExpiryCandidates))
	h.mux.HandleFunc("/memories/expired", h.withAuth(h.handleDeleteExpired))
}

// Handler returns the underlying http.Handler for use with httptest.Server or
// any other HTTP server. This allows tests to create a test server without
// binding to a real TCP port via ListenAndServe.
func (h *HTTPServer) Handler() http.Handler {
	return h.mux
}

// ListenAndServe starts the HTTP server. It blocks until ctx is cancelled.
func (h *HTTPServer) ListenAndServe(ctx context.Context) error {
	addr := fmt.Sprintf(":%d", h.port)
	h.httpSrv = &http.Server{
		Addr:         addr,
		Handler:      h.mux,
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
func (h *HTTPServer) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.apiKey != "" {
			auth := r.Header.Get("Authorization")
			expected := "Bearer " + h.apiKey
			if auth != expected {
				writeJSON(w, http.StatusUnauthorized, map[string]string{
					"error": "unauthorized",
				})
				return
			}
		}
		next(w, r)
	}
}

// ─────────────────────────────────────────────────────────────
// Handlers
// ─────────────────────────────────────────────────────────────

// healthResponse is the JSON body returned by GET /health.
type healthResponse struct {
	Status     string `json:"status"`               // "ok" or "degraded"
	Qdrant     string `json:"qdrant"`                // collection status from Qdrant
	PointCount uint64 `json:"point_count"`           // total points in collection
	Error      string `json:"error,omitempty"`       // non-empty when degraded
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
	writeJSON(w, http.StatusOK, healthResponse{
		Status:     "ok",
		Qdrant:     stats.Status,
		PointCount: stats.PointCount,
	})
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
