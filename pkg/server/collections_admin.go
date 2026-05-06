// collections_admin.go — admin API skeleton for collection management
// (W20 Day2 Phase 1).
//
// Endpoints:
//
//	POST /collections        — register a new collection. Body: {"name":"...","ttl":"30d"}.
//	                           Returns 201 + collection info on success, 409 on duplicate,
//	                           400 on bad request.
//	GET  /collections        — list all registered collections.
//
// Phase 1 scope: in-memory registry only. Phase 2+ will wire actual Qdrant
// collection lifecycle (create/drop) and persistence.
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
)

// createCollectionRequest is the JSON body for POST /collections.
type createCollectionRequest struct {
	Name string `json:"name"`
	TTL  string `json:"ttl,omitempty"` // e.g. "30d", "720h", "" = no TTL
}

// collectionView is the JSON shape returned by the admin endpoints.
type collectionView struct {
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
	TTL       string    `json:"ttl,omitempty"`
}

// parseTTL accepts either a Go duration string (e.g. "720h") or a "Nd"
// shorthand for whole days. Empty input → nil (no TTL).
func parseTTL(s string) (*time.Duration, error) {
	if s == "" {
		return nil, nil
	}
	if len(s) > 1 && s[len(s)-1] == 'd' {
		days, err := strconv.Atoi(s[:len(s)-1])
		if err != nil || days <= 0 {
			return nil, fmt.Errorf("invalid ttl %q: expected positive integer days (e.g. 30d)", s)
		}
		d := time.Duration(days) * 24 * time.Hour
		return &d, nil
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return nil, fmt.Errorf("invalid ttl %q: %w", s, err)
	}
	if d <= 0 {
		return nil, fmt.Errorf("ttl must be positive, got %s", d)
	}
	return &d, nil
}

func toCollectionView(c collection.CollectionInfo) collectionView {
	v := collectionView{Name: c.Name, CreatedAt: c.CreatedAt}
	if c.TTL != nil {
		v.TTL = c.TTL.String()
	}
	return v
}

// handleCreateCollection — POST /collections
func (h *HTTPServer) handleCreateCollection(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed — use POST"})
		return
	}
	var body createCollectionRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body: " + err.Error()})
		return
	}
	if body.Name == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "name is required"})
		return
	}
	ttl, err := parseTTL(body.TTL)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if err := collection.DefaultRegistry.Register(body.Name, ttl); err != nil {
		writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
		return
	}
	info, _ := collection.DefaultRegistry.Get(body.Name)
	writeJSON(w, http.StatusCreated, toCollectionView(info))
}

// handleListCollections — GET /collections
func (h *HTTPServer) handleListCollections(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed — use GET"})
		return
	}
	cols := collection.DefaultRegistry.List()
	out := make([]collectionView, 0, len(cols))
	for _, c := range cols {
		out = append(out, toCollectionView(c))
	}
	writeJSON(w, http.StatusOK, map[string]any{"collections": out})
}
