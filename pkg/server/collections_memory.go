// collections_memory.go — per-collection CRUD routing (W20 Day2 Phase 2).
//
// Phase 2 scope: surface-level routing only. The handlers under
// /collections/{name}/memories validate that the URL-supplied collection
// matches the caller's resolved collection (per X-Caller-Type), and then
// delegate to the existing CRUD handlers. Underlying Qdrant storage is
// still single-collection — Phase 4 will wire per-collection Qdrant
// lifecycle and physical isolation.
//
// This gives BMO + reflection-engine a stable REST surface to start
// migrating against (reflection writes go to /collections/engram_reflection/...
// with X-Caller-Type: reflection), and we can swap the storage backend
// later without breaking callers.
package server

import (
	"net/http"

	"github.com/FBISiri/engram/pkg/collection"
)

// requireCollectionMatch validates that the {name} path param refers to
// a registered collection AND matches the caller's resolved collection.
// On mismatch, writes 403 + JSON error and returns false. On success
// returns true and the caller should proceed with the underlying handler.
//
// Rationale: namespace isolation must be enforced server-side. A caller
// with X-Caller-Type=user MUST NOT be able to write to engram_reflection
// just by hitting /collections/engram_reflection/memories.
func (h *HTTPServer) requireCollectionMatch(w http.ResponseWriter, r *http.Request) bool {
	name := r.PathValue("name")
	if name == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "collection name required"})
		return false
	}
	if _, ok := collection.DefaultRegistry.Get(name); !ok {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "collection not registered: " + name})
		return false
	}
	resolved := CollectionFromContext(r.Context())
	if name != resolved {
		writeJSON(w, http.StatusForbidden, map[string]string{
			"error":             "X-Caller-Type does not own this collection",
			"path_collection":   name,
			"caller_collection": resolved,
		})
		return false
	}
	return true
}

// ─────────────────────────────────────────────────────────────
// Per-collection CRUD wrappers
// ─────────────────────────────────────────────────────────────

func (h *HTTPServer) handleCollectionCreateMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handleCreateMemory(w, r)
}

func (h *HTTPServer) handleCollectionListMemories(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	// Phase 2: degrade to search with empty query → return recent N.
	// For now we expose this as a thin wrapper over /memories/search using
	// the caller's collection only. Body fields are forwarded.
	h.handleSearchMemories(w, r)
}

func (h *HTTPServer) handleCollectionGetMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handleGetMemory(w, r)
}

func (h *HTTPServer) handleCollectionPatchMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handlePatchMemory(w, r)
}

func (h *HTTPServer) handleCollectionPutMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handlePutMemory(w, r)
}

func (h *HTTPServer) handleCollectionDeleteMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handleDeleteMemory(w, r)
}

func (h *HTTPServer) handleCollectionResetMemory(w http.ResponseWriter, r *http.Request) {
	if !h.requireCollectionMatch(w, r) {
		return
	}
	h.handleResetMemory(w, r)
}
