// Package collection — multi-collection registry for Engram (W20 Day2 Phase 1).
//
// Engram supports multiple logical collections (engram_user, engram_agent_self,
// engram_reflection) so that writes from different caller types can be isolated.
// This avoids polluting the user-facing memory recall with reflection-engine
// generated insights, while still letting cross-collection search merge them
// when needed.
//
// Phase 1 scope (this file): in-memory registry only. Collections are resolved
// from the X-Caller-Type HTTP header; default = engram_user.
//
// Phase 2+ will wire actual Qdrant collection creation/teardown and lifecycle
// management. Persistence of the registry (SQLite vs in-memory rebuild on
// startup) is an open question for Day3 review.
package collection

import (
	"fmt"
	"sync"
	"time"
)

// CollectionInfo describes one registered collection.
type CollectionInfo struct {
	Name      string         `json:"name"`
	TTL       *time.Duration `json:"-"` // serialised separately as human-readable string
	CreatedAt time.Time      `json:"created_at"`
}

// Registry is a thread-safe in-memory map of registered collections.
type Registry struct {
	mu          sync.RWMutex
	collections map[string]CollectionInfo
}

// DefaultRegistry is the process-global registry. cmd/engram/main.go calls
// Init() at startup to register the three baseline collections.
var DefaultRegistry = &Registry{
	collections: make(map[string]CollectionInfo),
}

// Register adds a new collection. Returns an error if a collection with the
// same name already exists.
func (r *Registry) Register(name string, ttl *time.Duration) error {
	if name == "" {
		return fmt.Errorf("collection name cannot be empty")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.collections[name]; exists {
		return fmt.Errorf("collection %q already registered", name)
	}
	r.collections[name] = CollectionInfo{
		Name:      name,
		TTL:       ttl,
		CreatedAt: time.Now().UTC(),
	}
	return nil
}

// List returns a snapshot of all registered collections (unsorted).
func (r *Registry) List() []CollectionInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]CollectionInfo, 0, len(r.collections))
	for _, info := range r.collections {
		out = append(out, info)
	}
	return out
}

// Get fetches one collection by name. Returns (info, true) on hit.
func (r *Registry) Get(name string) (CollectionInfo, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	info, ok := r.collections[name]
	return info, ok
}

// Resolve maps an X-Caller-Type header value to its target collection name.
// Unknown / empty values fall back to engram_user (safe default — never
// silently routes user traffic to a privileged collection).
func (r *Registry) Resolve(callerType string) string {
	switch callerType {
	case "user":
		return CollectionUser
	case "agent-self":
		return CollectionAgentSelf
	case "reflection":
		return CollectionReflection
	default:
		return CollectionUser
	}
}

// Canonical collection names (single source of truth).
const (
	CollectionUser       = "engram_user"
	CollectionAgentSelf  = "engram_agent_self"
	CollectionReflection = "engram_reflection"
)

// Init registers the three baseline collections. Idempotent: re-registration
// errors are swallowed (process restart is the normal path that re-invokes
// Init on an empty registry; explicit re-init via tests is also fine).
func (r *Registry) Init() {
	for _, name := range []string{CollectionUser, CollectionAgentSelf, CollectionReflection} {
		_ = r.Register(name, nil) // ignore "already registered" — idempotent
	}
}
