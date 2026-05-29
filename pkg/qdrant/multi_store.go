// Package qdrant — MultiStore routes operations across multiple physical Qdrant
// collections (Phase 4: physical collection isolation).
//
// Each logical collection (engram_user, engram_agent_self, engram_reflection)
// maps to a dedicated *Store instance. This replaces the Phase 1-3 approach of
// using a single Qdrant collection with a `collection` payload field filter.
package qdrant

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/FBISiri/engram/pkg/memory"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// MultiStore implements memory.Store by routing operations to the appropriate
// physical Qdrant collection. It is safe for concurrent use.
type MultiStore struct {
	stores     map[string]*Store
	defaultCol string // used when mem.Collection is empty or unknown
}

// NewMultiStore creates a MultiStore backed by the provided store map.
// defaultCol is the fallback collection for writes with an unset Collection field.
func NewMultiStore(stores map[string]*Store, defaultCol string) *MultiStore {
	return &MultiStore{stores: stores, defaultCol: defaultCol}
}

// Close closes all backing stores.
func (m *MultiStore) Close() error {
	var firstErr error
	for _, s := range m.stores {
		if err := s.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// EnsureCollection initializes all physical collections.
func (m *MultiStore) EnsureCollection(ctx context.Context) error {
	for col, s := range m.stores {
		if err := s.EnsureCollection(ctx); err != nil {
			return fmt.Errorf("multi_store: ensure %s: %w", col, err)
		}
	}
	return nil
}

// Insert routes the memory to its target physical collection.
// If mem.Collection is empty or unknown, it falls back to defaultCol and
// stamps mem.Collection so the payload reflects the actual physical collection.
func (m *MultiStore) Insert(ctx context.Context, mem *memory.Memory, vector []float32) error {
	col := mem.Collection
	if col == "" {
		col = m.defaultCol
	}
	s, ok := m.stores[col]
	if !ok {
		col = m.defaultCol
		s = m.stores[col]
	}
	mem.Collection = col
	return s.Insert(ctx, mem, vector)
}

// Search fans out to all stores matching the collection filter (or all stores
// if no collection filter is present). Results are merged and sorted by cosine
// score descending, then truncated to opts.Limit.
//
// A single-store failure is logged as WARN and skipped (partial results) so
// one unavailable collection does not take down the entire search. The B2
// X-Engram-Partial header will be added in Phase 5-B2 on top of this.
//
// Fan-out uses a goroutine per store (currently 3). If future collections are
// added, continue using sync.WaitGroup/errgroup — do not loop-expand manually.
func (m *MultiStore) Search(ctx context.Context, vector []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	targets := m.searchTargets(opts.Filters)

	type result struct {
		col     string
		results []memory.ScoredMemory
		err     error
	}
	ch := make(chan result, len(targets))

	for i, s := range targets {
		col := m.colNameForStore(s, i)
		go func(col string, s *Store) {
			res, err := s.Search(ctx, vector, opts)
			ch <- result{col: col, results: res, err: err}
		}(col, s)
	}

	var all []memory.ScoredMemory
	for range targets {
		r := <-ch
		if r.err != nil {
			log.Printf("WARN handleSearch fan-out: store=%s err=%v", r.col, r.err)
			continue
		}
		all = append(all, r.results...)
	}

	// Global sort by raw cosine score descending, then truncate to limit.
	sort.Slice(all, func(i, j int) bool {
		return all[i].Score > all[j].Score
	})
	if opts.Limit > 0 && len(all) > opts.Limit {
		all = all[:opts.Limit]
	}
	return all, nil
}

// colNameForStore returns a human-readable name for log messages.
// It uses the store's collection name when available, falling back to its index.
func (m *MultiStore) colNameForStore(s *Store, idx int) string {
	if s != nil {
		return s.CollectionName()
	}
	return fmt.Sprintf("store[%d]", idx)
}

// searchTargets returns the stores to query based on the collection filter.
// A {field:"collection", op:OpIn, value:[]string{...}} filter selects specific
// stores; absence of such a filter means all stores.
func (m *MultiStore) searchTargets(filters []memory.Filter) []*Store {
	for _, f := range filters {
		if f.Field == "collection" && f.Op == memory.OpIn {
			cols, ok := f.Value.([]string)
			if !ok || len(cols) == 0 {
				continue
			}
			var targets []*Store
			for _, col := range cols {
				if s, ok := m.stores[col]; ok {
					targets = append(targets, s)
				}
			}
			if len(targets) > 0 {
				return targets
			}
		}
	}
	// No collection filter → fan-out to all stores.
	return m.allStores()
}

// Scroll returns memories from the targeted stores.
// When a single collection filter is present, full Qdrant-native pagination is
// supported. For multi-store fan-out (no filter or multiple collections), the
// implementation paginates through each store completely and returns all
// results without external pagination (next offset = "").
func (m *MultiStore) Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	// Determine if we have a single-collection target.
	col := m.singleScrollTarget(opts.Filters)
	if col != "" {
		s, ok := m.stores[col]
		if !ok {
			return nil, "", nil
		}
		return s.Scroll(ctx, opts)
	}

	// Fan-out: paginate through every store completely.
	targets := m.allStores()
	var all []memory.Memory
	for _, s := range targets {
		pageOpts := opts
		pageOpts.Offset = "" // always start from the beginning per store
		for {
			batch, next, err := s.Scroll(ctx, pageOpts)
			if err != nil {
				return nil, "", err
			}
			all = append(all, batch...)
			if next == "" {
				break
			}
			pageOpts.Offset = next
		}
	}
	return all, "", nil
}

// singleScrollTarget returns the collection name if the filters specify exactly
// one collection; otherwise returns "".
func (m *MultiStore) singleScrollTarget(filters []memory.Filter) string {
	for _, f := range filters {
		if f.Field == "collection" && f.Op == memory.OpIn {
			cols, ok := f.Value.([]string)
			if ok && len(cols) == 1 {
				return cols[0]
			}
		}
	}
	return ""
}

// Delete removes memories by IDs from all physical collections.
// Qdrant delete is idempotent (no error for non-existent IDs).
func (m *MultiStore) Delete(ctx context.Context, ids []string) (int, error) {
	for _, s := range m.stores {
		if _, err := s.Delete(ctx, ids); err != nil {
			return 0, err
		}
	}
	return len(ids), nil
}

// Update modifies payload fields of a memory across all physical collections.
// SetPayload on a non-existent ID is documented as a no-op, but the gRPC
// layer may surface a NotFound status when the point is not in a given
// collection. We ignore NotFound errors so the fan-out never fails because
// the point only lives in one of the three physical collections.
//
// Resilience: we try every store regardless of errors. If at least one store
// succeeds we return nil immediately (the ID was found and updated). If all
// stores error we return nil when all errors are NotFound-type (the ID was
// already deleted — treat as an orphan success so callers move forward). Only
// when all stores error AND at least one error is a genuine non-NotFound error
// do we surface an error. This prevents a broken/empty store (e.g.
// engram_agent_self at 0 points) from blocking updates to the correct store.
func (m *MultiStore) Update(ctx context.Context, id string, fields map[string]any) error {
	var firstRealErr error
	for _, s := range m.stores {
		if err := s.Update(ctx, id, fields); err != nil {
			if !isGRPCNotFound(err) && firstRealErr == nil {
				firstRealErr = err
			}
			continue // try remaining stores regardless of error type
		}
		return nil // at least one store succeeded — ID was updated
	}
	// All stores errored. firstRealErr is nil iff every error was NotFound
	// (orphan — the memory was deleted). Treat as success so the engine can
	// advance; the orphan won't appear in future Scroll results.
	return firstRealErr
}

// isGRPCNotFound walks the error chain looking for a gRPC NotFound status.
// Also matches any gRPC error whose message contains "not found" — Qdrant
// versions have varied in which status code they return for SetPayload on a
// non-existent point (NotFound, Internal, Unknown, and others observed in
// production). Matching the message broadly is the safest defence.
func isGRPCNotFound(err error) bool {
	for err != nil {
		if s, ok := status.FromError(err); ok {
			if s.Code() == codes.NotFound {
				return true
			}
			// Belt-and-suspenders: match any gRPC code whose message says "not found".
			if strings.Contains(strings.ToLower(s.Message()), "not found") {
				return true
			}
		}
		err = errors.Unwrap(err)
	}
	return false
}

// SearchByIDs retrieves specific memories by their IDs from all stores.
// A single-store failure is logged as WARN and skipped (partial results) so
// a broken/empty store does not prevent lookups against healthy stores.
// Callers that need strict all-or-nothing semantics should check the length
// of the result against the input, but for existence-check uses (e.g.
// filterExistingIDs) partial results are preferable to a hard error.
func (m *MultiStore) SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error) {
	var all []memory.Memory
	for _, s := range m.stores {
		results, err := s.SearchByIDs(ctx, ids)
		if err != nil {
			log.Printf("WARN SearchByIDs fan-out: store=%s err=%v (skipping)", s.CollectionName(), err)
			continue // partial results — don't let one broken store block others
		}
		all = append(all, results...)
	}
	return all, nil
}

// Stats returns aggregated statistics across all physical collections.
func (m *MultiStore) Stats(ctx context.Context) (*memory.CollectionStats, error) {
	agg := &memory.CollectionStats{}
	for _, s := range m.stores {
		stats, err := s.Stats(ctx)
		if err != nil {
			return nil, err
		}
		agg.PointCount += stats.PointCount
		agg.VectorCount += stats.VectorCount
		agg.IndexedCount += stats.IndexedCount
		agg.SegmentCount += stats.SegmentCount
		agg.Status = stats.Status
	}
	return agg, nil
}

// DeleteExpired removes expired memories from all physical collections.
func (m *MultiStore) DeleteExpired(ctx context.Context) (int, error) {
	total := 0
	for _, s := range m.stores {
		n, err := s.DeleteExpired(ctx)
		if err != nil {
			return total, err
		}
		total += n
	}
	return total, nil
}

// PerCollectionStats returns the point count for each physical collection.
// Errors for individual stores are silently skipped (partial results returned).
func (m *MultiStore) PerCollectionStats(ctx context.Context) map[string]uint64 {
	result := make(map[string]uint64, len(m.stores))
	for name, s := range m.stores {
		stats, err := s.Stats(ctx)
		if err != nil {
			continue
		}
		result[name] = stats.PointCount
	}
	return result
}

// allStores returns all store instances in a consistent (name-sorted) order.
func (m *MultiStore) allStores() []*Store {
	names := make([]string, 0, len(m.stores))
	for name := range m.stores {
		names = append(names, name)
	}
	sort.Strings(names)
	stores := make([]*Store, 0, len(names))
	for _, name := range names {
		stores = append(stores, m.stores[name])
	}
	return stores
}
