package memory

import (
	"context"
	"sort"
)

// FieldCreatedAt is the payload field name for a memory's creation timestamp.
const FieldCreatedAt = "created_at"

// FieldCollection is the payload field name for a memory's logical collection.
const FieldCollection = "collection"

// DefaultListLimit is the default number of entries returned by a time-window
// listing when no explicit limit is supplied.
const DefaultListLimit = 50

// scrollPageSize is the fixed per-page fetch size used internally by
// ListMemories' pagination loop. It is deliberately DECOUPLED from the caller's
// result limit: coupling them breaks under inclusive-offset scroll semantics
// (a page of size N advances by only N-1 new IDs, so limit=1 would advance by
// zero and stop after a single arbitrary point). A large fixed page guarantees
// forward progress regardless of how small the result limit is.
const scrollPageSize = 512

// ListMemoriesOptions describes a filter-only, time-window listing request. It
// is the semantic (time_start/time_end) shape that the transport layer speaks;
// BuildTimeWindowScrollOptions translates it into the lower-level ScrollOptions
// understood by Store.Scroll (which has no time semantics of its own).
type ListMemoriesOptions struct {
	TimeStart   float64  // Unix ts lower bound; 0 = no lower bound.
	TimeEnd     float64  // Unix ts upper bound; 0 = no upper bound.
	Collections []string // logical collection names; empty = all collections.
	Limit       int      // <= 0 => DefaultListLimit.
}

// BuildTimeWindowScrollOptions translates a time-window listing request into
// filter-only ScrollOptions. time_start/time_end become created_at gte/lte
// filters; collections becomes a collection IN filter. It performs no vector
// search and incurs no embedding cost. Encapsulated here (rather than inline in
// a transport handler) so future list-by-type / list-by-tags requirements reuse
// the same translation.
func BuildTimeWindowScrollOptions(opts ListMemoriesOptions) ScrollOptions {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}

	var filters []Filter
	if opts.TimeStart > 0 {
		filters = append(filters, Filter{Field: FieldCreatedAt, Op: OpGte, Value: opts.TimeStart})
	}
	if opts.TimeEnd > 0 {
		filters = append(filters, Filter{Field: FieldCreatedAt, Op: OpLte, Value: opts.TimeEnd})
	}
	if len(opts.Collections) > 0 {
		filters = append(filters, Filter{Field: FieldCollection, Op: OpIn, Value: opts.Collections})
	}

	return ScrollOptions{Limit: limit, Filters: filters}
}

// ListMemories performs a filter-only, time-window listing: it paginates
// Store.Scroll fully, then returns the NEWEST `limit` entries within the window,
// presented in created_at ASCENDING order (chronological display order for the
// Recent Memory view). No vector search, no embedding cost.
//
// Full pagination is REQUIRED for correctness: Qdrant scroll returns points in
// point-ID order (NOT time order), so the true oldest-N cannot be known from a
// single page — every window match must be collected before sorting and
// truncating. The fan-out MultiStore path already returns all matches in one
// call (next=""), so the loop runs once there; the single-collection path is
// paginated here so both paths are consistent.
//
// COST / v1 LIMITATION: this loads every matching entry of each target store
// into memory before truncating — there is no Qdrant order_by pushdown to fetch
// only the oldest-N server-side. This is acceptable ONLY because the sole caller
// always passes a bounded created_at time window (plus a limit); it is NOT cheap
// on an unfiltered whole-DB call. Do not use as a general "dump everything" path.
//
// Offset-semantics safety: Qdrant's scroll offset may be INCLUSIVE of the
// boundary point ID (re-emitting it on the next page). We dedup by ID and stop
// as soon as a page yields no new unique IDs, so pagination is correct and
// loop-free under EITHER inclusive- or exclusive-offset semantics.
func ListMemories(ctx context.Context, store Store, opts ListMemoriesOptions) ([]Memory, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultListLimit
	}

	scrollOpts := BuildTimeWindowScrollOptions(opts)
	// Decouple the per-page fetch size from the result limit (see scrollPageSize).
	// The caller's limit is applied ONLY to the final truncate below.
	scrollOpts.Limit = scrollPageSize

	var all []Memory
	seen := make(map[string]struct{})
	offset := ""
	for {
		pageOpts := scrollOpts
		pageOpts.Offset = offset
		page, next, err := store.Scroll(ctx, pageOpts)
		if err != nil {
			return nil, err
		}

		newInPage := 0
		for _, m := range page {
			if _, dup := seen[m.ID]; dup {
				continue
			}
			seen[m.ID] = struct{}{}
			all = append(all, m)
			newInPage++
		}

		// Stop on: exhausted cursor, OR a page that added no new unique IDs
		// (guards against an infinite loop under inclusive-offset semantics
		// where the boundary point is re-emitted forever).
		if next == "" || newInPage == 0 {
			break
		}
		offset = next
	}

	// Deterministic order: stable sort by created_at with ID as a secondary key
	// so equal timestamps (float64 seconds collide often) order reproducibly.
	sort.SliceStable(all, func(i, j int) bool {
		if all[i].CreatedAt != all[j].CreatedAt {
			return all[i].CreatedAt < all[j].CreatedAt
		}
		return all[i].ID < all[j].ID
	})

	// Recent Memory view: keep the NEWEST `limit` entries (tail of the ascending
	// slice); the returned slice stays ascending for chronological display.
	if len(all) > limit {
		all = all[len(all)-limit:]
	}
	return all, nil
}
