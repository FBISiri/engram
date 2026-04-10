package memory

import "context"

// Store is the abstract interface for vector storage backends.
// Implementations must be safe for concurrent use.
type Store interface {
	// Insert stores a memory with its embedding vector.
	Insert(ctx context.Context, mem *Memory, vector []float32) error

	// Search returns scored memories matching the query vector.
	// Results are ordered by raw cosine similarity (caller applies final scoring).
	Search(ctx context.Context, vector []float32, opts SearchOptions) ([]ScoredMemory, error)

	// Scroll returns memories matching filters without requiring a query vector.
	// Results are ordered by creation time (newest first). Use offset for pagination.
	Scroll(ctx context.Context, opts ScrollOptions) ([]Memory, string, error)

	// Delete removes memories by IDs. Returns the number of successfully deleted items.
	Delete(ctx context.Context, ids []string) (int, error)

	// Update modifies payload fields of an existing memory without re-embedding.
	Update(ctx context.Context, id string, fields map[string]any) error

	// SearchByIDs retrieves specific memories by their IDs.
	SearchByIDs(ctx context.Context, ids []string) ([]Memory, error)

	// EnsureCollection creates the collection and indexes if they don't exist.
	EnsureCollection(ctx context.Context) error

	// Stats returns collection statistics.
	Stats(ctx context.Context) (*CollectionStats, error)

	// DeleteExpired removes all memories whose valid_until > 0 AND valid_until < now.
	// Returns the number of deleted memories.
	DeleteExpired(ctx context.Context) (int, error)
}

// ScrollOptions configures a filter-based scroll query (no vector needed).
type ScrollOptions struct {
	Limit   int
	Filters []Filter
	Offset  string // opaque offset token from previous Scroll call; empty for first page
}

// SearchOptions configures a vector search query.
type SearchOptions struct {
	Limit   int
	Filters []Filter
}

// FilterOp is a filter operation.
type FilterOp string

const (
	OpEq    FilterOp = "eq"
	OpIn    FilterOp = "in"
	OpGte   FilterOp = "gte"
	OpLte   FilterOp = "lte"
	OpRange FilterOp = "range"
)

// Filter is a single field filter for search.
type Filter struct {
	Field string
	Op    FilterOp
	Value any // string, []string, float64, [2]float64 (for range)
}

// CollectionStats holds collection-level statistics.
type CollectionStats struct {
	PointCount   uint64
	VectorCount  uint64
	IndexedCount uint64
	SegmentCount uint64
	Status       string
}
