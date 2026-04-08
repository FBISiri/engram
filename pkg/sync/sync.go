// Package sync provides write-through durability and ring-buffer synchronization
// for concurrent Engram sessions. When multiple agent sessions write memories
// simultaneously, this package ensures no writes are lost between Qdrant
// (vector store) and a local BoltDB commit log.
//
// Architecture:
//
//	Session A ──┐
//	Session B ──┼──► WriteThrough ──► Qdrant (vector store)
//	Session C ──┘         │
//	                       └──► BoltDB (commit log)
//	                             ▲
//	                    RingBuffer (in-memory, 200 entries)
//	                    flush every 5 minutes
//
// Usage:
//
//	store := sync.NewWriteThrough(qdrantStore, boltPath)
//	// store implements memory.Store, use as drop-in replacement
package sync

import (
	"context"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// CommitEntry represents a single write operation persisted to the commit log.
type CommitEntry struct {
	ID        string         `json:"id"`
	Op        CommitOp       `json:"op"`   // "insert" | "delete" | "update"
	Memory    *memory.Memory `json:"mem,omitempty"`
	Fields    map[string]any `json:"fields,omitempty"` // for update ops
	Timestamp float64        `json:"ts"`
}

// CommitOp is the type of write operation.
type CommitOp string

const (
	OpInsert CommitOp = "insert"
	OpDelete CommitOp = "delete"
	OpUpdate CommitOp = "update"
)

// CommitLog is the durable write-ahead log backend.
// Implementations must be safe for concurrent use.
type CommitLog interface {
	// Append durably persists a commit entry.
	Append(ctx context.Context, entry CommitEntry) error

	// ReadSince returns all entries after the given timestamp (for recovery).
	ReadSince(ctx context.Context, since float64) ([]CommitEntry, error)

	// Truncate removes entries older than the given timestamp.
	Truncate(ctx context.Context, before float64) error

	// Close releases resources.
	Close() error
}

// RingBufferConfig configures the in-memory ring buffer.
type RingBufferConfig struct {
	// Capacity is the maximum number of entries held in memory before flush.
	// Default: 200.
	Capacity int

	// FlushInterval is how often the buffer is persisted to the CommitLog.
	// Default: 5 minutes.
	FlushInterval time.Duration
}

// DefaultRingBufferConfig returns sensible defaults (200 entries, 5min flush).
func DefaultRingBufferConfig() RingBufferConfig {
	return RingBufferConfig{
		Capacity:      200,
		FlushInterval: 5 * time.Minute,
	}
}

// WriteThroughStore wraps a memory.Store with synchronous commit-log durability.
// Every Insert, Delete, and Update call is first written to the CommitLog, then
// forwarded to the underlying store. Reads are served directly from the store.
//
// The ring buffer holds the most recent entries in memory to serve fast replay
// requests (e.g., session startup sync) without hitting BoltDB.
type WriteThroughStore interface {
	memory.Store

	// Replay returns all commit entries since the given timestamp.
	// Used by sessions starting up to catch up on concurrent writes.
	Replay(ctx context.Context, since float64) ([]CommitEntry, error)

	// Flush forces the ring buffer to persist to the CommitLog immediately.
	Flush(ctx context.Context) error

	// Stats returns sync-specific statistics.
	SyncStats() SyncStats
}

// SyncStats holds runtime statistics for the write-through layer.
type SyncStats struct {
	TotalWrites     int64     // Total write ops processed
	BufferedEntries int       // Current ring buffer size
	LastFlushAt     time.Time // When the buffer was last flushed
	PendingFlush    int       // Entries not yet persisted to CommitLog
}
