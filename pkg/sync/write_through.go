package sync

import (
	"context"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// writeThroughStore implements WriteThroughStore.
type writeThroughStore struct {
	store memory.Store
	log   CommitLog
	cfg   RingBufferConfig

	mu          sync.Mutex
	ring        []CommitEntry
	totalWrites int64
	lastFlush   time.Time
	pending     int // entries in ring not yet flushed

	stopOnce sync.Once
	stopCh   chan struct{}
}

// NewWriteThrough creates a WriteThroughStore backed by a memory.Store and BoltDB commit log.
func NewWriteThrough(store memory.Store, boltPath string, cfg RingBufferConfig) (WriteThroughStore, CommitLog, error) {
	log, err := NewBoltCommitLog(boltPath)
	if err != nil {
		return nil, nil, err
	}
	return NewWriteThroughWithLog(store, log, cfg), log, nil
}

// NewWriteThroughWithLog creates a WriteThroughStore with a provided CommitLog (useful for testing).
func NewWriteThroughWithLog(store memory.Store, log CommitLog, cfg RingBufferConfig) WriteThroughStore {
	if cfg.Capacity <= 0 {
		cfg.Capacity = 200
	}
	if cfg.FlushInterval <= 0 {
		cfg.FlushInterval = 5 * time.Minute
	}
	wt := &writeThroughStore{
		store:     store,
		log:       log,
		cfg:       cfg,
		ring:      make([]CommitEntry, 0, cfg.Capacity),
		lastFlush: time.Now(),
		stopCh:    make(chan struct{}),
	}
	go wt.autoFlushLoop()
	return wt
}

func (w *writeThroughStore) autoFlushLoop() {
	ticker := time.NewTicker(w.cfg.FlushInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			w.Flush(context.Background())
		case <-w.stopCh:
			return
		}
	}
}

func (w *writeThroughStore) appendToRing(entry CommitEntry) {
	if len(w.ring) >= w.cfg.Capacity {
		// Evict oldest (ring semantics: shift left)
		copy(w.ring, w.ring[1:])
		w.ring = w.ring[:w.cfg.Capacity-1]
	}
	w.ring = append(w.ring, entry)
	w.pending++
	w.totalWrites++
}

// Insert writes to commit log first, then to the underlying store.
func (w *writeThroughStore) Insert(ctx context.Context, mem *memory.Memory, vector []float32) error {
	entry := CommitEntry{
		ID:        mem.ID,
		Op:        OpInsert,
		Memory:    mem,
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
	}

	if err := w.log.Append(ctx, entry); err != nil {
		return err
	}

	if err := w.store.Insert(ctx, mem, vector); err != nil {
		// Best-effort: log succeeded but store failed — entry is in commit log for replay
		return err
	}

	w.mu.Lock()
	w.appendToRing(entry)
	w.mu.Unlock()
	return nil
}

func (w *writeThroughStore) Delete(ctx context.Context, ids []string) (int, error) {
	entries := make([]CommitEntry, len(ids))
	ts := float64(time.Now().UnixNano()) / 1e9
	for i, id := range ids {
		entries[i] = CommitEntry{
			ID:        id,
			Op:        OpDelete,
			Timestamp: ts,
		}
	}

	for _, e := range entries {
		if err := w.log.Append(ctx, e); err != nil {
			return 0, err
		}
	}

	count, err := w.store.Delete(ctx, ids)
	if err != nil {
		return 0, err
	}

	w.mu.Lock()
	for _, e := range entries {
		w.appendToRing(e)
	}
	w.mu.Unlock()
	return count, nil
}

func (w *writeThroughStore) Update(ctx context.Context, id string, fields map[string]any) error {
	entry := CommitEntry{
		ID:        id,
		Op:        OpUpdate,
		Fields:    fields,
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
	}

	if err := w.log.Append(ctx, entry); err != nil {
		return err
	}

	if err := w.store.Update(ctx, id, fields); err != nil {
		return err
	}

	w.mu.Lock()
	w.appendToRing(entry)
	w.mu.Unlock()
	return nil
}

// Pass-through read operations directly to the underlying store.
func (w *writeThroughStore) Search(ctx context.Context, vector []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return w.store.Search(ctx, vector, opts)
}

func (w *writeThroughStore) Scroll(ctx context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	return w.store.Scroll(ctx, opts)
}

func (w *writeThroughStore) SearchByIDs(ctx context.Context, ids []string) ([]memory.Memory, error) {
	return w.store.SearchByIDs(ctx, ids)
}

func (w *writeThroughStore) EnsureCollection(ctx context.Context) error {
	return w.store.EnsureCollection(ctx)
}

func (w *writeThroughStore) Stats(ctx context.Context) (*memory.CollectionStats, error) {
	return w.store.Stats(ctx)
}

// DeleteExpired delegates to the underlying store.
// Expiry cleanup is a bulk background operation; individual IDs are not written
// to the commit log (they were already logged on original insert/update).
func (w *writeThroughStore) DeleteExpired(ctx context.Context) (int, error) {
	return w.store.DeleteExpired(ctx)
}

// Replay returns commit entries since the given timestamp from ring buffer or commit log.
func (w *writeThroughStore) Replay(ctx context.Context, since float64) ([]CommitEntry, error) {
	w.mu.Lock()
	// Check if ring buffer covers the requested range
	if len(w.ring) > 0 && w.ring[0].Timestamp >= since {
		// Ring buffer has all entries we need
		result := make([]CommitEntry, 0)
		for _, e := range w.ring {
			if e.Timestamp >= since {
				result = append(result, e)
			}
		}
		w.mu.Unlock()
		return result, nil
	}
	w.mu.Unlock()

	// Fall back to commit log
	return w.log.ReadSince(ctx, since)
}

// Flush persists ring buffer entries to the commit log.
func (w *writeThroughStore) Flush(ctx context.Context) error {
	w.mu.Lock()
	// Entries are already in the commit log (written on each operation),
	// so flush just resets the pending counter.
	w.pending = 0
	w.lastFlush = time.Now()
	w.mu.Unlock()
	return nil
}

// Close stops the auto-flush goroutine and closes the commit log.
func (w *writeThroughStore) Close() error {
	w.stopOnce.Do(func() { close(w.stopCh) })
	return w.log.Close()
}

// SyncStats returns current statistics.
func (w *writeThroughStore) SyncStats() SyncStats {
	w.mu.Lock()
	defer w.mu.Unlock()
	return SyncStats{
		TotalWrites:     w.totalWrites,
		BufferedEntries: len(w.ring),
		LastFlushAt:     w.lastFlush,
		PendingFlush:    w.pending,
	}
}
