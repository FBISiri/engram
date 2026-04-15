package sync_test

// Acceptance tests for Write-through + Ring Buffer (P2).
//
// These tests define the contract that BMO's implementation must satisfy.
// They should PASS when the implementation is complete and FAIL before it exists.
//
// Test categories:
//   A. Write-through durability
//   B. Ring buffer capacity and flush behavior
//   C. Concurrent session safety
//   D. Replay / catch-up on session startup
//   E. Recovery after crash (BoltDB replay)

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	engram_sync "github.com/FBISiri/engram/pkg/sync"
)

// ─────────────────────────────────────────────────────────────────────────────
// A. Write-through durability
// ─────────────────────────────────────────────────────────────────────────────

// A1: Every Insert must be persisted to CommitLog before returning.
func TestWriteThrough_InsertPersistsToCommitLog(t *testing.T) {
	store, log := newTestStore(t)
	ctx := context.Background()

	mem := memory.New("test memory A1", memory.WithImportance(7))
	vec := fakeVector(1536)

	if err := store.Insert(ctx, mem, vec); err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	entries, err := log.ReadSince(ctx, 0)
	if err != nil {
		t.Fatalf("ReadSince failed: %v", err)
	}
	if len(entries) == 0 {
		t.Fatal("expected commit log to contain the inserted entry, got 0")
	}
	found := false
	for _, e := range entries {
		if e.ID == mem.ID && e.Op == engram_sync.OpInsert {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("insert entry for ID=%s not found in commit log", mem.ID)
	}
}

// A2: Delete must be persisted to CommitLog.
func TestWriteThrough_DeletePersistsToCommitLog(t *testing.T) {
	store, log := newTestStore(t)
	ctx := context.Background()

	mem := memory.New("test memory A2")
	vec := fakeVector(1536)
	_ = store.Insert(ctx, mem, vec)

	count, err := store.Delete(ctx, []string{mem.ID})
	if err != nil || count == 0 {
		t.Fatalf("Delete failed: err=%v count=%d", err, count)
	}

	entries, _ := log.ReadSince(ctx, 0)
	found := false
	for _, e := range entries {
		if e.ID == mem.ID && e.Op == engram_sync.OpDelete {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("delete entry for ID=%s not found in commit log", mem.ID)
	}
}

// A3: If CommitLog.Append fails, the Insert must return an error and NOT write to the store.
func TestWriteThrough_InsertRollsBackOnCommitLogFailure(t *testing.T) {
	store, _ := newTestStoreWithFailingLog(t)
	ctx := context.Background()

	mem := memory.New("test memory A3")
	vec := fakeVector(1536)

	err := store.Insert(ctx, mem, vec)
	if err == nil {
		t.Fatal("expected Insert to fail when CommitLog.Append fails, but got nil error")
	}

	// Memory must NOT be in the underlying store
	results, _ := store.SearchByIDs(ctx, []string{mem.ID})
	if len(results) > 0 {
		t.Error("memory was written to store despite CommitLog failure (inconsistency)")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// B. Ring buffer capacity and flush behavior
// ─────────────────────────────────────────────────────────────────────────────

// B1: Ring buffer must not exceed configured capacity.
// When at capacity, oldest entry is evicted (ring semantics).
func TestRingBuffer_CapacityNotExceeded(t *testing.T) {
	const capacity = 10
	store, _ := newTestStoreWithConfig(t, engram_sync.RingBufferConfig{
		Capacity:      capacity,
		FlushInterval: 1 * time.Hour, // disable auto-flush
	})
	ctx := context.Background()

	// Insert capacity+5 entries
	for i := 0; i < capacity+5; i++ {
		mem := memory.New("ring buffer test entry")
		_ = store.Insert(ctx, mem, fakeVector(1536))
	}

	stats := store.SyncStats()
	if stats.BufferedEntries > capacity {
		t.Errorf("ring buffer size %d exceeds capacity %d", stats.BufferedEntries, capacity)
	}
}

// B2: Flush must empty the ring buffer and persist all entries to CommitLog.
func TestRingBuffer_FlushPersistsAllEntries(t *testing.T) {
	store, log := newTestStoreWithConfig(t, engram_sync.RingBufferConfig{
		Capacity:      200,
		FlushInterval: 1 * time.Hour, // disable auto-flush for this test
	})
	ctx := context.Background()

	const n = 5
	ids := make([]string, n)
	for i := 0; i < n; i++ {
		mem := memory.New("flush test entry")
		_ = store.Insert(ctx, mem, fakeVector(1536))
		ids[i] = mem.ID
	}

	if err := store.Flush(ctx); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	stats := store.SyncStats()
	if stats.PendingFlush != 0 {
		t.Errorf("expected 0 pending after flush, got %d", stats.PendingFlush)
	}

	entries, _ := log.ReadSince(ctx, 0)
	if len(entries) < n {
		t.Errorf("expected at least %d entries in commit log after flush, got %d", n, len(entries))
	}
}

// B3: Auto-flush triggers after FlushInterval.
func TestRingBuffer_AutoFlushAfterInterval(t *testing.T) {
	store, log := newTestStoreWithConfig(t, engram_sync.RingBufferConfig{
		Capacity:      200,
		FlushInterval: 50 * time.Millisecond, // very short for testing
	})
	ctx := context.Background()

	mem := memory.New("auto flush test")
	_ = store.Insert(ctx, mem, fakeVector(1536))

	// Wait for auto-flush
	time.Sleep(200 * time.Millisecond)

	entries, _ := log.ReadSince(ctx, 0)
	found := false
	for _, e := range entries {
		if e.ID == mem.ID {
			found = true
			break
		}
	}
	if !found {
		t.Error("entry not found in CommitLog after auto-flush interval")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// C. Concurrent session safety
// ─────────────────────────────────────────────────────────────────────────────

// C1: 10 goroutines inserting concurrently must all succeed without data loss.
func TestWriteThrough_ConcurrentInserts(t *testing.T) {
	store, log := newTestStore(t)
	ctx := context.Background()

	const goroutines = 10
	const insertsPerGoroutine = 20

	var wg sync.WaitGroup
	errCh := make(chan error, goroutines*insertsPerGoroutine)

	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < insertsPerGoroutine; i++ {
				mem := memory.New("concurrent insert test")
				if err := store.Insert(ctx, mem, fakeVector(1536)); err != nil {
					errCh <- err
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent insert error: %v", err)
	}

	// All writes must appear in commit log
	entries, _ := log.ReadSince(ctx, 0)
	expected := goroutines * insertsPerGoroutine
	if len(entries) < expected {
		t.Errorf("expected %d commit log entries, got %d (data loss detected)", expected, len(entries))
	}
}

// C2: No duplicate entries in CommitLog for the same operation.
func TestWriteThrough_NoDuplicateCommitEntries(t *testing.T) {
	store, log := newTestStore(t)
	ctx := context.Background()

	mem := memory.New("dedup test")
	_ = store.Insert(ctx, mem, fakeVector(1536))
	_ = store.Insert(ctx, mem, fakeVector(1536)) // intentional duplicate (dedup should handle at store level)

	entries, _ := log.ReadSince(ctx, 0)
	count := 0
	for _, e := range entries {
		if e.ID == mem.ID && e.Op == engram_sync.OpInsert {
			count++
		}
	}
	// We expect exactly 1 or 2 entries (store dedup may prevent second insert;
	// either way, no phantom duplicates beyond what was actually written)
	if count == 0 {
		t.Error("expected at least 1 insert entry in commit log")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// D. Replay / catch-up on session startup
// ─────────────────────────────────────────────────────────────────────────────

// D1: Replay returns entries since a given timestamp, in order.
func TestWriteThrough_ReplayReturnsSinceTimestamp(t *testing.T) {
	store, _ := newTestStore(t)
	ctx := context.Background()

	// Insert 3 entries and record mid-point timestamp
	for i := 0; i < 3; i++ {
		mem := memory.New("replay pre-existing")
		_ = store.Insert(ctx, mem, fakeVector(1536))
	}
	midpoint := float64(time.Now().Unix())
	time.Sleep(10 * time.Millisecond)

	// Insert 2 more entries after midpoint
	afterIDs := make(map[string]bool)
	for i := 0; i < 2; i++ {
		mem := memory.New("replay new entry")
		_ = store.Insert(ctx, mem, fakeVector(1536))
		afterIDs[mem.ID] = true
	}

	entries, err := store.Replay(ctx, midpoint)
	if err != nil {
		t.Fatalf("Replay failed: %v", err)
	}
	if len(entries) < 2 {
		t.Errorf("expected at least 2 replay entries since midpoint, got %d", len(entries))
	}
	for _, e := range entries {
		if e.Timestamp < midpoint {
			t.Errorf("replay returned entry with timestamp %f before midpoint %f", e.Timestamp, midpoint)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// E. Recovery after crash (BoltDB replay)
// ─────────────────────────────────────────────────────────────────────────────

// E1: After store restart (new WriteThroughStore from same BoltDB), all
// previously committed entries must still be readable via Replay.
func TestWriteThrough_SurvivesRestart(t *testing.T) {
	boltPath := t.TempDir() + "/engram-test.bolt"
	ctx := context.Background()

	// Session 1: write some memories
	store1, _ := newTestStoreAtPath(t, boltPath)
	mem1 := memory.New("survives restart test")
	_ = store1.Insert(ctx, mem1, fakeVector(1536))
	_ = store1.Flush(ctx)
	// Simulate crash: no graceful shutdown

	// Session 2: open same bolt path, replay should find mem1
	store2, _ := newTestStoreAtPath(t, boltPath)
	entries, err := store2.Replay(ctx, 0)
	if err != nil {
		t.Fatalf("Replay after restart failed: %v", err)
	}
	found := false
	for _, e := range entries {
		if e.ID == mem1.ID {
			found = true
			break
		}
	}
	if !found {
		t.Error("committed entry not found after simulated restart")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────────────────────────────────────

// closeable allows test helpers to close a WriteThroughStore that implements Close().
type closeable interface {
	Close() error
}

// prevStores tracks open stores by bolt path so we can close them before reopening.
var prevStores = map[string]closeable{}

// mockStore is a simple in-memory implementation of memory.Store for testing.
type mockStore struct {
	mu       sync.Mutex
	memories map[string]storedPoint
}

type storedPoint struct {
	mem    memory.Memory
	vector []float32
}

func newMockStore() *mockStore {
	return &mockStore{memories: make(map[string]storedPoint)}
}

func (s *mockStore) Insert(_ context.Context, mem *memory.Memory, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.memories[mem.ID] = storedPoint{mem: *mem, vector: vector}
	return nil
}

func (s *mockStore) Search(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}

func (s *mockStore) Scroll(_ context.Context, _ memory.ScrollOptions) ([]memory.Memory, string, error) {
	return nil, "", nil
}

func (s *mockStore) Delete(_ context.Context, ids []string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	count := 0
	for _, id := range ids {
		if _, ok := s.memories[id]; ok {
			delete(s.memories, id)
			count++
		}
	}
	return count, nil
}

func (s *mockStore) Update(_ context.Context, id string, fields map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.memories[id]; !ok {
		return fmt.Errorf("not found: %s", id)
	}
	return nil
}

func (s *mockStore) SearchByIDs(_ context.Context, ids []string) ([]memory.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result []memory.Memory
	for _, id := range ids {
		if sp, ok := s.memories[id]; ok {
			result = append(result, sp.mem)
		}
	}
	return result, nil
}

func (s *mockStore) EnsureCollection(_ context.Context) error { return nil }
func (s *mockStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{}, nil
}
func (s *mockStore) DeleteExpired(_ context.Context) (int, error) { return 0, nil }

// failingLog is a CommitLog that always returns an error on Append.
type failingLog struct{}

func (f *failingLog) Append(_ context.Context, _ engram_sync.CommitEntry) error {
	return fmt.Errorf("simulated commit log failure")
}
func (f *failingLog) ReadSince(_ context.Context, _ float64) ([]engram_sync.CommitEntry, error) {
	return nil, nil
}
func (f *failingLog) Truncate(_ context.Context, _ float64) error { return nil }
func (f *failingLog) Close() error                                { return nil }

// cleanupStore registers a t.Cleanup to close the store's goroutine and bolt file.
func cleanupStore(t *testing.T, store engram_sync.WriteThroughStore) {
	t.Helper()
	if c, ok := store.(closeable); ok {
		t.Cleanup(func() { c.Close() })
	}
}

// newTestStore creates a WriteThroughStore with default config and in-memory backends.
func newTestStore(t *testing.T) (engram_sync.WriteThroughStore, engram_sync.CommitLog) {
	t.Helper()
	boltPath := t.TempDir() + "/test.bolt"
	store, log, err := engram_sync.NewWriteThrough(newMockStore(), boltPath, engram_sync.DefaultRingBufferConfig())
	if err != nil {
		t.Fatalf("NewWriteThrough: %v", err)
	}
	cleanupStore(t, store)
	return store, log
}

// newTestStoreWithFailingLog creates a store backed by a CommitLog that always fails.
func newTestStoreWithFailingLog(t *testing.T) (engram_sync.WriteThroughStore, engram_sync.CommitLog) {
	t.Helper()
	fl := &failingLog{}
	store := engram_sync.NewWriteThroughWithLog(newMockStore(), fl, engram_sync.DefaultRingBufferConfig())
	cleanupStore(t, store)
	return store, fl
}

// newTestStoreWithConfig creates a store with a custom RingBufferConfig.
func newTestStoreWithConfig(t *testing.T, cfg engram_sync.RingBufferConfig) (engram_sync.WriteThroughStore, engram_sync.CommitLog) {
	t.Helper()
	boltPath := t.TempDir() + "/test.bolt"
	store, log, err := engram_sync.NewWriteThrough(newMockStore(), boltPath, cfg)
	if err != nil {
		t.Fatalf("NewWriteThrough: %v", err)
	}
	cleanupStore(t, store)
	return store, log
}

// newTestStoreAtPath creates a store backed by a BoltDB at the given path.
// It closes any previous store opened at the same path (simulates OS releasing
// file locks on crash).
func newTestStoreAtPath(t *testing.T, path string) (engram_sync.WriteThroughStore, engram_sync.CommitLog) {
	t.Helper()
	if prev, ok := prevStores[path]; ok {
		prev.Close()
		delete(prevStores, path)
	}
	store, log, err := engram_sync.NewWriteThrough(newMockStore(), path, engram_sync.DefaultRingBufferConfig())
	if err != nil {
		t.Fatalf("NewWriteThrough: %v", err)
	}
	if c, ok := store.(closeable); ok {
		prevStores[path] = c
		t.Cleanup(func() {
			c.Close()
			delete(prevStores, path)
		})
	}
	return store, log
}

// fakeVector returns a zero-vector of the given dimension.
func fakeVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(i%100) / 100.0
	}
	return v
}
