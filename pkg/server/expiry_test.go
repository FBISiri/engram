package server_test

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/server"
)

// fakeStore is a minimal in-memory Store implementation for testing expiry cleanup.
type fakeStore struct {
	deleteExpiredCalls atomic.Int32
	deleteExpiredErr   error
	deleteExpiredN     int
}

func (f *fakeStore) DeleteExpired(_ context.Context) (int, error) {
	f.deleteExpiredCalls.Add(1)
	return f.deleteExpiredN, f.deleteExpiredErr
}

// Unused Store methods — stubbed to satisfy the interface.
func (f *fakeStore) Insert(_ context.Context, _ *memory.Memory, _ []float32) error { return nil }
func (f *fakeStore) Search(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (f *fakeStore) Scroll(_ context.Context, _ memory.ScrollOptions) ([]memory.Memory, string, error) {
	return nil, "", nil
}
func (f *fakeStore) Delete(_ context.Context, _ []string) (int, error) { return 0, nil }
func (f *fakeStore) Update(_ context.Context, _ string, _ map[string]any) error {
	return nil
}
func (f *fakeStore) SearchByIDs(_ context.Context, _ []string) ([]memory.Memory, error) {
	return nil, nil
}
func (f *fakeStore) EnsureCollection(_ context.Context) error { return nil }
func (f *fakeStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{}, nil
}

// TestStartExpiryCleanup_Ticks verifies that the cleanup goroutine calls
// DeleteExpired at least twice within a short interval and stops on ctx cancel.
func TestStartExpiryCleanup_Ticks(t *testing.T) {
	t.Parallel()

	store := &fakeStore{deleteExpiredN: 3}
	ctx, cancel := context.WithCancel(context.Background())

	const interval = 50 * time.Millisecond
	server.StartExpiryCleanup(ctx, store, interval)

	// Wait long enough for at least 2 ticks.
	time.Sleep(130 * time.Millisecond)
	cancel()

	// Give the goroutine a moment to observe cancellation.
	time.Sleep(20 * time.Millisecond)

	calls := store.deleteExpiredCalls.Load()
	if calls < 2 {
		t.Errorf("expected at least 2 DeleteExpired calls, got %d", calls)
	}
}

// TestStartExpiryCleanup_StopsOnCancel verifies that after cancel,
// no further DeleteExpired calls are made.
func TestStartExpiryCleanup_StopsOnCancel(t *testing.T) {
	t.Parallel()

	store := &fakeStore{}
	ctx, cancel := context.WithCancel(context.Background())

	const interval = 30 * time.Millisecond
	server.StartExpiryCleanup(ctx, store, interval)

	// Let it tick at least once.
	time.Sleep(50 * time.Millisecond)
	cancel()
	time.Sleep(20 * time.Millisecond)

	before := store.deleteExpiredCalls.Load()

	// Wait another full interval; no new calls should arrive.
	time.Sleep(60 * time.Millisecond)
	after := store.deleteExpiredCalls.Load()

	if after != before {
		t.Errorf("DeleteExpired was called %d times after cancel (expected 0 new calls)", after-before)
	}
}

// TestStartExpiryCleanup_DefaultInterval verifies that passing interval=0
// uses the default (10 min) without panicking.
func TestStartExpiryCleanup_DefaultInterval(t *testing.T) {
	t.Parallel()

	store := &fakeStore{}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Should not panic and no ticks expected within 50ms for a 10-min interval.
	server.StartExpiryCleanup(ctx, store, 0)
	time.Sleep(50 * time.Millisecond)

	if calls := store.deleteExpiredCalls.Load(); calls != 0 {
		t.Errorf("expected 0 calls within 50ms with 10-min interval, got %d", calls)
	}
}
