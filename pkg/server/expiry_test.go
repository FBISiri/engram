package server_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/server"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
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

// evapSeededStore embeds fakeStore and overrides Scroll to expose seeded
// evaporation-reason expired memories so scanExpiring can attribute them by
// type, while DeleteExpired reports a non-zero delete count (triggering the
// metrics feed in runExpiryTick).
type evapSeededStore struct {
	*fakeStore
	mems []memory.Memory
}

func (s *evapSeededStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	return append([]memory.Memory(nil), s.mems...), "", nil
}

// TestStartExpiryCleanup_FeedsMetrics (R2-A) starts the cleanup goroutine with
// a real *metrics.Metrics and a short interval, seeds evaporation-reason
// expired memories, and polls the hard-deleted counter with a deadline until it
// advances.
func TestStartExpiryCleanup_FeedsMetrics(t *testing.T) {
	t.Parallel()

	past := float64(time.Now().Add(-time.Hour).Unix())
	store := &evapSeededStore{
		fakeStore: &fakeStore{deleteExpiredN: 2},
		mems: []memory.Memory{
			{ID: "a", Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: past, Metadata: map[string]any{"deprecated_reason": "evaporation"}},
			{ID: "b", Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: past, Metadata: map[string]any{"deprecated_reason": "evaporation"}},
		},
	}

	m := metrics.New(nil, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server.StartExpiryCleanup(ctx, store, 20*time.Millisecond, m)

	deadline := time.Now().Add(1 * time.Second)
	for {
		if counterVal(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")) > 0 {
			return // counter advanced — success
		}
		if time.Now().After(deadline) {
			t.Fatalf("EvaporationHardDeletedTotal{event} did not advance within deadline")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestStartExpiryCleanup_NilMetricsNoPanic (R2-B) starts the goroutine with NO
// metrics arg on a store whose tick deletes something and asserts it ticks once
// without panicking.
func TestStartExpiryCleanup_NilMetricsNoPanic(t *testing.T) {
	t.Parallel()

	store := &fakeStore{deleteExpiredN: 3}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server.StartExpiryCleanup(ctx, store, 20*time.Millisecond, nil)

	deadline := time.Now().Add(1 * time.Second)
	for {
		if store.deleteExpiredCalls.Load() > 0 {
			return // ticked once without panicking — success
		}
		if time.Now().After(deadline) {
			t.Fatalf("DeleteExpired was not called within deadline")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestNewHTTPServerWiresMetrics (R2-C) verifies NewHTTPServer registers metrics
// on the Server (srv.Metrics() != nil) and that the SAME metrics object backs
// the /metrics endpoint: incrementing a counter via srv.Metrics() surfaces it
// in the /metrics output, proving pointer identity of the registry.
func TestNewHTTPServerWiresMetrics(t *testing.T) {
	t.Parallel()

	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
	}
	srv := server.NewServer(&fakeStore{}, nil, cfg)
	httpSrv := server.NewHTTPServer(srv, 0, "")

	if srv.Metrics() == nil {
		t.Fatal("srv.Metrics() is nil after NewHTTPServer")
	}

	// Feed a unique observation through the getter's metrics object.
	srv.Metrics().EvaporationHardDeletedTotal.WithLabelValues("event").Add(7)

	ts := httptest.NewServer(httpSrv.Handler())
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	got := string(body)

	if !strings.Contains(got, "engram_evaporation_hard_deleted_total") {
		t.Fatalf("/metrics does not serve srv.Metrics() registry; body missing counter:\n%s", got)
	}
	if !strings.Contains(got, `engram_evaporation_hard_deleted_total{type="event"} 7`) {
		t.Fatalf("/metrics missing the value fed via srv.Metrics() (registry not pointer-equal):\n%s", got)
	}
}

// counterVal reads the current value of a prometheus counter (external-package
// analogue of the internal counterValue helper).
func counterVal(t *testing.T, c prometheus.Counter) float64 {
	t.Helper()
	var m dto.Metric
	if err := c.Write(&m); err != nil {
		t.Fatalf("counter.Write: %v", err)
	}
	return m.GetCounter().GetValue()
}
