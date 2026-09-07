package server

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// evapRecordingStore is a minimal Store that supports Scroll (with offset
// pagination) and records Update calls, for evaporation sweep tests.
type evapRecordingStore struct {
	mu      sync.Mutex
	mems    []memory.Memory
	updates map[string]map[string]any
}

func newEvapStore(mems ...memory.Memory) *evapRecordingStore {
	return &evapRecordingStore{mems: mems, updates: map[string]map[string]any{}}
}

func (s *evapRecordingStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	start := 0
	if opts.Offset != "" {
		start, _ = strconv.Atoi(opts.Offset)
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	end := start + limit
	if end >= len(s.mems) {
		return append([]memory.Memory(nil), s.mems[start:]...), "", nil
	}
	return append([]memory.Memory(nil), s.mems[start:end]...), strconv.Itoa(end), nil
}

func (s *evapRecordingStore) Update(_ context.Context, id string, fields map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.updates[id] = fields
	return nil
}

func (s *evapRecordingStore) updateCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.updates)
}

// Unused Store methods.
func (s *evapRecordingStore) Insert(context.Context, *memory.Memory, []float32) error { return nil }
func (s *evapRecordingStore) Search(context.Context, []float32, memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (s *evapRecordingStore) Delete(context.Context, []string) (int, error) { return 0, nil }
func (s *evapRecordingStore) SearchByIDs(context.Context, []string) ([]memory.Memory, error) {
	return nil, nil
}
func (s *evapRecordingStore) EnsureCollection(context.Context) error { return nil }
func (s *evapRecordingStore) Stats(context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{}, nil
}
func (s *evapRecordingStore) DeleteExpired(context.Context) (int, error) { return 0, nil }

func evapEvent(id string, importance, ageDays float64) memory.Memory {
	return memory.Memory{
		ID:              id,
		Type:            memory.TypeEvent,
		Importance:      importance,
		CreatedAt:       float64(time.Now().Unix()) - ageDays*86400.0,
		LifecycleStatus: memory.LifecycleActive,
	}
}

func enabledCfg() memory.EvaporationConfig {
	c := memory.DefaultEvaporationConfig()
	c.Enabled = true
	c.DryRun = false // exercise the mutating path
	return c
}

// liveCfg is the disabled-but-mutating base config used to seed s.evapCfg in
// tests that then hot-reload Enabled=true (the override inherits DryRun=false).
func liveCfg() memory.EvaporationConfig {
	c := memory.DefaultEvaporationConfig()
	c.DryRun = false
	return c
}

// TestEvaporationSweep_Deprecation: age 90d event drops below threshold and is
// deprecated; age 0d and 30d events are left untouched.
func TestEvaporationSweep_Deprecation(t *testing.T) {
	store := newEvapStore(
		evapEvent("fresh", 3, 0),
		evapEvent("mid", 3, 30),
		evapEvent("old", 3, 90),
	)
	dep, skipped, err := runEvaporationSweep(context.Background(), store, enabledCfg(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if dep != 1 || skipped != 0 {
		t.Fatalf("deprecated=%d skipped=%d, want 1/0", dep, skipped)
	}
	fields, ok := store.updates["old"]
	if !ok {
		t.Fatal("expected 'old' to be deprecated")
	}
	if fields["lifecycle_status"] != memory.LifecycleDeprecated {
		t.Errorf("lifecycle_status = %v, want deprecated", fields["lifecycle_status"])
	}
	meta, _ := fields["metadata"].(map[string]any)
	if meta == nil || meta["deprecated_reason"] != "evaporation" {
		t.Errorf("metadata deprecated_reason missing: %v", fields["metadata"])
	}
	if _, ok := store.updates["fresh"]; ok {
		t.Error("'fresh' should not be deprecated")
	}
	if _, ok := store.updates["mid"]; ok {
		t.Error("'mid' should not be deprecated")
	}
}

// TestEvaporationSweep_BatchLimit: with 250 evaporated candidates, at most
// SweepBatchLimit (100) are deprecated in one sweep.
func TestEvaporationSweep_BatchLimit(t *testing.T) {
	var mems []memory.Memory
	for i := 0; i < 250; i++ {
		mems = append(mems, evapEvent("m"+strconv.Itoa(i), 3, 200))
	}
	store := newEvapStore(mems...)
	dep, _, err := runEvaporationSweep(context.Background(), store, enabledCfg(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if dep != 100 {
		t.Fatalf("deprecated=%d, want 100 (batch limit)", dep)
	}
	if store.updateCount() != 100 {
		t.Fatalf("update count=%d, want 100", store.updateCount())
	}
}

// TestEvaporationSweepTick_HonorsRuntimeConfig proves the split-brain fix: a
// tick reads the server's CURRENT effective config, so a memory_apply_config
// hot-reload that enables evaporation makes the sweep start deprecating without
// a restart.
func TestEvaporationSweepTick_HonorsRuntimeConfig(t *testing.T) {
	store := newEvapStore(evapEvent("old", 3, 90))
	s := &Server{store: store, evapCfg: liveCfg()} // disabled, dry_run off

	// Disabled: tick is a no-op.
	if dep, _ := s.evaporationSweepTick(context.Background(), time.Now()); dep != 0 {
		t.Fatalf("disabled tick deprecated=%d, want 0", dep)
	}
	if store.updateCount() != 0 {
		t.Fatalf("disabled tick updated %d, want 0", store.updateCount())
	}

	// Hot-reload enable via the same accessor the query path uses.
	en := true
	s.overrides.mu.Lock()
	s.applyEvaporationOverride(&EvaporationConfigInput{Enabled: &en})
	s.overrides.mu.Unlock()

	if dep, _ := s.evaporationSweepTick(context.Background(), time.Now()); dep != 1 {
		t.Fatalf("enabled tick deprecated=%d, want 1", dep)
	}
}

// TestEvaporationSweep_RecordsMetrics verifies runEvaporationSweep records into
// a real *Metrics handle (R8).
func TestEvaporationSweep_RecordsMetrics(t *testing.T) {
	store := newEvapStore(evapEvent("old", 3, 90), evapEvent("fresh", 3, 0))
	m := metrics.New(nil, nil)
	dep, _, err := runEvaporationSweep(context.Background(), store, enabledCfg(), m)
	if err != nil {
		t.Fatal(err)
	}
	if dep != 1 {
		t.Fatalf("deprecated=%d, want 1", dep)
	}
	if got := counterValue(t, m.EvaporationSweepTotal); got != 1 {
		t.Errorf("EvaporationSweepTotal=%v, want 1", got)
	}
	if got := counterValue(t, m.EvaporationDeprecatedTotal.WithLabelValues("event")); got != 1 {
		t.Errorf("EvaporationDeprecatedTotal{event}=%v, want 1", got)
	}
}

// TestSweepIntervalHonorsHotReload proves sweepInterval() (which the ticker-
// reset path reads each tick) reflects a hot-reloaded SweepIntervalH.
func TestSweepIntervalHonorsHotReload(t *testing.T) {
	s := &Server{store: newEvapStore(), evapCfg: memory.DefaultEvaporationConfig()}
	if got := s.sweepInterval(); got != 6*time.Hour {
		t.Fatalf("default interval=%s, want 6h", got)
	}
	v := 12
	s.overrides.mu.Lock()
	s.applyEvaporationOverride(&EvaporationConfigInput{SweepIntervalH: &v})
	s.overrides.mu.Unlock()
	if got := s.sweepInterval(); got != 12*time.Hour {
		t.Fatalf("reloaded interval=%s, want 12h", got)
	}
	z := 0
	s.overrides.mu.Lock()
	s.applyEvaporationOverride(&EvaporationConfigInput{SweepIntervalH: &z})
	s.overrides.mu.Unlock()
	if got := s.sweepInterval(); got != 6*time.Hour {
		t.Fatalf("zero interval=%s, want 6h default", got)
	}
}

// TestSearchObservesEffectiveImportanceMetric proves the query-path samples
// engram_evaporation_effective_importance (spec R8) per returned result.
func TestSearchObservesEffectiveImportanceMetric(t *testing.T) {
	srv, store := newTestServer()
	srv.evapCfg = enabledCfg()
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	ctx := context.Background()
	vec, _ := srv.embedder.Embed(ctx, "hello evaporation world")
	mem := memory.New("hello evaporation world", memory.WithType(memory.TypeEvent), memory.WithImportance(5))
	if err := store.Insert(ctx, mem, vec); err != nil {
		t.Fatal(err)
	}

	body, _ := json.Marshal(map[string]any{"query": "hello evaporation world", "limit": 5})
	resp, err := http.Post(ts.URL+"/memories/search", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d, want 200", resp.StatusCode)
	}

	if got := histogramCount(t, srv.metrics.EvaporationEffectiveImportance.WithLabelValues("event")); got < 1 {
		t.Errorf("effective_importance histogram sample count=%d, want >=1", got)
	}
}

// histogramCount reads the sample count of a prometheus histogram observer.
func histogramCount(t *testing.T, o prometheus.Observer) uint64 {
	t.Helper()
	m, ok := o.(prometheus.Metric)
	if !ok {
		t.Fatal("observer is not a prometheus.Metric")
	}
	var dm dto.Metric
	if err := m.Write(&dm); err != nil {
		t.Fatal(err)
	}
	return dm.GetHistogram().GetSampleCount()
}

// counterValue reads the current value of a prometheus counter metric.
func counterValue(t *testing.T, m prometheus.Metric) float64 {
	t.Helper()
	var dm dto.Metric
	if err := m.Write(&dm); err != nil {
		t.Fatal(err)
	}
	return dm.GetCounter().GetValue()
}

// TestPutRevivesEvaporationDeprecated proves spec §5.5/§7.2: PUT-updating a
// memory that was deprecated by evaporation, raising its importance above the
// eviction threshold, restores lifecycle_status to active and clears the
// deprecated_* metadata.
func TestPutRevivesEvaporationDeprecated(t *testing.T) {
	srv, store := newTestServer()
	srv.evapCfg = enabledCfg()
	h := NewHTTPServer(srv, 0, "")
	ts := httptest.NewServer(h.Handler())
	defer ts.Close()

	now := float64(time.Now().Unix())
	mem := memory.Memory{
		ID:              "rev1",
		Type:            memory.TypeEvent,
		Content:         "stale event",
		Source:          "agent",
		Importance:      2,
		CreatedAt:       now,
		UpdatedAt:       now,
		LifecycleStatus: memory.LifecycleDeprecated,
		Metadata: map[string]any{
			"deprecated_reason":                   "evaporation",
			"deprecated_at":                       now,
			"effective_importance_at_deprecation": 0.4,
			"source_type":                         "tool_output",
		},
	}
	if err := store.Insert(context.Background(), &mem, make([]float32, 8)); err != nil {
		t.Fatal(err)
	}

	body, _ := json.Marshal(map[string]any{
		"content":    "revived event",
		"type":       "event",
		"importance": 9,
		"metadata":   map[string]any{"source_type": "tool_output"},
	})
	req, _ := http.NewRequest(http.MethodPut, ts.URL+"/memories/rev1", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d, want 200", resp.StatusCode)
	}

	var out memory.Memory
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.LifecycleStatus != memory.LifecycleActive {
		t.Errorf("lifecycle_status=%q, want active", out.LifecycleStatus)
	}
	if _, ok := out.Metadata["deprecated_reason"]; ok {
		t.Errorf("deprecated_reason should be cleared, got %v", out.Metadata["deprecated_reason"])
	}
}

// TestEvaporationSweep_DryRunZeroUpdates proves spec §4.5/§6.6: with dry_run=true
// the sweep identifies candidates and emits candidate metrics but performs ZERO
// store.Update calls.
func TestEvaporationSweep_DryRunZeroUpdates(t *testing.T) {
	store := newEvapStore(
		evapEvent("a", 3, 200),
		evapEvent("b", 3, 200),
		evapEvent("c", 3, 200),
	)
	cfg := memory.DefaultEvaporationConfig()
	cfg.Enabled = true
	cfg.DryRun = true
	m := metrics.New(nil, nil)

	dep, skipped, err := runEvaporationSweep(context.Background(), store, cfg, m)
	if err != nil {
		t.Fatal(err)
	}
	if dep != 0 || skipped != 0 {
		t.Fatalf("dry-run deprecated=%d skipped=%d, want 0/0", dep, skipped)
	}
	if store.updateCount() != 0 {
		t.Fatalf("dry-run performed %d store.Update calls, want 0", store.updateCount())
	}
	if got := counterValue(t, m.EvaporationDryRunCandidatesTotal.WithLabelValues("event")); got != 3 {
		t.Errorf("dry_run_candidates{event}=%v, want 3", got)
	}
	if got := counterValue(t, m.EvaporationScannedTotal); got != 3 {
		t.Errorf("scanned_total=%v, want 3", got)
	}
}

// TestEvaporationSweep_P1Structural is the adversarial regression (spec §6.3):
// even with HALF_LIFE_IDENTITY=1 and HALF_LIFE_DIRECTIVE=1, a 10-year-old
// identity/directive memory is exempted by P1 and never deprecated.
func TestEvaporationSweep_P1Structural(t *testing.T) {
	mk := func(id string, ty memory.MemoryType) memory.Memory {
		return memory.Memory{
			ID:              id,
			Type:            ty,
			Importance:      3,
			CreatedAt:       float64(time.Now().Unix()) - 3650*86400.0, // 10 years
			LifecycleStatus: memory.LifecycleActive,
		}
	}
	store := newEvapStore(mk("id1", memory.TypeIdentity), mk("dir1", memory.TypeDirective))
	cfg := memory.DefaultEvaporationConfig()
	cfg.Enabled = true
	cfg.DryRun = false
	cfg.HalfLifeDays[memory.TypeIdentity] = 1  // adversarial typo
	cfg.HalfLifeDays[memory.TypeDirective] = 1 // adversarial typo

	dep, _, err := runEvaporationSweep(context.Background(), store, cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if dep != 0 {
		t.Fatalf("deprecated=%d, want 0 (P1 must be structural)", dep)
	}
	if store.updateCount() != 0 {
		t.Fatalf("identity/directive updated %d times, want 0", store.updateCount())
	}
}

// TestStartEvaporationSweep_DisabledNoop: disabled config performs no updates
// (the ticker is bounded at the boot interval, so nothing fires quickly).
func TestStartEvaporationSweep_DisabledNoop(t *testing.T) {
	store := newEvapStore(evapEvent("old", 3, 500))
	s := &Server{store: store, evapCfg: memory.DefaultEvaporationConfig()} // Enabled=false
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s.StartEvaporationSweep(ctx)
	time.Sleep(30 * time.Millisecond)
	if store.updateCount() != 0 {
		t.Errorf("disabled sweep updated %d memories, want 0", store.updateCount())
	}
}
