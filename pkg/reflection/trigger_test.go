package reflection

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

// ── in-package mock store ───────────────────────────────────────────────────

type triggerMockStore struct {
	mu       sync.Mutex
	memories map[string]memory.Memory
}

func newTriggerMockStore() *triggerMockStore {
	return &triggerMockStore{memories: make(map[string]memory.Memory)}
}

func (s *triggerMockStore) Insert(_ context.Context, mem *memory.Memory, _ []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.memories[mem.ID] = *mem
	return nil
}

func (s *triggerMockStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	limit := opts.Limit
	if limit == 0 {
		limit = 50
	}
	var results []memory.Memory
	for _, m := range s.memories {
		results = append(results, m)
		if len(results) >= limit {
			break
		}
	}
	return results, "", nil
}

func (s *triggerMockStore) Update(_ context.Context, id string, fields map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if m, ok := s.memories[id]; ok {
		if m.Metadata == nil {
			m.Metadata = make(map[string]any)
		}
		for k, v := range fields {
			m.Metadata[k] = v
		}
		s.memories[id] = m
	}
	return nil
}

// Stub implementations for the rest of the Store interface.
func (s *triggerMockStore) Search(_ context.Context, _ []float32, _ memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (s *triggerMockStore) Delete(_ context.Context, _ []string) (int, error)       { return 0, nil }
func (s *triggerMockStore) SearchByIDs(_ context.Context, _ []string) ([]memory.Memory, error) {
	return nil, nil
}
func (s *triggerMockStore) EnsureCollection(_ context.Context) error { return nil }
func (s *triggerMockStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return &memory.CollectionStats{PointCount: uint64(len(s.memories)), Status: "green"}, nil
}
func (s *triggerMockStore) DeleteExpired(_ context.Context) (int, error) { return 0, nil }

// ── helpers ─────────────────────────────────────────────────────────────────

// makeEngine returns an Engine backed by a fresh mock store, with the given
// config, and overrides the ~/.siri directory to a temp dir so tests don't
// touch the real filesystem.
func makeEngine(t *testing.T, cfg Config) (*Engine, *triggerMockStore) {
	t.Helper()
	store := newTriggerMockStore()
	eng := NewEngine(store, nil, cfg)
	// Override home → temp dir so siriDirPath() creates files under t.TempDir().
	t.Setenv("HOME", t.TempDir())
	return eng, store
}

// addMemory is a shorthand to populate the mock store.
func addMemory(store *triggerMockStore, importance float64, reflected bool) {
	m := memory.Memory{
		ID:         memory.New("tmp").ID,
		Type:       memory.TypeEvent,
		Content:    "test memory",
		Importance: importance,
		CreatedAt:  float64(time.Now().Unix()),
		Metadata:   map[string]any{},
	}
	if reflected {
		m.Metadata["reflected"] = true
	}
	store.mu.Lock()
	store.memories[m.ID] = m
	store.mu.Unlock()
}

// ── Gate 1: time interval ───────────────────────────────────────────────────

func TestCheck_Gate1_TooSoon(t *testing.T) {
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	// Enough importance to trigger.
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false) // 5*5 = 25 > 10
	}

	// Write a last-run time that is only 30 minutes ago.
	dir, _ := siriDirPath()
	recentTime := time.Now().Add(-30 * time.Minute).UTC().Format(time.RFC3339)
	_ = os.WriteFile(filepath.Join(dir, reflectionLastRunFile), []byte(recentTime), 0644)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ShouldTrigger {
		t.Error("expected ShouldTrigger=false (too soon), got true")
	}
	if result.SkipReason == "" {
		t.Error("expected non-empty SkipReason when too soon")
	}
}

func TestCheck_Gate1_SufficientInterval(t *testing.T) {
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	// Enough importance.
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false) // 25 > 10
	}

	// Write a last-run time that is 3 hours ago (> 2h min interval).
	dir, _ := siriDirPath()
	oldTime := time.Now().Add(-3 * time.Hour).UTC().Format(time.RFC3339)
	_ = os.WriteFile(filepath.Join(dir, reflectionLastRunFile), []byte(oldTime), 0644)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true after sufficient interval, skip_reason=%q", result.SkipReason)
	}
}

func TestCheck_Gate1_NoLastRunFile(t *testing.T) {
	// First run ever: no last-run file should not fail.
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false)
	}

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should reach Gate 3 (importance check); with 25 importance > threshold 10, should trigger.
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true on first run, skip_reason=%q", result.SkipReason)
	}
	if result.HoursSinceLastRun != 0 {
		t.Errorf("expected HoursSinceLastRun=0 when no last run file, got %.2f", result.HoursSinceLastRun)
	}
}

// ── Gate 2: daily run limit ─────────────────────────────────────────────────

func TestCheck_Gate2_DailyLimitReached(t *testing.T) {
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false)
	}

	// Simulate max runs today.
	dir, _ := siriDirPath()
	_ = writeDailyCount(filepath.Join(dir, reflectionDailyFile), reflectionMaxPerDay)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ShouldTrigger {
		t.Error("expected ShouldTrigger=false when daily limit reached")
	}
	if result.RunsToday != reflectionMaxPerDay {
		t.Errorf("expected RunsToday=%d, got %d", reflectionMaxPerDay, result.RunsToday)
	}
}

func TestCheck_Gate2_DailyLimitNotYetReached(t *testing.T) {
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false)
	}

	// 1 run today (below limit of 3).
	dir, _ := siriDirPath()
	_ = writeDailyCount(filepath.Join(dir, reflectionDailyFile), 1)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true (1 < %d), skip_reason=%q", reflectionMaxPerDay, result.SkipReason)
	}
	if result.RunsToday != 1 {
		t.Errorf("expected RunsToday=1, got %d", result.RunsToday)
	}
}

func TestCheck_Gate2_DailyCountFromYesterdayResets(t *testing.T) {
	cfg := Config{Threshold: 10, MinIntervalH: 2.0, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)
	for i := 0; i < 5; i++ {
		addMemory(store, 5, false)
	}

	// Write yesterday's date with count = 3 (max).
	dir, _ := siriDirPath()
	yesterday := time.Now().UTC().AddDate(0, 0, -1).Format("2006-01-02")
	path := filepath.Join(dir, reflectionDailyFile)
	_ = os.WriteFile(path, []byte(yesterday+" 3\n"), 0644)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should reset to 0 (new day) and pass Gate 2.
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true (yesterday's count resets), skip_reason=%q", result.SkipReason)
	}
	if result.RunsToday != 0 {
		t.Errorf("expected RunsToday=0 after day reset, got %d", result.RunsToday)
	}
}

// ── Gate 3: importance accumulation ─────────────────────────────────────────

func TestCheck_Gate3_BelowThreshold(t *testing.T) {
	cfg := Config{Threshold: 50, MinIntervalH: 0.001, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)

	// Only 2 unreflected memories with importance 5 each = total 10 < 50.
	addMemory(store, 5, false)
	addMemory(store, 5, false)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ShouldTrigger {
		t.Error("expected ShouldTrigger=false when importance below threshold")
	}
	if result.UnreflectedCount != 2 {
		t.Errorf("expected UnreflectedCount=2, got %d", result.UnreflectedCount)
	}
	if result.AccumulatedImportance != 10 {
		t.Errorf("expected AccumulatedImportance=10, got %.1f", result.AccumulatedImportance)
	}
}

func TestCheck_Gate3_AtThreshold(t *testing.T) {
	cfg := Config{Threshold: 50, MinIntervalH: 0.001, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)

	// Exactly 10 memories of importance 5 = total 50 == threshold → should trigger.
	for i := 0; i < 10; i++ {
		addMemory(store, 5, false)
	}

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true at threshold, skip_reason=%q", result.SkipReason)
	}
}

func TestCheck_Gate3_AlreadyReflectedExcluded(t *testing.T) {
	cfg := Config{Threshold: 50, MinIntervalH: 0.001, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)

	// Add 10 memories importance=5 but all reflected → total unreflected = 0 < 50.
	for i := 0; i < 10; i++ {
		addMemory(store, 5, true)
	}
	// Add 3 unreflected importance=5 → total 15 < 50.
	for i := 0; i < 3; i++ {
		addMemory(store, 5, false)
	}

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=false (only 15 unreflected importance), skip_reason=%q", result.SkipReason)
	}
	if result.UnreflectedCount != 3 {
		t.Errorf("expected UnreflectedCount=3 (reflected ones excluded), got %d", result.UnreflectedCount)
	}
}

func TestCheck_Gate3_MixedReflected(t *testing.T) {
	cfg := Config{Threshold: 50, MinIntervalH: 0.001, MaxInputSize: 20}
	eng, store := makeEngine(t, cfg)

	// 5 reflected importance=10 (excluded) + 10 unreflected importance=6 = 60 >= 50.
	for i := 0; i < 5; i++ {
		addMemory(store, 10, true)
	}
	for i := 0; i < 10; i++ {
		addMemory(store, 6, false)
	}

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.ShouldTrigger {
		t.Errorf("expected ShouldTrigger=true (60 unreflected importance), skip_reason=%q", result.SkipReason)
	}
	if result.UnreflectedCount != 10 {
		t.Errorf("expected UnreflectedCount=10, got %d", result.UnreflectedCount)
	}
	if result.AccumulatedImportance != 60 {
		t.Errorf("expected AccumulatedImportance=60, got %.1f", result.AccumulatedImportance)
	}
}

// ── CheckResult fields ───────────────────────────────────────────────────────

func TestCheck_ThresholdPopulatedInResult(t *testing.T) {
	cfg := Config{Threshold: 42, MinIntervalH: 0.001, MaxInputSize: 20}
	eng, _ := makeEngine(t, cfg)

	result, err := eng.Check(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Threshold != 42 {
		t.Errorf("expected Threshold=42 in result, got %.1f", result.Threshold)
	}
}

// ── updateLastRun ────────────────────────────────────────────────────────────

func TestUpdateLastRun_WritesTimestampAndCount(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dir, err := siriDirPath()
	if err != nil {
		t.Fatalf("siriDirPath: %v", err)
	}

	if err := updateLastRun(); err != nil {
		t.Fatalf("updateLastRun: %v", err)
	}

	// Verify last-run file exists and parses as a recent timestamp.
	ts, err := readTimestampFile(filepath.Join(dir, reflectionLastRunFile))
	if err != nil {
		t.Fatalf("readTimestampFile: %v", err)
	}
	if ts.IsZero() {
		t.Error("expected non-zero timestamp after updateLastRun")
	}
	if time.Since(ts) > 5*time.Second {
		t.Errorf("expected recent timestamp, got %v", ts)
	}

	// Verify daily count is 1.
	count, err := readDailyCount(filepath.Join(dir, reflectionDailyFile))
	if err != nil {
		t.Fatalf("readDailyCount: %v", err)
	}
	if count != 1 {
		t.Errorf("expected daily count=1 after first updateLastRun, got %d", count)
	}

	// Call again — should increment to 2.
	if err := updateLastRun(); err != nil {
		t.Fatalf("second updateLastRun: %v", err)
	}
	count, _ = readDailyCount(filepath.Join(dir, reflectionDailyFile))
	if count != 2 {
		t.Errorf("expected daily count=2 after second updateLastRun, got %d", count)
	}
}

// ── fetchUnreflected ─────────────────────────────────────────────────────────

func TestFetchUnreflected_RespectsLimit(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	store := newTriggerMockStore()
	eng := NewEngine(store, nil, DefaultConfig())

	// Add 50 unreflected memories.
	for i := 0; i < 50; i++ {
		addMemory(store, 5, false)
	}

	unreflected, err := eng.fetchUnreflected(context.Background(), 20)
	if err != nil {
		t.Fatalf("fetchUnreflected: %v", err)
	}
	if len(unreflected) > 20 {
		t.Errorf("fetchUnreflected should respect limit=20, got %d", len(unreflected))
	}
}

func TestFetchUnreflected_ExcludesReflected(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	store := newTriggerMockStore()
	eng := NewEngine(store, nil, DefaultConfig())

	// 10 unreflected + 5 reflected.
	for i := 0; i < 10; i++ {
		addMemory(store, 5, false)
	}
	for i := 0; i < 5; i++ {
		addMemory(store, 5, true)
	}

	unreflected, err := eng.fetchUnreflected(context.Background(), 200)
	if err != nil {
		t.Fatalf("fetchUnreflected: %v", err)
	}
	if len(unreflected) != 10 {
		t.Errorf("expected 10 unreflected, got %d", len(unreflected))
	}
	for _, m := range unreflected {
		if isReflected(m) {
			t.Errorf("fetchUnreflected returned a reflected memory: %s", m.ID)
		}
	}
}

// ── readTimestampFile ────────────────────────────────────────────────────────

func TestReadTimestampFile_MissingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "no_file")
	ts, err := readTimestampFile(path)
	if err != nil {
		t.Fatalf("expected no error for missing file, got: %v", err)
	}
	if !ts.IsZero() {
		t.Errorf("expected zero time for missing file, got %v", ts)
	}
}

func TestReadTimestampFile_ValidRFC3339(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ts")
	now := time.Now().UTC().Truncate(time.Second)
	_ = os.WriteFile(path, []byte(now.Format(time.RFC3339)), 0644)

	ts, err := readTimestampFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ts.Equal(now) {
		t.Errorf("expected %v, got %v", now, ts)
	}
}

func TestReadTimestampFile_CorruptedContent_FallsBackToMtime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ts")
	_ = os.WriteFile(path, []byte("not-a-timestamp"), 0644)

	ts, err := readTimestampFile(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Should fall back to mtime, which is recent.
	if ts.IsZero() {
		t.Error("expected non-zero time (mtime fallback) for corrupted timestamp file")
	}
	if time.Since(ts) > 10*time.Second {
		t.Errorf("mtime fallback should be recent, got %v", ts)
	}
}
