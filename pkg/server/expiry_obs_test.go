package server

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
)

// TestNextSweepEstimate covers the three branches of the pure estimator.
func TestNextSweepEstimate(t *testing.T) {
	t.Parallel()
	interval := 6 * time.Hour

	// hasSwept -> last + interval, basis "last_sweep".
	last := time.Date(2026, 1, 2, 3, 0, 0, 0, time.UTC)
	if est, basis, ok := nextSweepEstimate(last, true, time.Time{}, interval); !ok ||
		basis != "last_sweep" || !est.Equal(last.Add(interval)) {
		t.Fatalf("hasSwept: got (%v,%q,%v), want (%v,last_sweep,true)", est, basis, ok, last.Add(interval))
	}

	// no sweep yet, started set -> started + interval, basis "process_start".
	started := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	if est, basis, ok := nextSweepEstimate(time.Time{}, false, started, interval); !ok ||
		basis != "process_start" || !est.Equal(started.Add(interval)) {
		t.Fatalf("process_start: got (%v,%q,%v), want (%v,process_start,true)", est, basis, ok, started.Add(interval))
	}

	// all-zero -> ok == false.
	if est, basis, ok := nextSweepEstimate(time.Time{}, false, time.Time{}, interval); ok ||
		basis != "" || !est.IsZero() {
		t.Fatalf("all-zero: got (%v,%q,%v), want (zero,\"\",false)", est, basis, ok)
	}
}

// expiryObsStore is a Store supporting Scroll (offset pagination over a fixed
// slice) and DeleteExpired (returns count of expired memories) for the expiry
// observability helper test.
type expiryObsStore struct {
	mems []memory.Memory
}

func (s *expiryObsStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	// Mirror real qdrant semantics: EXCLUDE expired memories.
	now := float64(time.Now().Unix())
	var visible []memory.Memory
	for i := range s.mems {
		if s.mems[i].ValidUntil > 0 && s.mems[i].ValidUntil < now {
			continue
		}
		visible = append(visible, s.mems[i])
	}
	return pageMems(visible, opts)
}

// ScrollExpired returns ONLY expired memories, mirroring the real store's
// expiry-attribution path (implements expiredScroller).
func (s *expiryObsStore) ScrollExpired(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	now := float64(time.Now().Unix())
	var expired []memory.Memory
	for i := range s.mems {
		if s.mems[i].ValidUntil > 0 && s.mems[i].ValidUntil < now {
			expired = append(expired, s.mems[i])
		}
	}
	return pageMems(expired, opts)
}

// pageMems applies offset/limit pagination over a fixed slice.
func pageMems(mems []memory.Memory, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	start := 0
	if opts.Offset != "" {
		start, _ = strconv.Atoi(opts.Offset)
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}
	end := start + limit
	if end >= len(mems) {
		return append([]memory.Memory(nil), mems[start:]...), "", nil
	}
	return append([]memory.Memory(nil), mems[start:end]...), strconv.Itoa(end), nil
}

func (s *expiryObsStore) DeleteExpired(_ context.Context) (int, error) {
	now := float64(time.Now().Unix())
	n := 0
	for i := range s.mems {
		if s.mems[i].ValidUntil > 0 && s.mems[i].ValidUntil < now {
			n++
		}
	}
	return n, nil
}

func (s *expiryObsStore) Insert(context.Context, *memory.Memory, []float32) error { return nil }
func (s *expiryObsStore) Search(context.Context, []float32, memory.SearchOptions) ([]memory.ScoredMemory, error) {
	return nil, nil
}
func (s *expiryObsStore) Delete(context.Context, []string) (int, error) { return 0, nil }
func (s *expiryObsStore) Update(context.Context, string, map[string]any) error {
	return nil
}
func (s *expiryObsStore) SearchByIDs(context.Context, []string) ([]memory.Memory, error) {
	return nil, nil
}
func (s *expiryObsStore) EnsureCollection(context.Context) error { return nil }
func (s *expiryObsStore) Stats(context.Context) (*memory.CollectionStats, error) {
	return &memory.CollectionStats{}, nil
}

// TestExpiryHardDeletedCounter seeds expired memories across >=2 collections
// and mixed deprecated_reason, then runs one expiry tick with a metrics handle
// and asserts the evaporation-type hard-deleted counter advanced by the count
// of evaporation-reason expired memories.
func TestExpiryHardDeletedCounter(t *testing.T) {
	t.Parallel()
	past := float64(time.Now().Add(-time.Hour).Unix())
	future := float64(time.Now().Add(time.Hour).Unix())

	mk := func(id, coll, reason string, typ memory.MemoryType, validUntil float64) memory.Memory {
		return memory.Memory{
			ID:         id,
			Type:       typ,
			Collection: coll,
			ValidUntil: validUntil,
			Metadata:   map[string]any{"deprecated_reason": reason},
		}
	}
	store := &expiryObsStore{mems: []memory.Memory{
		mk("a", "engram_user", "evaporation", memory.TypeEvent, past),
		mk("b", "engram_user", "evaporation", memory.TypeEvent, past),
		mk("c", "engram_reflection", "ttl", memory.TypeInsight, past),
		mk("d", "engram_user", "", memory.TypeEvent, past),              // ttl_or_unset
		mk("e", "engram_user", "evaporation", memory.TypeEvent, future), // not expired
	}}

	m := metrics.New(nil, nil)
	runExpiryTick(context.Background(), store, m, time.Now())

	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")); got != 2 {
		t.Fatalf("EvaporationHardDeletedTotal{event}=%v, want 2", got)
	}
	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("insight")); got != 0 {
		t.Fatalf("EvaporationHardDeletedTotal{insight}=%v, want 0 (ttl reason)", got)
	}
}

// TestScanExpiringBreakdown verifies the collection/reason breakdown is
// attributable across mixed collections and reasons.
func TestScanExpiringBreakdown(t *testing.T) {
	t.Parallel()
	past := float64(time.Now().Add(-time.Hour).Unix())
	mk := func(coll, reason string) memory.Memory {
		return memory.Memory{
			Type:       memory.TypeEvent,
			Collection: coll,
			ValidUntil: past,
			Metadata:   map[string]any{"deprecated_reason": reason},
		}
	}
	store := &expiryObsStore{mems: []memory.Memory{
		mk("engram_user", "evaporation"),
		mk("engram_reflection", "ttl"),
		{Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: past}, // no metadata reason
	}}
	byColl, byReason, _ := scanExpiring(context.Background(), store, time.Now())
	if byColl["engram_user"] != 2 || byColl["engram_reflection"] != 1 {
		t.Fatalf("byCollection=%v", byColl)
	}
	if byReason["evaporation"] != 1 || byReason["ttl"] != 1 || byReason[expiryReasonUnset] != 1 {
		t.Fatalf("byReason=%v", byReason)
	}
}
