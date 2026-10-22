package server

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/qdrant"
)

// TestExpiryActionTotalMixed exercises R2(a): a single detailed tick carrying a
// MIXED set of deleted + healed + protected records increments
// engram_expiry_action_total once per record, keyed by action+type, while
// EvaporationHardDeletedTotal keeps its narrower (evaporation-reason physical
// delete) meaning.
func TestExpiryActionTotalMixed(t *testing.T) {
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))

	past := float64(time.Now().Add(-time.Hour).Unix())
	evap := func(id string, ty memory.MemoryType) memory.Memory {
		return memory.Memory{ID: id, Type: ty, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}}
	}
	store := &detailedStore{
		expiryObsStore: &expiryObsStore{mems: []memory.Memory{
			evap("d1", memory.TypeEvent),
			evap("d2", memory.TypeEvent),
			evap("h1", memory.TypeInsight),
			evap("p1", memory.TypeEvent),
		}},
		records: []qdrant.ExpiryRecord{
			{ID: "d1", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionDeleted, Importance: 2},
			{ID: "d2", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionDeleted, Importance: 3},
			{ID: "h1", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionHealed, Importance: 10},
			{ID: "p1", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 9},
		},
	}

	m := metrics.New(nil, nil)
	runExpiryTick(context.Background(), store, m, time.Now())

	cases := []struct {
		action, typ string
		want        float64
	}{
		{"deleted", "event", 2},
		{"healed", "insight", 1},
		{"protected", "event", 1},
		{"deleted", "insight", 0},
		{"healed", "event", 0},
	}
	for _, c := range cases {
		if got := counterValue(t, m.ExpiryActionTotal.WithLabelValues(c.action, c.typ)); got != c.want {
			t.Fatalf("ExpiryActionTotal{action=%s,type=%s}=%v, want %v", c.action, c.typ, got, c.want)
		}
	}
	// EvaporationHardDeletedTotal unchanged: only the 2 evaporation-reason
	// physical deletes (both type=event) counted.
	if got := counterValue(t, m.EvaporationHardDeletedTotal.WithLabelValues("event")); got != 2 {
		t.Fatalf("EvaporationHardDeletedTotal{event}=%v, want 2", got)
	}
}

// TestExpiryActionTotalLegacyBranch covers the legacy count-only store: n
// physical deletes land on action=deleted,type=unknown.
func TestExpiryActionTotalLegacyBranch(t *testing.T) {
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))

	past := float64(time.Now().Add(-time.Hour).Unix())
	store := &expiryObsStore{mems: []memory.Memory{
		{ID: "d1", Type: memory.TypeEvent, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}},
		{ID: "d2", Type: memory.TypeInsight, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}},
	}}

	m := metrics.New(nil, nil)
	runExpiryTick(context.Background(), store, m, time.Now())

	if got := counterValue(t, m.ExpiryActionTotal.WithLabelValues("deleted", "unknown")); got != 2 {
		t.Fatalf("ExpiryActionTotal{deleted,unknown}=%v, want 2", got)
	}
}

// TestExpiryActionTotalNilMetrics covers R2(b): a tick with nil metrics must not
// panic on either branch.
func TestExpiryActionTotalNilMetrics(t *testing.T) {
	t.Setenv("ENGRAM_DELETE_AUDIT_PATH", filepath.Join(t.TempDir(), "deletes.jsonl"))

	past := float64(time.Now().Add(-time.Hour).Unix())
	evap := func(id string, ty memory.MemoryType) memory.Memory {
		return memory.Memory{ID: id, Type: ty, Collection: "engram_user", ValidUntil: past,
			Metadata: map[string]any{"deprecated_reason": "evaporation"}}
	}
	detailed := &detailedStore{
		expiryObsStore: &expiryObsStore{mems: []memory.Memory{evap("d1", memory.TypeEvent)}},
		records: []qdrant.ExpiryRecord{
			{ID: "d1", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionDeleted, Importance: 2},
			{ID: "h1", Collection: "engram_user", Type: "insight", Action: qdrant.ExpiryActionHealed, Importance: 10},
			{ID: "p1", Collection: "engram_user", Type: "event", Action: qdrant.ExpiryActionProtected, Importance: 9},
		},
	}
	legacy := &expiryObsStore{mems: []memory.Memory{evap("d1", memory.TypeEvent)}}

	// Must not panic on either branch.
	runExpiryTick(context.Background(), detailed, nil, time.Now())
	runExpiryTick(context.Background(), legacy, nil, time.Now())
}
