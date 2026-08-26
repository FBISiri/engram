package server

import "testing"

// TestImportanceMonitor_MeanAndTrigger verifies mean calc + threshold trigger.
func TestImportanceMonitor_MeanAndTrigger(t *testing.T) {
	m := NewImportanceMonitor(10, 7.0)
	var last *Advisory
	// 10 writes at importance 8 → mean 8.0 > 7.0.
	for i := 0; i < 10; i++ {
		last = m.Record("c", 8)
	}
	if last == nil {
		t.Fatal("expected importance_inflation advisory")
	}
	if last.Type != "importance_inflation" || last.Severity != "warning" {
		t.Errorf("unexpected advisory: %+v", last)
	}
	if mean, ok := last.Data["recent_mean"].(float64); !ok || mean < 7.9 {
		t.Errorf("expected recent_mean ~8.0, got %v", last.Data["recent_mean"])
	}
	if sc, ok := last.Data["sample_count"].(int); !ok || sc != 10 {
		t.Errorf("expected sample_count 10, got %v", last.Data["sample_count"])
	}
}

// TestImportanceMonitor_BelowThreshold: mean under threshold → no advisory.
func TestImportanceMonitor_BelowThreshold(t *testing.T) {
	m := NewImportanceMonitor(10, 7.0)
	var last *Advisory
	for i := 0; i < 10; i++ {
		last = m.Record("c", 5)
	}
	if last != nil {
		t.Fatalf("expected no advisory for mean 5.0, got %+v", last)
	}
}

// TestImportanceMonitor_PartialFill: fewer than window/2 samples → no advisory
// even if the mean exceeds the threshold.
func TestImportanceMonitor_PartialFill(t *testing.T) {
	m := NewImportanceMonitor(10, 7.0)
	// window/2 == 5; 4 samples is below the minimum.
	for i := 0; i < 4; i++ {
		if a := m.Record("c", 10); a != nil {
			t.Fatalf("write %d: expected no advisory below window/2, got %+v", i, a)
		}
	}
	// 5th sample reaches window/2 → advisory.
	if a := m.Record("c", 10); a == nil {
		t.Fatal("expected advisory once sample_count reaches window/2")
	}
}

// TestImportanceMonitor_PerCollectionIsolation: each collection has its own ring.
func TestImportanceMonitor_PerCollectionIsolation(t *testing.T) {
	m := NewImportanceMonitor(10, 7.0)
	for i := 0; i < 10; i++ {
		m.Record("high", 9)
		m.Record("low", 2)
	}
	if a := m.Record("low", 2); a != nil {
		t.Errorf("low collection must not trigger, got %+v", a)
	}
	if a := m.Record("high", 9); a == nil {
		t.Error("high collection must trigger")
	}
}

// TestImportanceMonitor_RingOverwrite: the ring keeps only the last N values, so
// old low values are evicted by newer high ones.
func TestImportanceMonitor_RingOverwrite(t *testing.T) {
	m := NewImportanceMonitor(4, 7.0)
	// Fill with low values.
	for i := 0; i < 4; i++ {
		m.Record("c", 1)
	}
	// Overwrite entirely with high values → mean should climb to 9.
	var last *Advisory
	for i := 0; i < 4; i++ {
		last = m.Record("c", 9)
	}
	if last == nil {
		t.Fatal("expected advisory after ring overwritten with high values")
	}
	if mean := last.Data["recent_mean"].(float64); mean < 8.9 {
		t.Errorf("expected mean ~9 after overwrite, got %v", mean)
	}
}

// TestRingStats verifies mean/p90 computation directly.
func TestRingStats(t *testing.T) {
	r := &importanceRing{values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, count: 10}
	mean, p90 := ringStats(r)
	if mean != 5.5 {
		t.Errorf("mean: want 5.5, got %v", mean)
	}
	if p90 != 9 {
		t.Errorf("p90: want 9, got %v", p90)
	}
}
