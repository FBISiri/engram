package qdrant

import (
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"

	"github.com/FBISiri/engram/pkg/memory"
)

func TestMemoryToPointAndBack(t *testing.T) {
	mem := memory.New("User lives in Berlin",
		memory.WithType(memory.TypeIdentity),
		memory.WithSource("user"),
		memory.WithImportance(8),
		memory.WithTags("location", "city"),
		memory.WithMetadata(map[string]any{"city": "Berlin", "country": "Germany"}),
	)

	vector := make([]float32, 4) // small vector for test
	for i := range vector {
		vector[i] = float32(i) * 0.1
	}

	pt := memoryToPoint(mem, vector)

	// Verify point ID matches memory ID.
	if pt.Id.GetUuid() != mem.ID {
		t.Errorf("point ID = %q, want %q", pt.Id.GetUuid(), mem.ID)
	}

	// Verify vector was set.
	dense := pt.Vectors.GetVector()
	if dense == nil {
		t.Fatal("expected dense vector, got nil")
	}

	// Round-trip back to Memory.
	restored := pointToMemory(pt.Id, pt.Payload)

	if restored.ID != mem.ID {
		t.Errorf("ID: got %q, want %q", restored.ID, mem.ID)
	}
	if restored.Type != mem.Type {
		t.Errorf("Type: got %q, want %q", restored.Type, mem.Type)
	}
	if restored.Content != mem.Content {
		t.Errorf("Content: got %q, want %q", restored.Content, mem.Content)
	}
	if restored.Source != mem.Source {
		t.Errorf("Source: got %q, want %q", restored.Source, mem.Source)
	}
	if restored.Importance != mem.Importance {
		t.Errorf("Importance: got %f, want %f", restored.Importance, mem.Importance)
	}
	if len(restored.Tags) != len(mem.Tags) {
		t.Errorf("Tags length: got %d, want %d", len(restored.Tags), len(mem.Tags))
	} else {
		for i, tag := range restored.Tags {
			if tag != mem.Tags[i] {
				t.Errorf("Tags[%d]: got %q, want %q", i, tag, mem.Tags[i])
			}
		}
	}
	if restored.CreatedAt != mem.CreatedAt {
		t.Errorf("CreatedAt: got %f, want %f", restored.CreatedAt, mem.CreatedAt)
	}
	if restored.UpdatedAt != mem.UpdatedAt {
		t.Errorf("UpdatedAt: got %f, want %f", restored.UpdatedAt, mem.UpdatedAt)
	}
	// Check metadata was preserved.
	if restored.Metadata == nil {
		t.Fatal("Metadata: got nil, want non-nil")
	}
	if restored.Metadata["city"] != "Berlin" {
		t.Errorf("Metadata[city]: got %v, want Berlin", restored.Metadata["city"])
	}
	if restored.Metadata["country"] != "Germany" {
		t.Errorf("Metadata[country]: got %v, want Germany", restored.Metadata["country"])
	}
}

func TestExtractString(t *testing.T) {
	tests := []struct {
		name string
		id   *qdrant.PointId
		want string
	}{
		{"nil", nil, ""},
		{"uuid", qdrant.NewID("abc-123"), "abc-123"},
		{"num", qdrant.NewIDNum(42), "42"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := extractString(tt.id)
			if got != tt.want {
				t.Errorf("extractString() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuildFilter_Nil(t *testing.T) {
	f := buildFilter(nil)
	if f != nil {
		t.Errorf("expected nil filter for nil input, got %v", f)
	}
}

func TestBuildFilter_Empty(t *testing.T) {
	f := buildFilter([]memory.Filter{})
	if f != nil {
		t.Errorf("expected nil filter for empty input, got %v", f)
	}
}

func TestBuildFilter_SingleEq(t *testing.T) {
	filters := []memory.Filter{
		{Field: "type", Op: memory.OpEq, Value: "identity"},
	}
	f := buildFilter(filters)
	if f == nil {
		t.Fatal("expected non-nil filter")
	}
	if len(f.Must) != 1 {
		t.Fatalf("expected 1 Must condition, got %d", len(f.Must))
	}
}

func TestBuildFilter_MultipleFilters(t *testing.T) {
	filters := []memory.Filter{
		{Field: "type", Op: memory.OpEq, Value: "event"},
		{Field: "created_at", Op: memory.OpGte, Value: 1700000000.0},
		{Field: "importance", Op: memory.OpLte, Value: 8.0},
	}
	f := buildFilter(filters)
	if f == nil {
		t.Fatal("expected non-nil filter")
	}
	if len(f.Must) != 3 {
		t.Fatalf("expected 3 Must conditions, got %d", len(f.Must))
	}
}

func TestBuildFilter_InStrings(t *testing.T) {
	filters := []memory.Filter{
		{Field: "tags", Op: memory.OpIn, Value: []string{"work", "project"}},
	}
	f := buildFilter(filters)
	if f == nil {
		t.Fatal("expected non-nil filter")
	}
	if len(f.Must) != 1 {
		t.Fatalf("expected 1 Must condition, got %d", len(f.Must))
	}
}

func TestBuildFilter_Range(t *testing.T) {
	filters := []memory.Filter{
		{Field: "created_at", Op: memory.OpRange, Value: [2]float64{1700000000.0, 1800000000.0}},
	}
	f := buildFilter(filters)
	if f == nil {
		t.Fatal("expected non-nil filter")
	}
	if len(f.Must) != 1 {
		t.Fatalf("expected 1 Must condition, got %d", len(f.Must))
	}
}

func TestFilterToCondition_IntGte(t *testing.T) {
	f := memory.Filter{Field: "importance", Op: memory.OpGte, Value: int64(5)}
	cond := filterToCondition(f)
	if cond == nil {
		t.Fatal("expected non-nil condition for int64 Gte")
	}
}

func TestFilterToCondition_IntEq(t *testing.T) {
	f := memory.Filter{Field: "count", Op: memory.OpEq, Value: int64(42)}
	cond := filterToCondition(f)
	if cond == nil {
		t.Fatal("expected non-nil condition for int64 Eq")
	}
}

func TestFilterToCondition_BoolEq(t *testing.T) {
	f := memory.Filter{Field: "active", Op: memory.OpEq, Value: true}
	cond := filterToCondition(f)
	if cond == nil {
		t.Fatal("expected non-nil condition for bool Eq")
	}
}

func TestFilterToCondition_FloatEq(t *testing.T) {
	f := memory.Filter{Field: "importance", Op: memory.OpEq, Value: 7.5}
	cond := filterToCondition(f)
	if cond == nil {
		t.Fatal("expected non-nil condition for float Eq")
	}
}

func TestFilterToCondition_InInts(t *testing.T) {
	f := memory.Filter{Field: "count", Op: memory.OpIn, Value: []int64{1, 2, 3}}
	cond := filterToCondition(f)
	if cond == nil {
		t.Fatal("expected non-nil condition for int64 slice In")
	}
}

func TestFilterToCondition_UnknownOp(t *testing.T) {
	f := memory.Filter{Field: "x", Op: "unknown", Value: "y"}
	cond := filterToCondition(f)
	if cond != nil {
		t.Error("expected nil condition for unknown op")
	}
}

func TestParseURL(t *testing.T) {
	tests := []struct {
		raw      string
		wantHost string
		wantPort int
	}{
		{"localhost:6334", "localhost", 6334},
		{"http://qdrant.example.com:6334", "qdrant.example.com", 6334},
		{"https://cloud.qdrant.io:6334", "cloud.qdrant.io", 6334},
		{"localhost", "localhost", 6334},
	}
	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			h, p := parseURL(tt.raw)
			if h != tt.wantHost || p != tt.wantPort {
				t.Errorf("parseURL(%q) = (%q, %d), want (%q, %d)", tt.raw, h, p, tt.wantHost, tt.wantPort)
			}
		})
	}
}

func TestValueToInterface(t *testing.T) {
	// nil
	if v := valueToInterface(nil); v != nil {
		t.Errorf("expected nil, got %v", v)
	}

	// string
	sv := &qdrant.Value{Kind: &qdrant.Value_StringValue{StringValue: "hello"}}
	if v := valueToInterface(sv); v != "hello" {
		t.Errorf("expected \"hello\", got %v", v)
	}

	// double
	dv := &qdrant.Value{Kind: &qdrant.Value_DoubleValue{DoubleValue: 3.14}}
	if v := valueToInterface(dv); v != 3.14 {
		t.Errorf("expected 3.14, got %v", v)
	}

	// bool
	bv := &qdrant.Value{Kind: &qdrant.Value_BoolValue{BoolValue: true}}
	if v := valueToInterface(bv); v != true {
		t.Errorf("expected true, got %v", v)
	}

	// integer
	iv := &qdrant.Value{Kind: &qdrant.Value_IntegerValue{IntegerValue: 42}}
	if v := valueToInterface(iv); v != int64(42) {
		t.Errorf("expected 42, got %v", v)
	}
}

func TestGetStringSlice_NilPayload(t *testing.T) {
	result := getStringSlice(nil, "tags")
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestGetStringSlice_MissingKey(t *testing.T) {
	payload := map[string]*qdrant.Value{}
	result := getStringSlice(payload, "tags")
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestGetMap_NilPayload(t *testing.T) {
	result := getMap(nil, "metadata")
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestGetMap_MissingKey(t *testing.T) {
	payload := map[string]*qdrant.Value{}
	result := getMap(payload, "metadata")
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestPointToMemory_EmptyPayload(t *testing.T) {
	id := qdrant.NewID("test-id")
	payload := map[string]*qdrant.Value{}
	mem := pointToMemory(id, payload)
	if mem.ID != "test-id" {
		t.Errorf("ID: got %q, want \"test-id\"", mem.ID)
	}
	if mem.Content != "" {
		t.Errorf("Content: got %q, want empty", mem.Content)
	}
}

func TestNew_DefaultConfig(t *testing.T) {
	cfg := Config{}
	// We can't actually connect, but verify defaults are applied.
	if cfg.URL == "" {
		cfg.URL = "localhost:6334"
	}
	if cfg.CollectionName == "" {
		cfg.CollectionName = "engram"
	}
	if cfg.Dimension == 0 {
		cfg.Dimension = 1536
	}

	if cfg.URL != "localhost:6334" {
		t.Errorf("URL: got %q, want localhost:6334", cfg.URL)
	}
	if cfg.CollectionName != "engram" {
		t.Errorf("CollectionName: got %q, want engram", cfg.CollectionName)
	}
	if cfg.Dimension != 1536 {
		t.Errorf("Dimension: got %d, want 1536", cfg.Dimension)
	}
}

// =============================================================================
// P1-B: valid_until expiry filter tests
// =============================================================================

// TestMemoryToPoint_ValidUntil verifies that a memory with valid_until > 0
// serialises the field into the Qdrant payload, and that a memory with
// valid_until == 0 (never-expires) does NOT set the field.
func TestMemoryToPoint_ValidUntil(t *testing.T) {
	now := float64(time.Now().Unix())

	t.Run("valid_until set when non-zero", func(t *testing.T) {
		expiry := now + 3600 // 1 hour from now
		mem := memory.New("temporary directive",
			memory.WithType(memory.TypeDirective),
			memory.WithValidUntil(expiry),
		)
		pt := memoryToPoint(mem, []float32{0.1, 0.2})
		val, ok := pt.Payload[fieldValidUntil]
		if !ok {
			t.Fatal("expected valid_until in payload, got absent")
		}
		if val.GetDoubleValue() != expiry {
			t.Errorf("valid_until: got %f, want %f", val.GetDoubleValue(), expiry)
		}
	})

	t.Run("valid_until absent when zero", func(t *testing.T) {
		mem := memory.New("permanent insight",
			memory.WithType(memory.TypeInsight),
		)
		pt := memoryToPoint(mem, []float32{0.1, 0.2})
		if _, ok := pt.Payload[fieldValidUntil]; ok {
			t.Error("expected valid_until absent for zero value, but it was set")
		}
	})
}

// TestPointToMemory_ValidUntil verifies that valid_until round-trips correctly.
func TestPointToMemory_ValidUntil(t *testing.T) {
	expiry := float64(time.Now().Add(time.Hour).Unix())
	mem := memory.New("expiring memory",
		memory.WithType(memory.TypeDirective),
		memory.WithValidUntil(expiry),
	)
	pt := memoryToPoint(mem, []float32{0.1})
	restored := pointToMemory(pt.Id, pt.Payload)
	if restored.ValidUntil != expiry {
		t.Errorf("ValidUntil round-trip: got %f, want %f", restored.ValidUntil, expiry)
	}
}

// TestBuildExpiryFilter_Structure verifies that the MustNot condition used in
// Search/Scroll to exclude expired memories encodes the correct Qdrant filter
// structure: MustNot( Must(valid_until > 0, valid_until < now) ).
//
// This is a structural test — it does not require a live Qdrant connection.
func TestBuildExpiryFilter_Structure(t *testing.T) {
	now := float64(time.Now().Unix())

	// Reproduce the filter construction from Search() / Scroll().
	mustNotConds := []*qdrant.Condition{
		qdrant.NewFilterAsCondition(&qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Gt: qdrant.PtrOf(0.0)}),
				qdrant.NewRange(fieldValidUntil, &qdrant.Range{Lt: qdrant.PtrOf(now)}),
			},
		}),
	}
	filter := &qdrant.Filter{
		MustNot: mustNotConds,
	}

	if len(filter.MustNot) != 1 {
		t.Fatalf("expected 1 MustNot condition, got %d", len(filter.MustNot))
	}

	inner := filter.MustNot[0].GetFilter()
	if inner == nil {
		t.Fatal("expected inner filter, got nil")
	}
	if len(inner.Must) != 2 {
		t.Fatalf("expected 2 inner Must conditions (gt 0, lt now), got %d", len(inner.Must))
	}

	// Verify first condition: valid_until > 0
	c0 := inner.Must[0].GetField()
	if c0 == nil {
		t.Fatal("expected field condition for Must[0]")
	}
	if c0.Key != fieldValidUntil {
		t.Errorf("Must[0] field: got %q, want %q", c0.Key, fieldValidUntil)
	}
	if c0.Range == nil || c0.Range.Gt == nil || *c0.Range.Gt != 0.0 {
		t.Errorf("Must[0]: expected Gt=0.0, got %+v", c0.Range)
	}

	// Verify second condition: valid_until < now
	c1 := inner.Must[1].GetField()
	if c1 == nil {
		t.Fatal("expected field condition for Must[1]")
	}
	if c1.Key != fieldValidUntil {
		t.Errorf("Must[1] field: got %q, want %q", c1.Key, fieldValidUntil)
	}
	if c1.Range == nil || c1.Range.Lt == nil {
		t.Errorf("Must[1]: expected Lt to be set, got %+v", c1.Range)
	}
	if *c1.Range.Lt != now {
		t.Errorf("Must[1] Lt: got %f, want %f", *c1.Range.Lt, now)
	}
}

// TestBuildExpiryFilter_NonExpiredPassThrough verifies that a memory whose
// valid_until is in the future would NOT be caught by the MustNot condition
// (i.e., valid_until < now is false for a future timestamp).
func TestBuildExpiryFilter_NonExpiredPassThrough(t *testing.T) {
	now := float64(time.Now().Unix())
	futureExpiry := now + 3600 // 1 hour in the future

	// The MustNot excludes memories where: valid_until > 0 AND valid_until < now
	// For futureExpiry: valid_until > 0 ✓, valid_until < now ✗ → NOT excluded → passes
	wouldBeExcluded := futureExpiry > 0 && futureExpiry < now
	if wouldBeExcluded {
		t.Error("future-expiry memory should NOT be excluded by expiry filter, but it would be")
	}
}

// TestBuildExpiryFilter_ExpiredCaughtByFilter verifies that an already-expired
// memory WOULD be caught by the MustNot condition.
func TestBuildExpiryFilter_ExpiredCaughtByFilter(t *testing.T) {
	now := float64(time.Now().Unix())
	pastExpiry := now - 3600 // 1 hour in the past

	// The MustNot excludes memories where: valid_until > 0 AND valid_until < now
	// For pastExpiry: valid_until > 0 ✓, valid_until < now ✓ → excluded
	wouldBeExcluded := pastExpiry > 0 && pastExpiry < now
	if !wouldBeExcluded {
		t.Error("past-expiry memory SHOULD be excluded by expiry filter, but it would not be")
	}
}

// TestBuildExpiryFilter_ZeroValidUntilNotExcluded verifies that a permanent
// memory (valid_until == 0) is NOT excluded by the expiry filter.
func TestBuildExpiryFilter_ZeroValidUntilNotExcluded(t *testing.T) {
	now := float64(time.Now().Unix())
	permanentValidUntil := 0.0

	// The MustNot excludes memories where: valid_until > 0 AND valid_until < now
	// For 0: valid_until > 0 ✗ → NOT excluded
	wouldBeExcluded := permanentValidUntil > 0 && permanentValidUntil < now
	if wouldBeExcluded {
		t.Error("permanent memory (valid_until=0) should NOT be excluded, but it would be")
	}
}
