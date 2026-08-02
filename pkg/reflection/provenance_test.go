package reflection

import (
	"errors"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
)

// findSourceTypeFilter returns the first metadata.source_type filter with the
// given op, or nil.
func findSourceTypeFilter(filters []memory.Filter, op memory.FilterOp) *memory.Filter {
	for i := range filters {
		if filters[i].Field == "metadata.source_type" && filters[i].Op == op {
			return &filters[i]
		}
	}
	return nil
}

func TestBuildEvidenceFilters_WarnMode(t *testing.T) {
	cfg := ProvenanceFilterConfig{
		Enabled:            true,
		Mode:               ProvenanceModeWarn,
		AllowedProvenances: []string{"user_input"},
	}
	filters := BuildEvidenceFilters(cfg)
	if len(filters) != 0 {
		t.Fatalf("warn mode should return no filters, got %d: %+v", len(filters), filters)
	}
}

func TestBuildEvidenceFilters_DefaultMode(t *testing.T) {
	cfg := ProvenanceFilterConfig{
		Enabled:            true,
		Mode:               ProvenanceModeDefault,
		AllowedProvenances: []string{"user_input", "web_search"},
	}
	filters := BuildEvidenceFilters(cfg)
	if len(filters) != 1 {
		t.Fatalf("default mode should return exactly 1 filter, got %d", len(filters))
	}
	in := findSourceTypeFilter(filters, memory.OpIn)
	if in == nil {
		t.Fatal("expected an OpIn filter on metadata.source_type")
	}
	vals, ok := in.Value.([]string)
	if !ok || len(vals) != 2 || vals[0] != "user_input" || vals[1] != "web_search" {
		t.Errorf("unexpected OpIn value: %v", in.Value)
	}
	if findSourceTypeFilter(filters, memory.OpIsNull) != nil {
		t.Error("default mode must NOT emit an IsNull filter")
	}
}

func TestBuildEvidenceFilters_BlockMode(t *testing.T) {
	cfg := ProvenanceFilterConfig{
		Enabled:            true,
		Mode:               ProvenanceModeBlock,
		AllowedProvenances: []string{"user_input"},
	}
	filters := BuildEvidenceFilters(cfg)
	if len(filters) != 2 {
		t.Fatalf("block mode should return exactly 2 filters, got %d: %+v", len(filters), filters)
	}
	if findSourceTypeFilter(filters, memory.OpIn) == nil {
		t.Error("block mode must emit an OpIn filter on metadata.source_type")
	}
	if findSourceTypeFilter(filters, memory.OpIsNull) == nil {
		t.Error("block mode must emit an OpIsNull filter on metadata.source_type")
	}
}

func TestBuildEvidenceFilters_Disabled(t *testing.T) {
	// Disabled config.
	if f := BuildEvidenceFilters(ProvenanceFilterConfig{Enabled: false, Mode: ProvenanceModeDefault, AllowedProvenances: []string{"x"}}); len(f) != 0 {
		t.Errorf("disabled config should return no filters, got %+v", f)
	}
	// Enabled but empty whitelist.
	if f := BuildEvidenceFilters(ProvenanceFilterConfig{Enabled: true, Mode: ProvenanceModeDefault}); len(f) != 0 {
		t.Errorf("empty whitelist should return no filters, got %+v", f)
	}
}

func TestEnforceWriteBackProvenance_BlockMode(t *testing.T) {
	cfg := ProvenanceFilterConfig{Enabled: true, Mode: ProvenanceModeBlock, AllowedProvenances: []string{"reflection"}}

	// Missing source_type → rejected.
	md := map[string]any{"foo": "bar"}
	if err := EnforceWriteBackProvenance(cfg, md); !errors.Is(err, ErrMissingProvenance) {
		t.Fatalf("expected ErrMissingProvenance, got %v", err)
	}

	// Present but not in whitelist → rejected.
	md2 := map[string]any{"source_type": "web_search"}
	if err := EnforceWriteBackProvenance(cfg, md2); !errors.Is(err, ErrInvalidProvenance) {
		t.Fatalf("expected ErrInvalidProvenance, got %v", err)
	}

	// Present and allowed → OK.
	md3 := map[string]any{"source_type": "reflection"}
	if err := EnforceWriteBackProvenance(cfg, md3); err != nil {
		t.Fatalf("expected nil error for allowed provenance, got %v", err)
	}
}

func TestEnforceWriteBackProvenance_DefaultMode(t *testing.T) {
	cfg := ProvenanceFilterConfig{Enabled: true, Mode: ProvenanceModeDefault}
	md := map[string]any{"foo": "bar"}
	if err := EnforceWriteBackProvenance(cfg, md); err != nil {
		t.Fatalf("default mode should not error, got %v", err)
	}
	if md["source_type"] != "reflection" {
		t.Errorf("default mode should auto-set source_type=reflection, got %v", md["source_type"])
	}

	// Disabled config is a no-op (does not mutate).
	md2 := map[string]any{}
	if err := EnforceWriteBackProvenance(ProvenanceFilterConfig{Enabled: false}, md2); err != nil {
		t.Fatalf("disabled enforcement should not error, got %v", err)
	}
	if _, ok := md2["source_type"]; ok {
		t.Error("disabled enforcement should not mutate metadata")
	}
}

func TestProvenanceFilterConfig_Migration(t *testing.T) {
	// Legacy flat fields migrate into the richer struct (default mode).
	legacy := Config{RequireProvenance: true, AllowedProvenances: []string{"user_input", "web_search"}}
	got := legacy.resolveProvenanceFilter()
	if !got.Enabled {
		t.Fatal("expected migrated config to be Enabled")
	}
	if got.Mode != ProvenanceModeDefault {
		t.Errorf("expected default mode, got %q", got.Mode)
	}
	if len(got.AllowedProvenances) != 2 {
		t.Errorf("expected 2 allowed provenances, got %v", got.AllowedProvenances)
	}

	// Explicit ProvenanceFilter takes precedence over legacy fields.
	explicit := Config{
		RequireProvenance:  true,
		AllowedProvenances: []string{"legacy"},
		ProvenanceFilter:   ProvenanceFilterConfig{Enabled: true, Mode: ProvenanceModeBlock, AllowedProvenances: []string{"new"}},
	}
	got2 := explicit.resolveProvenanceFilter()
	if got2.Mode != ProvenanceModeBlock || len(got2.AllowedProvenances) != 1 || got2.AllowedProvenances[0] != "new" {
		t.Errorf("explicit ProvenanceFilter should win, got %+v", got2)
	}

	// Enabled ProvenanceFilter with empty mode defaults to "default".
	emptyMode := Config{ProvenanceFilter: ProvenanceFilterConfig{Enabled: true, AllowedProvenances: []string{"x"}}}
	if m := emptyMode.resolveProvenanceFilter().Mode; m != ProvenanceModeDefault {
		t.Errorf("empty mode should default to ProvenanceModeDefault, got %q", m)
	}

	// No provenance config at all → disabled.
	if emptyCfg := (Config{}).resolveProvenanceFilter(); emptyCfg.Enabled {
		t.Error("empty Config should resolve to disabled provenance filter")
	}
}
