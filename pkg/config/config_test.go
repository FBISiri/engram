package config

import (
	"os"
	"strings"
	"testing"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/reflection"
)

// clearEngramEnv removes every ENGRAM_* variable for the duration of the test so
// that Load() observes a pristine environment. t.Setenv restores originals on
// cleanup, so tests remain hermetic and order-independent.
func clearEngramEnv(t *testing.T) {
	t.Helper()
	for _, kv := range os.Environ() {
		key := kv[:strings.IndexByte(kv, '=')]
		if strings.HasPrefix(key, "ENGRAM_") {
			t.Setenv(key, "")
			_ = os.Unsetenv(key)
		}
	}
}

func TestLoad_Defaults(t *testing.T) {
	clearEngramEnv(t)
	c := Load()

	if c == nil {
		t.Fatal("Load returned nil")
	}
	// Spot-check one field of each type / section against documented defaults.
	checks := []struct {
		name string
		got  any
		want any
	}{
		{"QdrantURL", c.QdrantURL, "localhost:6334"},
		{"QdrantUseTLS", c.QdrantUseTLS, false},
		{"EmbedderProvider", c.EmbedderProvider, "openai"},
		{"EmbeddingModel", c.EmbeddingModel, "text-embedding-3-small"},
		{"EmbeddingDimension", c.EmbeddingDimension, 1536},
		{"OpenAIBaseURL", c.OpenAIBaseURL, "https://api.openai.com/v1"},
		{"WeightRelevance", c.Weights.Relevance, 1.0},
		{"WeightRecency", c.Weights.Recency, 0.5},
		{"WeightImportance", c.Weights.Importance, 0.3},
		{"MMRLambda", c.MMRLambda, 0.5},
		{"DedupThreshold", c.DedupThreshold, 0.92},
		{"Transport", c.Transport, "stdio"},
		{"HTTPPort", c.HTTPPort, 8080},
		{"ReflectionEnabled", c.ReflectionEnabled, false},
		{"ReflectionTrigger", c.ReflectionTrigger, "count"},
		{"ReflectionCount", c.ReflectionCount, 10},
		{"ReflectionModel", c.ReflectionModel, "claude-sonnet-4-20250514"},
		{"ProvenanceMode", c.ProvenanceMode, "warn"},
	}
	for _, ck := range checks {
		if ck.got != ck.want {
			t.Errorf("%s default = %v, want %v", ck.name, ck.got, ck.want)
		}
	}
	// Decay must equal the memory package default.
	if c.Decay != memory.DefaultDecayConfig() {
		t.Errorf("Decay default = %+v, want %+v", c.Decay, memory.DefaultDecayConfig())
	}
	if c.RequireProvenance {
		t.Error("RequireProvenance default should be false")
	}
	if len(c.AllowedProvenances) != 0 {
		t.Errorf("AllowedProvenances default = %v, want empty", c.AllowedProvenances)
	}

	// A-MAC MVP per-type importance bounds [min,max] and defaults.
	wantBounds := map[memory.MemoryType][2]float64{
		memory.TypeIdentity:  {7, 9},
		memory.TypeDirective: {6, 10},
		memory.TypeInsight:   {5, 8},
		memory.TypeEvent:     {3, 7},
	}
	for ty, want := range wantBounds {
		if got := c.ImportanceBounds[ty]; got != want {
			t.Errorf("ImportanceBounds[%s] = %v, want %v", ty, got, want)
		}
	}
	if got := c.ImportanceDefaults[memory.TypeIdentity]; got != 7 {
		t.Errorf("ImportanceDefaults[identity] = %v, want 7", got)
	}
}

func TestLoad_EnvOverride(t *testing.T) {
	clearEngramEnv(t)
	env := map[string]string{
		"ENGRAM_QDRANT_URL":          "qdrant.internal:7000",
		"ENGRAM_QDRANT_USE_TLS":      "true",
		"ENGRAM_EMBEDDER_PROVIDER":   "voyage",
		"ENGRAM_EMBEDDING_DIMENSION": "1024",
		"ENGRAM_WEIGHT_RELEVANCE":    "2.5",
		"ENGRAM_MMR_LAMBDA":          "0.7",
		"ENGRAM_DEDUP_THRESHOLD":     "0.70",
		"ENGRAM_HTTP_PORT":           "9090",
		"ENGRAM_REFLECTION_ENABLED":  "1",
		"ENGRAM_REFLECTION_COUNT":    "25",
		"ENGRAM_REQUIRE_PROVENANCE":  "true",
		"ENGRAM_ALLOWED_PROVENANCES": "user_input,web_search,document",
		"ENGRAM_PROVENANCE_MODE":     "strict",
	}
	for k, v := range env {
		t.Setenv(k, v)
	}
	c := Load()

	if c.QdrantURL != "qdrant.internal:7000" {
		t.Errorf("QdrantURL = %q", c.QdrantURL)
	}
	if !c.QdrantUseTLS {
		t.Error("QdrantUseTLS should be true")
	}
	if c.EmbedderProvider != "voyage" {
		t.Errorf("EmbedderProvider = %q", c.EmbedderProvider)
	}
	if c.EmbeddingDimension != 1024 {
		t.Errorf("EmbeddingDimension = %d", c.EmbeddingDimension)
	}
	if c.Weights.Relevance != 2.5 {
		t.Errorf("Weights.Relevance = %v", c.Weights.Relevance)
	}
	if c.MMRLambda != 0.7 {
		t.Errorf("MMRLambda = %v", c.MMRLambda)
	}
	if c.DedupThreshold != 0.70 {
		t.Errorf("DedupThreshold = %v", c.DedupThreshold)
	}
	if c.HTTPPort != 9090 {
		t.Errorf("HTTPPort = %d", c.HTTPPort)
	}
	if !c.ReflectionEnabled {
		t.Error("ReflectionEnabled should be true (env=1)")
	}
	if c.ReflectionCount != 25 {
		t.Errorf("ReflectionCount = %d", c.ReflectionCount)
	}
	if !c.RequireProvenance {
		t.Error("RequireProvenance should be true")
	}
	if c.ProvenanceMode != "strict" {
		t.Errorf("ProvenanceMode = %q, want strict", c.ProvenanceMode)
	}
	wantProv := []string{"user_input", "web_search", "document"}
	if len(c.AllowedProvenances) != len(wantProv) {
		t.Fatalf("AllowedProvenances = %v, want %v", c.AllowedProvenances, wantProv)
	}
	for i, v := range wantProv {
		if c.AllowedProvenances[i] != v {
			t.Errorf("AllowedProvenances[%d] = %q, want %q", i, c.AllowedProvenances[i], v)
		}
	}
}

func TestEnvStr(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal string
		want       string
	}{
		{"unset returns default", false, "", "def", "def"},
		{"empty returns default", true, "", "def", "def"},
		{"set returns value", true, "custom", "def", "custom"},
		{"whitespace is a value", true, " ", "def", " "},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_STR"
			_ = os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envStr(key, tt.defaultVal); got != tt.want {
				t.Errorf("envStr = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestEnvInt(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal int
		want       int
	}{
		{"unset returns default", false, "", 7, 7},
		{"empty returns default", true, "", 7, 7},
		{"valid int", true, "42", 7, 42},
		{"negative int", true, "-3", 7, -3},
		{"malformed returns default", true, "notanint", 7, 7},
		{"float string returns default", true, "3.14", 7, 7},
		{"overflow returns default", true, "99999999999999999999999", 7, 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_INT"
			_ = os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envInt(key, tt.defaultVal); got != tt.want {
				t.Errorf("envInt = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestEnvFloat(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal float64
		want       float64
	}{
		{"unset returns default", false, "", 0.5, 0.5},
		{"empty returns default", true, "", 0.5, 0.5},
		{"valid float", true, "0.82", 0.5, 0.82},
		{"integer string", true, "2", 0.5, 2.0},
		{"scientific notation", true, "1e-2", 0.5, 0.01},
		{"malformed returns default", true, "abc", 0.5, 0.5},
		{"trailing junk returns default", true, "0.7x", 0.5, 0.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_FLOAT"
			_ = os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envFloat(key, tt.defaultVal); got != tt.want {
				t.Errorf("envFloat = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestEnvBool(t *testing.T) {
	tests := []struct {
		name       string
		set        bool
		val        string
		defaultVal bool
		want       bool
	}{
		{"unset returns default true", false, "", true, true},
		{"unset returns default false", false, "", false, false},
		{"empty returns default", true, "", true, true},
		{"true literal", true, "true", false, true},
		{"1 is true", true, "1", false, true},
		{"0 is false", true, "0", true, false},
		{"false literal", true, "false", true, false},
		{"TRUE uppercase", true, "TRUE", false, true},
		{"malformed returns default true", true, "yes", true, true},
		{"malformed returns default false", true, "nope", false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const key = "ENGRAM_TEST_BOOL"
			_ = os.Unsetenv(key)
			if tt.set {
				t.Setenv(key, tt.val)
			}
			if got := envBool(key, tt.defaultVal); got != tt.want {
				t.Errorf("envBool = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParsePrincipalKeys(t *testing.T) {
	cases := []struct {
		in   string
		want map[string]string
	}{
		{"", nil},
		{"pigo:abc", map[string]string{"pigo": "abc"}},
		{"pigo:abc,reflection:def", map[string]string{"pigo": "abc", "reflection": "def"}},
		{" pigo : abc , ", map[string]string{"pigo": "abc"}},
		{"malformed", nil},
		{":nokey,notype:", nil},
		// invalid/typo caller-type is skipped (must not default to engram_user).
		{"reflectionn:key", nil},
		{"pigo:abc,reflectionn:bad", map[string]string{"pigo": "abc"}},
		{"agent-self:xyz", map[string]string{"agent-self": "xyz"}},
	}
	for _, tc := range cases {
		got := parsePrincipalKeys(tc.in)
		if len(got) != len(tc.want) {
			t.Errorf("parsePrincipalKeys(%q) = %v, want %v", tc.in, got, tc.want)
			continue
		}
		for k, v := range tc.want {
			if got[k] != v {
				t.Errorf("parsePrincipalKeys(%q)[%s] = %q, want %q", tc.in, k, got[k], v)
			}
		}
	}
}

// =============================================================================
// Gap coverage: focused ENGRAM_PROVENANCE_MODE / ENGRAM_REQUIRE_PROVENANCE
// loading. TestLoad_Defaults / TestLoad_EnvOverride spot-check these broadly;
// these add single-purpose cases, notably the invalid-mode fallback which no
// existing test exercises.
// =============================================================================

func TestConfig_ProvenanceModeDefault(t *testing.T) {
	clearEngramEnv(t)
	if got := Load().ProvenanceMode; got != "warn" {
		t.Errorf("ProvenanceMode default = %q, want warn", got)
	}
}

func TestConfig_ProvenanceModeStrict(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_PROVENANCE_MODE", "strict")
	if got := Load().ProvenanceMode; got != "strict" {
		t.Errorf("ProvenanceMode = %q, want strict", got)
	}
}

// R1: "default" is a valid ENGRAM_PROVENANCE_MODE and passes through unchanged.
func TestProvenanceModeDefaultPassesThrough(t *testing.T) {
	if got := provenanceMode("default"); got != "default" {
		t.Errorf("provenanceMode(\"default\") = %q, want default", got)
	}
}

func TestConfig_ProvenanceModeDefaultEnv(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_PROVENANCE_MODE", "default")
	if got := Load().ProvenanceMode; got != "default" {
		t.Errorf("ProvenanceMode = %q, want default", got)
	}
}

// R3: Config.ProvenanceFilterConfig() maps the flat fields to a
// reflection.ProvenanceFilterConfig, including the mode string translation.
func TestConfig_ProvenanceFilterConfig(t *testing.T) {
	cases := []struct {
		mode     string
		wantMode reflection.ProvenanceFilterMode
	}{
		{"warn", reflection.ProvenanceModeWarn},
		{"strict", reflection.ProvenanceModeBlock},
		{"default", reflection.ProvenanceModeDefault},
		{"bogus", reflection.ProvenanceModeDefault},
	}
	for _, tc := range cases {
		c := &Config{
			RequireProvenance:  true,
			AllowedProvenances: []string{"user_input", "web_search"},
			ProvenanceMode:     tc.mode,
		}
		pf := c.ProvenanceFilterConfig()
		if !pf.Enabled {
			t.Errorf("mode %q: Enabled = false, want true", tc.mode)
		}
		if pf.Mode != tc.wantMode {
			t.Errorf("mode %q: Mode = %q, want %q", tc.mode, pf.Mode, tc.wantMode)
		}
		if len(pf.AllowedProvenances) != 2 {
			t.Errorf("mode %q: AllowedProvenances = %v, want 2 entries", tc.mode, pf.AllowedProvenances)
		}
	}

	// RequireProvenance=false → Enabled=false.
	if (&Config{RequireProvenance: false, ProvenanceMode: "default"}).ProvenanceFilterConfig().Enabled {
		t.Error("Enabled should be false when RequireProvenance is false")
	}
}

func TestConfig_ProvenanceModeInvalid(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_PROVENANCE_MODE", "loose")
	if got := Load().ProvenanceMode; got != "warn" {
		t.Errorf("invalid ProvenanceMode should fall back to warn, got %q", got)
	}
}

func TestConfig_RequireProvenanceDefault(t *testing.T) {
	clearEngramEnv(t)
	if Load().RequireProvenance {
		t.Error("RequireProvenance default should be false")
	}
}

func TestConfig_RequireProvenanceTrue(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_REQUIRE_PROVENANCE", "true")
	if !Load().RequireProvenance {
		t.Error("RequireProvenance should be true when ENGRAM_REQUIRE_PROVENANCE=true")
	}
}

func TestEvaporationConfigDefaults(t *testing.T) {
	clearEngramEnv(t)
	c := Load().Evaporation
	if c.Enabled {
		t.Error("Evaporation.Enabled default should be false")
	}
	if c.HalfLifeDays[memory.TypeEvent] != 30 || c.HalfLifeDays[memory.TypeInsight] != 180 ||
		c.HalfLifeDays[memory.TypeDirective] != 0 || c.HalfLifeDays[memory.TypeIdentity] != 0 {
		t.Errorf("unexpected default half-lives: %+v", c.HalfLifeDays)
	}
	if c.AccessBoostAlpha != 0.15 || c.EvictionThreshold != 1.0 || c.SweepIntervalH != 6 || c.SweepBatchLimit != 100 {
		t.Errorf("unexpected scalar defaults: %+v", c)
	}
}

func TestEvaporationConfigFromEnv(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_EVAPORATION_ENABLED", "true")
	t.Setenv("ENGRAM_EVAPORATION_HALF_LIFE_EVENT", "45")
	t.Setenv("ENGRAM_EVAPORATION_HALF_LIFE_INSIGHT", "120")
	t.Setenv("ENGRAM_EVAPORATION_HALF_LIFE_DIRECTIVE", "400")
	t.Setenv("ENGRAM_EVAPORATION_HALF_LIFE_IDENTITY", "10")
	t.Setenv("ENGRAM_EVAPORATION_ACCESS_BOOST_ALPHA", "0.2")
	t.Setenv("ENGRAM_EVAPORATION_EVICTION_THRESHOLD", "1.5")
	t.Setenv("ENGRAM_EVAPORATION_SWEEP_INTERVAL_H", "12")
	t.Setenv("ENGRAM_EVAPORATION_SWEEP_BATCH_LIMIT", "50")

	c := Load().Evaporation
	if !c.Enabled {
		t.Error("Enabled should be true")
	}
	if c.HalfLifeDays[memory.TypeEvent] != 45 || c.HalfLifeDays[memory.TypeInsight] != 120 ||
		c.HalfLifeDays[memory.TypeDirective] != 400 || c.HalfLifeDays[memory.TypeIdentity] != 10 {
		t.Errorf("half-lives not loaded: %+v", c.HalfLifeDays)
	}
	if c.AccessBoostAlpha != 0.2 || c.EvictionThreshold != 1.5 || c.SweepIntervalH != 12 || c.SweepBatchLimit != 50 {
		t.Errorf("scalars not loaded: %+v", c)
	}
}

// TestEvaporationConfigV2Defaults covers the spec §4.8 v2 defaults.
func TestEvaporationConfigV2Defaults(t *testing.T) {
	clearEngramEnv(t)
	c := Load().Evaporation
	if !c.DryRun {
		t.Error("DryRun default should be true")
	}
	if c.DecayBasis != "last_access" {
		t.Errorf("DecayBasis default = %q, want last_access", c.DecayBasis)
	}
	if c.MinAgeDays != 14 || c.ObservationDays != 30 {
		t.Errorf("MinAgeDays=%v ObservationDays=%v, want 14/30", c.MinAgeDays, c.ObservationDays)
	}
	if c.ProtectImportance != 8 || c.ProtectAccessCount != 5 || c.ProtectRecentAccessDays != 30 {
		t.Errorf("protect scalars = %v/%v/%v, want 8/5/30", c.ProtectImportance, c.ProtectAccessCount, c.ProtectRecentAccessDays)
	}
	if !c.ProtectCorroborated {
		t.Error("ProtectCorroborated default should be true")
	}
	if len(c.ProtectTypes) != 2 || c.ProtectTypes[0] != memory.TypeIdentity || c.ProtectTypes[1] != memory.TypeDirective {
		t.Errorf("ProtectTypes = %v, want [identity directive]", c.ProtectTypes)
	}
	wantTags := []string{"permanent", "frank-feedback", "directive", "identity"}
	if len(c.ProtectTags) != len(wantTags) {
		t.Fatalf("ProtectTags = %v, want %v", c.ProtectTags, wantTags)
	}
	for i, w := range wantTags {
		if c.ProtectTags[i] != w {
			t.Errorf("ProtectTags[%d] = %q, want %q", i, c.ProtectTags[i], w)
		}
	}
}

// TestEvaporationConfigV2FromEnv covers csv parsing and scalar overrides.
func TestEvaporationConfigV2FromEnv(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_EVAPORATION_DRY_RUN", "false")
	t.Setenv("ENGRAM_EVAPORATION_DECAY_BASIS", "created")
	t.Setenv("ENGRAM_EVAPORATION_MIN_AGE_DAYS", "7")
	t.Setenv("ENGRAM_EVAPORATION_OBSERVATION_DAYS", "45")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_IMPORTANCE", "9")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_ACCESS_COUNT", "3")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_RECENT_ACCESS_DAYS", "14")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_CORROBORATED", "false")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_TYPES", " identity , insight ")
	t.Setenv("ENGRAM_EVAPORATION_PROTECT_TAGS", "pinned, keep")

	c := Load().Evaporation
	if c.DryRun {
		t.Error("DryRun should be false")
	}
	if c.DecayBasis != "created" {
		t.Errorf("DecayBasis = %q, want created", c.DecayBasis)
	}
	if c.MinAgeDays != 7 || c.ObservationDays != 45 {
		t.Errorf("MinAgeDays=%v ObservationDays=%v, want 7/45", c.MinAgeDays, c.ObservationDays)
	}
	if c.ProtectImportance != 9 || c.ProtectAccessCount != 3 || c.ProtectRecentAccessDays != 14 {
		t.Errorf("protect scalars = %v/%v/%v, want 9/3/14", c.ProtectImportance, c.ProtectAccessCount, c.ProtectRecentAccessDays)
	}
	if c.ProtectCorroborated {
		t.Error("ProtectCorroborated should be false")
	}
	if len(c.ProtectTypes) != 2 || c.ProtectTypes[0] != memory.TypeIdentity || c.ProtectTypes[1] != memory.TypeInsight {
		t.Errorf("ProtectTypes = %v, want [identity insight] (trimmed csv)", c.ProtectTypes)
	}
	if len(c.ProtectTags) != 2 || c.ProtectTags[0] != "pinned" || c.ProtectTags[1] != "keep" {
		t.Errorf("ProtectTags = %v, want [pinned keep]", c.ProtectTags)
	}
}

// TestEvaporationDecayBasis_InvalidFallsBack: an invalid DECAY_BASIS is loaded
// verbatim (config layer) but EffectiveImportance falls back to last_access
// with a stderr warning (memory layer). Here we assert the config carries the
// raw value; the fallback behaviour is covered in the memory package.
func TestEvaporationDecayBasis_InvalidLoaded(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_EVAPORATION_DECAY_BASIS", "bogus")
	if got := Load().Evaporation.DecayBasis; got != "bogus" {
		t.Errorf("DecayBasis = %q, want bogus (raw passthrough)", got)
	}
}

// TestLoad_DedupThresholdGlobalFallback verifies per-type dedup thresholds fall
// back to the GLOBAL env ENGRAM_DEDUP_THRESHOLD when the per-type var is unset,
// while a per-type var still overrides the global (spec R1 precedence).
func TestLoad_DedupThresholdGlobalFallback(t *testing.T) {
	clearEngramEnv(t)
	t.Setenv("ENGRAM_DEDUP_THRESHOLD", "0.80")
	t.Setenv("ENGRAM_DEDUP_THRESHOLD_INSIGHT", "0.70")
	c := Load()

	// identity/directive/event: no per-type var → global 0.80 wins over A-MAC default.
	for _, ty := range []memory.MemoryType{memory.TypeIdentity, memory.TypeDirective, memory.TypeEvent} {
		if got := c.DedupThresholds[ty]; got != 0.80 {
			t.Errorf("%s threshold = %v, want 0.80 (global fallback)", ty, got)
		}
	}
	// insight: per-type var overrides the global.
	if got := c.DedupThresholds[memory.TypeInsight]; got != 0.70 {
		t.Errorf("insight threshold = %v, want 0.70 (per-type override)", got)
	}

	// With no env at all, A-MAC hard-coded defaults apply.
	clearEngramEnv(t)
	c2 := Load()
	if got := c2.DedupThresholds[memory.TypeInsight]; got != 0.92 {
		t.Errorf("insight default = %v, want 0.92 (A-MAC default)", got)
	}
	if got := c2.DedupThresholds[memory.TypeIdentity]; got != 0.95 {
		t.Errorf("identity default = %v, want 0.95 (A-MAC default)", got)
	}
}
