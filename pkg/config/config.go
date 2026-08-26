// Package config provides configuration loading from environment variables.
package config

import (
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/reflection"
)

// Config holds all Engram configuration.
type Config struct {
	// Storage
	QdrantURL    string
	QdrantAPIKey string
	QdrantUseTLS bool

	// Embedding
	EmbedderProvider   string // "openai" | "voyage"
	EmbeddingModel     string
	EmbeddingDimension int
	OpenAIAPIKey       string
	OpenAIBaseURL      string
	VoyageAPIKey       string

	// Scoring
	Weights        memory.ScoringWeights
	Decay          memory.DecayConfig
	MMRLambda      float64
	DedupThreshold float64

	// Evaporation (importance runtime decay). Feature-flagged off by default.
	Evaporation memory.EvaporationConfig

	// A-MAC (Adaptive Memory Admission Control) — spec v1. All gated behind
	// AMACEnabled (ENGRAM_AMAC_ENABLED, default false). When disabled the
	// server uses the legacy uniform policy (global dedup threshold, default
	// importance 5 clamped [1,10], no rate limiting).
	AMACEnabled        bool
	DedupThresholds    map[memory.MemoryType]float64    // per-type dedup thresholds
	ImportanceDefaults map[memory.MemoryType]float64    // per-type importance defaults
	ImportanceBounds   map[memory.MemoryType][2]float64 // per-type [min,max] bounds
	RateLimits         map[memory.MemoryType]int        // per-type hourly write limits

	// Server
	Transport string // "stdio", "http", "both"
	HTTPPort  int
	APIKey    string
	// PrincipalKeys maps a caller type ("user", "agent-self", "reflection",
	// "pigo") to a dedicated API key. A request authenticating with a
	// principal key has its caller type derived from the key — the
	// self-declared X-Caller-Type header is ignored. Parsed from
	// ENGRAM_PRINCIPAL_KEYS="pigo:key1,reflection:key2". Optional; the
	// legacy shared APIKey keeps working alongside.
	PrincipalKeys map[string]string

	// Reflection
	ReflectionEnabled  bool
	ReflectionTrigger  string // "count", "cron", "manual"
	ReflectionCount    int
	ReflectionModel    string
	ReflectionMode     string        // ENGRAM_REFLECTION_MODE: "v1" | "v2" (focal point)
	DialecticTimeout   time.Duration // ENGRAM_DIALECTIC_TIMEOUT
	RequireProvenance  bool          // ENGRAM_REQUIRE_PROVENANCE
	AllowedProvenances []string      // ENGRAM_ALLOWED_PROVENANCES (comma-separated)
	ProvenanceMode     string        // ENGRAM_PROVENANCE_MODE: "warn" (default) | "strict" | "default"

	// Write Checkpoints (soft advisory feedback). All gated behind
	// WriteCheckpointsEnabled (ENGRAM_WRITE_CHECKPOINTS_ENABLED, default false).
	WriteCheckpointsEnabled bool

	// CP1: Dedup Advisory
	CPDedupAdvisoryEnabled  bool
	CPDedupAdvisoryMinScore float64

	// CP2: Importance Monitor
	CPImportanceMonitorEnabled bool
	CPImportanceWindow         int
	CPImportanceThreshold      float64

	// CP3: Rate Limit Warning
	CPRateLimitWarningEnabled  bool
	CPRateLimitWarningFraction float64

	// CP4: Content Check
	CPContentCheckEnabled bool
	CPContentMinLength    int
	CPContentMaxLength    int
	CPRequireTags         bool
	CPRequireSourceType   bool
}

// ProvenanceFilterConfig builds a reflection.ProvenanceFilterConfig from the
// flat config fields, making the mapping from the env-loaded fields to the
// reflection filter shape explicit. The ProvenanceMode string is mapped to the
// reflection ProvenanceFilterMode: "warn"→warn, "strict"→block, "default"→default
// (any other value falls back to default, consistent with
// reflection.BuildEvidenceFilters).
func (c *Config) ProvenanceFilterConfig() reflection.ProvenanceFilterConfig {
	var mode reflection.ProvenanceFilterMode
	switch c.ProvenanceMode {
	case "warn":
		mode = reflection.ProvenanceModeWarn
	case "strict":
		mode = reflection.ProvenanceModeBlock
	default: // "default" and unknown values
		mode = reflection.ProvenanceModeDefault
	}
	return reflection.ProvenanceFilterConfig{
		Enabled:            c.RequireProvenance,
		Mode:               mode,
		AllowedProvenances: c.AllowedProvenances,
	}
}

// Load reads configuration from environment variables with sensible defaults.
func Load() *Config {
	return &Config{
		// Storage
		QdrantURL:    envStr("ENGRAM_QDRANT_URL", "localhost:6334"),
		QdrantAPIKey: envStr("ENGRAM_QDRANT_API_KEY", ""),
		QdrantUseTLS: envBool("ENGRAM_QDRANT_USE_TLS", false),

		// Embedding
		EmbedderProvider:   envStr("ENGRAM_EMBEDDER_PROVIDER", "openai"),
		EmbeddingModel:     envStr("ENGRAM_EMBEDDING_MODEL", "text-embedding-3-small"),
		EmbeddingDimension: envInt("ENGRAM_EMBEDDING_DIMENSION", 1536),
		OpenAIAPIKey:       envStr("ENGRAM_OPENAI_API_KEY", ""),
		OpenAIBaseURL:      envStr("ENGRAM_OPENAI_BASE_URL", "https://api.openai.com/v1"),
		VoyageAPIKey:       envStr("ENGRAM_VOYAGE_API_KEY", ""),

		// Scoring
		Weights: memory.ScoringWeights{
			Relevance:  envFloat("ENGRAM_WEIGHT_RELEVANCE", 1.0),
			Recency:    envFloat("ENGRAM_WEIGHT_RECENCY", 0.5),
			Importance: envFloat("ENGRAM_WEIGHT_IMPORTANCE", 0.3),
		},
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      envFloat("ENGRAM_MMR_LAMBDA", 0.5),
		DedupThreshold: envFloat("ENGRAM_DEDUP_THRESHOLD", 0.92),

		Evaporation: loadEvaporationConfig(),

		// A-MAC
		AMACEnabled:        envBool("ENGRAM_AMAC_ENABLED", false),
		DedupThresholds:    loadTypeDedupThresholds(),
		ImportanceDefaults: DefaultImportanceDefaults(),
		ImportanceBounds:   DefaultImportanceBounds(),
		RateLimits:         loadRateLimits(),

		// Server
		Transport:     envStr("ENGRAM_TRANSPORT", "stdio"),
		HTTPPort:      envInt("ENGRAM_HTTP_PORT", 8080),
		APIKey:        envStr("ENGRAM_API_KEY", ""),
		PrincipalKeys: parsePrincipalKeys(envStr("ENGRAM_PRINCIPAL_KEYS", "")),

		// Reflection
		ReflectionEnabled:  envBool("ENGRAM_REFLECTION_ENABLED", false),
		ReflectionTrigger:  envStr("ENGRAM_REFLECTION_TRIGGER", "count"),
		ReflectionCount:    envInt("ENGRAM_REFLECTION_COUNT", 10),
		ReflectionModel:    envStr("ENGRAM_REFLECTION_MODEL", "claude-sonnet-4-20250514"),
		ReflectionMode:     envStr("ENGRAM_REFLECTION_MODE", ""),
		DialecticTimeout:   envDuration("ENGRAM_DIALECTIC_TIMEOUT", 0),
		RequireProvenance:  envBool("ENGRAM_REQUIRE_PROVENANCE", false),
		AllowedProvenances: parseCommaList(envStr("ENGRAM_ALLOWED_PROVENANCES", "")),
		ProvenanceMode:     provenanceMode(envStr("ENGRAM_PROVENANCE_MODE", "warn")),

		// Write Checkpoints
		WriteCheckpointsEnabled: envBool("ENGRAM_WRITE_CHECKPOINTS_ENABLED", false),
		// CP1
		CPDedupAdvisoryEnabled:  envBool("ENGRAM_CP_DEDUP_ADVISORY_ENABLED", true),
		CPDedupAdvisoryMinScore: envFloat("ENGRAM_CP_DEDUP_ADVISORY_MIN_SCORE", 0.70),
		// CP2
		CPImportanceMonitorEnabled: envBool("ENGRAM_CP_IMPORTANCE_MONITOR_ENABLED", true),
		CPImportanceWindow:         envInt("ENGRAM_CP_IMPORTANCE_WINDOW", 50),
		CPImportanceThreshold:      envFloat("ENGRAM_CP_IMPORTANCE_THRESHOLD", 7.0),
		// CP3
		CPRateLimitWarningEnabled:  envBool("ENGRAM_CP_RATE_LIMIT_WARNING_ENABLED", true),
		CPRateLimitWarningFraction: envFloat("ENGRAM_CP_RATE_LIMIT_WARNING_FRACTION", 0.80),
		// CP4
		CPContentCheckEnabled: envBool("ENGRAM_CP_CONTENT_CHECK_ENABLED", true),
		CPContentMinLength:    envInt("ENGRAM_CP_CONTENT_MIN_LENGTH", 20),
		CPContentMaxLength:    envInt("ENGRAM_CP_CONTENT_MAX_LENGTH", 2000),
		CPRequireTags:         envBool("ENGRAM_CP_REQUIRE_TAGS", true),
		CPRequireSourceType:   envBool("ENGRAM_CP_REQUIRE_SOURCE_TYPE", true),
	}
}

// DefaultImportanceDefaults returns the A-MAC per-type importance defaults
// (spec v1 §3.2). directive defaults highest, event lowest.
func DefaultImportanceDefaults() map[memory.MemoryType]float64 {
	return map[memory.MemoryType]float64{
		memory.TypeIdentity:  envFloat("ENGRAM_IMPORTANCE_DEFAULT_IDENTITY", 6),
		memory.TypeDirective: envFloat("ENGRAM_IMPORTANCE_DEFAULT_DIRECTIVE", 7),
		memory.TypeInsight:   envFloat("ENGRAM_IMPORTANCE_DEFAULT_INSIGHT", 5),
		memory.TypeEvent:     envFloat("ENGRAM_IMPORTANCE_DEFAULT_EVENT", 4),
	}
}

// loadEvaporationConfig starts from DefaultEvaporationConfig() and overrides
// each field from ENGRAM_EVAPORATION_* environment variables.
func loadEvaporationConfig() memory.EvaporationConfig {
	c := memory.DefaultEvaporationConfig()
	c.Enabled = envBool("ENGRAM_EVAPORATION_ENABLED", c.Enabled)
	c.HalfLifeDays[memory.TypeEvent] = envFloat("ENGRAM_EVAPORATION_HALF_LIFE_EVENT", c.HalfLifeDays[memory.TypeEvent])
	c.HalfLifeDays[memory.TypeInsight] = envFloat("ENGRAM_EVAPORATION_HALF_LIFE_INSIGHT", c.HalfLifeDays[memory.TypeInsight])
	c.HalfLifeDays[memory.TypeDirective] = envFloat("ENGRAM_EVAPORATION_HALF_LIFE_DIRECTIVE", c.HalfLifeDays[memory.TypeDirective])
	c.HalfLifeDays[memory.TypeIdentity] = envFloat("ENGRAM_EVAPORATION_HALF_LIFE_IDENTITY", c.HalfLifeDays[memory.TypeIdentity])
	c.AccessBoostAlpha = envFloat("ENGRAM_EVAPORATION_ACCESS_BOOST_ALPHA", c.AccessBoostAlpha)
	c.EvictionThreshold = envFloat("ENGRAM_EVAPORATION_EVICTION_THRESHOLD", c.EvictionThreshold)
	c.SweepIntervalH = envInt("ENGRAM_EVAPORATION_SWEEP_INTERVAL_H", c.SweepIntervalH)
	c.SweepBatchLimit = envInt("ENGRAM_EVAPORATION_SWEEP_BATCH_LIMIT", c.SweepBatchLimit)
	return c
}

// DefaultImportanceBounds returns the A-MAC per-type [min,max] importance
// bounds (spec v1 §3.2). Values outside are clamped silently.
func DefaultImportanceBounds() map[memory.MemoryType][2]float64 {
	return map[memory.MemoryType][2]float64{
		memory.TypeIdentity:  {envFloat("ENGRAM_IMPORTANCE_MIN_IDENTITY", 5), envFloat("ENGRAM_IMPORTANCE_MAX_IDENTITY", 9)},
		memory.TypeDirective: {envFloat("ENGRAM_IMPORTANCE_MIN_DIRECTIVE", 5), envFloat("ENGRAM_IMPORTANCE_MAX_DIRECTIVE", 10)},
		memory.TypeInsight:   {envFloat("ENGRAM_IMPORTANCE_MIN_INSIGHT", 3), envFloat("ENGRAM_IMPORTANCE_MAX_INSIGHT", 8)},
		memory.TypeEvent:     {envFloat("ENGRAM_IMPORTANCE_MIN_EVENT", 1), envFloat("ENGRAM_IMPORTANCE_MAX_EVENT", 7)},
	}
}

// DefaultRateLimits returns the A-MAC per-type hourly write limits (spec v1
// §3.3), overridable via ENGRAM_RATE_LIMIT_{TYPE}.
func DefaultRateLimits() map[memory.MemoryType]int {
	return map[memory.MemoryType]int{
		memory.TypeIdentity:  envInt("ENGRAM_RATE_LIMIT_IDENTITY", 5),
		memory.TypeDirective: envInt("ENGRAM_RATE_LIMIT_DIRECTIVE", 10),
		memory.TypeInsight:   envInt("ENGRAM_RATE_LIMIT_INSIGHT", 20),
		memory.TypeEvent:     envInt("ENGRAM_RATE_LIMIT_EVENT", 50),
	}
}

func loadRateLimits() map[memory.MemoryType]int { return DefaultRateLimits() }

// loadTypeDedupThresholds builds the per-type dedup threshold map. Precedence
// (spec R1/§3.1/§4.3): per-type env ENGRAM_DEDUP_THRESHOLD_{TYPE} > global env
// ENGRAM_DEDUP_THRESHOLD > A-MAC hard-coded default (memory.TypeDedupThresholds).
func loadTypeDedupThresholds() map[memory.MemoryType]float64 {
	out := map[memory.MemoryType]float64{}
	for t, amacDefault := range memory.TypeDedupThresholds {
		// Global env overrides the A-MAC default; per-type env then overrides that.
		effectiveDefault := envFloat("ENGRAM_DEDUP_THRESHOLD", amacDefault)
		key := "ENGRAM_DEDUP_THRESHOLD_" + strings.ToUpper(string(t))
		out[t] = envFloat(key, effectiveDefault)
	}
	return out
}

// provenanceMode validates the ENGRAM_PROVENANCE_MODE value. Valid values are
// "warn", "strict" and "default"; any other value logs a warning and falls
// back to "warn".
func provenanceMode(raw string) string {
	switch raw {
	case "warn", "strict", "default":
		return raw
	default:
		log.Printf("WARN config: ENGRAM_PROVENANCE_MODE=%q is invalid, falling back to \"warn\"", raw)
		return "warn"
	}
}

// parseCommaList splits a comma-separated string into trimmed, non-empty
// elements. Returns nil for an empty/whitespace-only input.
func parseCommaList(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if part = strings.TrimSpace(part); part != "" {
			out = append(out, part)
		}
	}
	return out
}

func envStr(key, defaultVal string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultVal
}

// parsePrincipalKeys parses "callerType:key,callerType:key" into a map.
// Malformed entries (missing colon, empty type or key) are skipped. Entries
// whose caller-type is not a known/valid type are skipped + logged, so a typo
// like "reflectionn:key" cannot silently default to engram_user (Frank's
// collection).
func parsePrincipalKeys(raw string) map[string]string {
	if raw == "" {
		return nil
	}
	out := map[string]string{}
	for _, pair := range strings.Split(raw, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		ct, key, ok := strings.Cut(pair, ":")
		ct, key = strings.TrimSpace(ct), strings.TrimSpace(key)
		if !ok || ct == "" || key == "" {
			continue
		}
		if !collection.IsValidCallerType(ct) {
			log.Printf("WARN config: ENGRAM_PRINCIPAL_KEYS skipping unknown caller-type %q", ct)
			continue
		}
		out[ct] = key
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func envInt(key string, defaultVal int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return defaultVal
}

func envFloat(key string, defaultVal float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

func envDuration(key string, defaultVal time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return defaultVal
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		log.Printf("WARN: invalid duration for %s=%q, using default %v", key, v, defaultVal)
		return defaultVal
	}
	return d
}

func envBool(key string, defaultVal bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return defaultVal
}
