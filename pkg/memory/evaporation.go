package memory

import (
	"fmt"
	"math"
	"os"
	"sync"
	"time"
)

// EvaporationConfig holds per-type evaporation parameters. It governs the
// runtime decay of importance (distinct from DecayConfig, which governs
// search-scoring recency). See memory-evaporation-spec-v1.md §4.1 and the v2
// safety spec (spec-memory-evaporation.md §4).
type EvaporationConfig struct {
	Enabled           bool                   `json:"enabled"`
	HalfLifeDays      map[MemoryType]float64 `json:"half_life_days"`
	AccessBoostAlpha  float64                `json:"access_boost_alpha"`
	EvictionThreshold float64                `json:"eviction_threshold"`
	SweepIntervalH    int                    `json:"sweep_interval_hours"`
	SweepBatchLimit   int                    `json:"sweep_batch_limit"`

	// v2 safety fields (spec §4).
	DryRun     bool   `json:"dry_run"`     // when true the sweep performs no store.Update
	DecayBasis string `json:"decay_basis"` // "last_access" (default) | "created"

	MinAgeDays      float64 `json:"min_age_days"`     // P8: never evaporate younger than this
	ObservationDays float64 `json:"observation_days"` // window between deprecate and delete-eligible

	// Protection rule parameters (P1..P7).
	ProtectImportance       float64      `json:"protect_importance"`         // P2
	ProtectRecentAccessDays float64      `json:"protect_recent_access_days"` // P4
	ProtectAccessCount      int          `json:"protect_access_count"`       // P3
	ProtectTypes            []MemoryType `json:"protect_types"`              // P1
	ProtectTags             []string     `json:"protect_tags"`               // P5
	ProtectCorroborated     bool         `json:"protect_corroborated"`       // P7
}

// DefaultEvaporationConfig returns the recommended defaults. Feature-flagged
// off by default (Enabled=false) for backward compatibility. dry_run defaults
// true (spec §4.5): enabling evaporation lands in observe-only mode until an
// operator explicitly sets dry_run=false.
func DefaultEvaporationConfig() EvaporationConfig {
	return EvaporationConfig{
		Enabled: false,
		HalfLifeDays: map[MemoryType]float64{
			TypeIdentity:  0, // 0 = no decay (also structurally protected by P1)
			TypeEvent:     30,
			TypeInsight:   180,
			TypeDirective: 0, // v2: was 365 — directives never evaporate (spec §4.3)
		},
		AccessBoostAlpha:  0.15,
		EvictionThreshold: 1.0,
		SweepIntervalH:    6,
		SweepBatchLimit:   100,

		DryRun:          true,
		DecayBasis:      "last_access",
		MinAgeDays:      14,
		ObservationDays: 30,

		ProtectImportance:       8,
		ProtectRecentAccessDays: 30,
		ProtectAccessCount:      5,
		ProtectTypes:            []MemoryType{TypeIdentity, TypeDirective},
		ProtectTags:             []string{"permanent", "frank-feedback", "directive", "identity"},
		ProtectCorroborated:     true,
	}
}

var decayBasisWarnOnce sync.Once

// referenceTime returns the reinforcement-clock reference for decay (spec §4.2).
// Under DecayBasis="last_access" (default) decay runs from the later of
// created_at / last_accessed_at, so a memory that keeps getting read stays
// alive. Under "created" (legacy) it runs from created_at only. An
// invalid/empty basis falls back to last_access with a one-time stderr warning.
func referenceTime(m *Memory, basis string) float64 {
	switch basis {
	case "created":
		return m.CreatedAt
	case "last_access":
		// handled below
	default:
		decayBasisWarnOnce.Do(func() {
			fmt.Fprintf(os.Stderr, "[evaporation] invalid ENGRAM_EVAPORATION_DECAY_BASIS %q; falling back to \"last_access\"\n", basis)
		})
	}
	if m.LastAccessedAt > m.CreatedAt {
		return m.LastAccessedAt
	}
	return m.CreatedAt
}

// EffectiveImportance computes the evaporation-adjusted importance:
//
//	eff = base_importance × e^(-λ·Δt) × (1 + α·ln(1+access_count))
//
// where λ = ln(2)/half_life and Δt is days since the reinforcement-clock
// reference (spec §4.2). When evaporation is disabled, or the type has no (or
// non-positive) half-life (e.g. identity/directive), the raw importance is
// returned unchanged.
func EffectiveImportance(m *Memory, cfg EvaporationConfig, now time.Time) float64 {
	if !cfg.Enabled {
		return m.Importance
	}

	halfLife, ok := cfg.HalfLifeDays[m.Type]
	if !ok || halfLife <= 0 {
		return m.Importance // no decay for this type (identity/directive)
	}

	lambda := math.Ln2 / halfLife
	tRef := referenceTime(m, cfg.DecayBasis)
	daysPassed := math.Max(0, (float64(now.Unix())-tRef)/86400.0)

	decayFactor := math.Exp(-lambda * daysPassed)
	accessBoost := 1.0 + cfg.AccessBoostAlpha*math.Log(1.0+float64(m.AccessCount))

	effImp := m.Importance * decayFactor * accessBoost
	if effImp < 0 {
		effImp = 0
	}
	return effImp
}

// EvaporationExempt reports whether m must be protected from evaporation,
// returning (true, ruleID) on the FIRST matching protection rule P1..P8
// (spec §4.1). Bias: prefer under-deleting to over-deleting. P1 is STRUCTURAL —
// a protected type is exempt regardless of any configured half-life, so a
// config typo can never make identity/directive memories evaporable.
func EvaporationExempt(m *Memory, cfg EvaporationConfig, now time.Time) (bool, string) {
	// P1 — protected type (structural; independent of half-life config).
	for _, t := range cfg.ProtectTypes {
		if m.Type == t {
			return true, "P1"
		}
	}

	// P2 — importance floor.
	if cfg.ProtectImportance > 0 && m.Importance >= cfg.ProtectImportance {
		return true, "P2"
	}

	// P3 — access-count floor.
	if cfg.ProtectAccessCount > 0 && m.AccessCount >= int64(cfg.ProtectAccessCount) {
		return true, "P3"
	}

	// P4 — recently accessed.
	if cfg.ProtectRecentAccessDays > 0 && m.LastAccessedAt > 0 {
		ageDays := (float64(now.Unix()) - m.LastAccessedAt) / 86400.0
		if ageDays <= cfg.ProtectRecentAccessDays {
			return true, "P4"
		}
	}

	// P5 — protected tag.
	for _, tag := range cfg.ProtectTags {
		for _, mt := range m.Tags {
			if mt == tag {
				return true, "P5"
			}
		}
	}

	// P6 — superseded records are owned by consolidation, not evaporation.
	if m.SupersededBy != "" {
		return true, "P6"
	}

	// P7 — corroborated (non-empty provenance_history from dedup merges).
	if cfg.ProtectCorroborated && hasProvenanceHistory(m) {
		return true, "P7"
	}

	// P8 — too young.
	if cfg.MinAgeDays > 0 {
		ageDays := (float64(now.Unix()) - m.CreatedAt) / 86400.0
		if ageDays < cfg.MinAgeDays {
			return true, "P8"
		}
	}

	return false, ""
}

// hasProvenanceHistory reports whether m carries a non-empty provenance_history
// in its metadata (set when consolidation/dedup merged duplicates into it).
func hasProvenanceHistory(m *Memory) bool {
	if m.Metadata == nil {
		return false
	}
	v, ok := m.Metadata["provenance_history"]
	if !ok || v == nil {
		return false
	}
	switch t := v.(type) {
	case []any:
		return len(t) > 0
	case []string:
		return len(t) > 0
	case string:
		return t != ""
	default:
		return true // present, non-nil, unknown shape → treat as corroborated
	}
}
