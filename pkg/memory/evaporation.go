package memory

import (
	"math"
	"time"
)

// EvaporationConfig holds per-type evaporation parameters. It governs the
// runtime decay of importance (distinct from DecayConfig, which governs
// search-scoring recency). See memory-evaporation-spec-v1.md §4.1.
type EvaporationConfig struct {
	Enabled           bool                   `json:"enabled"`
	HalfLifeDays      map[MemoryType]float64 `json:"half_life_days"`
	AccessBoostAlpha  float64                `json:"access_boost_alpha"`
	EvictionThreshold float64                `json:"eviction_threshold"`
	SweepIntervalH    int                    `json:"sweep_interval_hours"`
	SweepBatchLimit   int                    `json:"sweep_batch_limit"`
}

// DefaultEvaporationConfig returns the recommended defaults. Feature-flagged
// off by default (Enabled=false) for backward compatibility.
func DefaultEvaporationConfig() EvaporationConfig {
	return EvaporationConfig{
		Enabled: false,
		HalfLifeDays: map[MemoryType]float64{
			TypeIdentity:  0, // 0 = no decay
			TypeEvent:     30,
			TypeInsight:   180,
			TypeDirective: 365,
		},
		AccessBoostAlpha:  0.15,
		EvictionThreshold: 1.0,
		SweepIntervalH:    6,
		SweepBatchLimit:   100,
	}
}

// EffectiveImportance computes the evaporation-adjusted importance:
//
//	eff = base_importance × e^(-λ·Δt) × (1 + α·ln(1+access_count))
//
// where λ = ln(2)/half_life and Δt is days since creation. When evaporation is
// disabled, or the type has no (or non-positive) half-life (e.g. identity), the
// raw importance is returned unchanged.
func EffectiveImportance(m *Memory, cfg EvaporationConfig) float64 {
	if !cfg.Enabled {
		return m.Importance
	}

	halfLife, ok := cfg.HalfLifeDays[m.Type]
	if !ok || halfLife <= 0 {
		return m.Importance // no decay for this type (identity)
	}

	lambda := math.Ln2 / halfLife
	daysPassed := math.Max(0, (float64(time.Now().Unix())-m.CreatedAt)/86400.0)

	decayFactor := math.Exp(-lambda * daysPassed)
	accessBoost := 1.0 + cfg.AccessBoostAlpha*math.Log(1.0+float64(m.AccessCount))

	effImp := m.Importance * decayFactor * accessBoost
	if effImp < 0 {
		effImp = 0
	}
	return effImp
}
