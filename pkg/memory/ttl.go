package memory

import "time"

// TTLConfig holds the TTL matrix: type × importance band → duration.
// Duration 0 means the memory never expires.
type TTLConfig struct {
	Rules map[MemoryType][3]time.Duration // [low (<5), mid (5-7), high (≥8)]
}

// DefaultTTLConfig returns the recommended TTL matrix.
//
// | type      | importance < 5 | importance 5-7 | importance ≥ 8 |
// |-----------|----------------|----------------|----------------|
// | identity  | permanent      | permanent      | permanent      |
// | directive | 90d            | permanent      | permanent      |
// | insight   | 30d            | 90d            | permanent      |
// | event     | 3d             | 7d             | 30d            |
func DefaultTTLConfig() TTLConfig {
	return TTLConfig{
		Rules: map[MemoryType][3]time.Duration{
			TypeIdentity:  {0, 0, 0},                                                            // permanent
			TypeDirective: {90 * 24 * time.Hour, 0, 0},                                          // 90d / perm / perm
			TypeInsight:   {30 * 24 * time.Hour, 90 * 24 * time.Hour, 0},                        // 30d / 90d / perm
			TypeEvent:     {3 * 24 * time.Hour, 7 * 24 * time.Hour, 30 * 24 * time.Hour},        // 3d / 7d / 30d
		},
	}
}

// importanceBand returns the index into the [3]Duration array:
//   0 = low  (importance < 5)
//   1 = mid  (5 <= importance < 8)
//   2 = high (importance >= 8)
func importanceBand(importance float64) int {
	switch {
	case importance >= 8:
		return 2
	case importance >= 5:
		return 1
	default:
		return 0
	}
}

// ComputeValidUntil calculates the valid_until timestamp for a memory.
//
// Rules:
//   - If the caller provided a non-zero valid_until, it is returned unchanged
//     (explicit override always wins).
//   - Otherwise the TTL matrix is consulted: type × importance band → duration.
//   - Duration 0 means permanent (returns 0).
//   - Special tag overrides:
//     "permanent" tag → returns 0 (never expires).
//     "time-sensitive" or "location" tag → forces 7-day TTL (unless permanent).
func ComputeValidUntil(cfg TTLConfig, memType MemoryType, importance float64, tags []string, explicitValidUntil float64) float64 {
	// Explicit override: caller already set valid_until
	if explicitValidUntil > 0 {
		return explicitValidUntil
	}

	// Tag-based overrides
	hasPermanent := false
	hasTimeSensitive := false
	for _, t := range tags {
		switch t {
		case "permanent":
			hasPermanent = true
		case "time-sensitive", "location":
			hasTimeSensitive = true
		}
	}

	if hasPermanent {
		return 0 // never expires
	}

	// Lookup TTL from matrix
	band := importanceBand(importance)
	durations, ok := cfg.Rules[memType]
	if !ok {
		// Unknown type: default to 90d for low, permanent for mid/high
		switch band {
		case 0:
			return float64(time.Now().Add(90 * 24 * time.Hour).Unix())
		default:
			return 0
		}
	}

	ttl := durations[band]
	if ttl == 0 {
		return 0 // permanent
	}

	// Tag override: time-sensitive/location forces 7d max
	if hasTimeSensitive {
		sevenDays := 7 * 24 * time.Hour
		if ttl > sevenDays {
			ttl = sevenDays
		}
	}

	return float64(time.Now().Add(ttl).Unix())
}
