package server

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
)

// StartEvaporationSweep launches a background goroutine that periodically
// deprecates memories whose effective_importance has decayed below the
// eviction threshold (spec §3.2.2). It mirrors StartExpiryCleanup: the
// goroutine respects ctx cancellation and stops cleanly when ctx is done.
//
// Unlike a static-config start, the ticker is ALWAYS started (bounded) and each
// tick re-reads the server's CURRENT effective config via evaporationConfig(),
// so memory_apply_config hot-reload (spec §4.3) can enable/tune the sweep at
// runtime. Ticks are no-ops while evaporation is disabled — preserving the
// feature-flag-off backward-compat behavior. Metrics are recorded through the
// server's *Metrics handle when present (nil-safe for the stdio-only path).
func (s *Server) StartEvaporationSweep(ctx context.Context) {
	interval := s.sweepInterval()

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		cur := interval

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(os.Stderr, "[evaporation] sweep goroutine stopped\n")
				return
			case t := <-ticker.C:
				s.evaporationSweepTick(ctx, t)
				// Honor a hot-reloaded SweepIntervalH by resetting the ticker cadence.
				if next := s.sweepInterval(); next != cur {
					ticker.Reset(next)
					cur = next
					fmt.Fprintf(os.Stderr, "[evaporation] sweep interval reset to %s\n", next)
				}
			}
		}
	}()

	fmt.Fprintf(os.Stderr, "[evaporation] sweep goroutine started (interval: %s)\n", interval)
}

// sweepInterval returns the current effective sweep interval, defaulting to 6h
// when SweepIntervalH is non-positive.
func (s *Server) sweepInterval() time.Duration {
	interval := time.Duration(s.evaporationConfig().SweepIntervalH) * time.Hour
	if interval <= 0 {
		interval = 6 * time.Hour
	}
	return interval
}

// evaporationSweepTick runs one sweep using the server's CURRENT effective
// config. It is a no-op when evaporation is disabled. Returns
// (deprecatedCount, skippedCount) for testability.
func (s *Server) evaporationSweepTick(ctx context.Context, t time.Time) (int, int) {
	cfg := s.evaporationConfig()
	if !cfg.Enabled {
		return 0, 0
	}
	deprecated, skipped, err := runEvaporationSweep(ctx, s.store, cfg, s.metrics)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[evaporation] sweep error at %s: %v\n", t.Format(time.RFC3339), err)
		return deprecated, skipped
	}
	fmt.Fprintf(os.Stderr, "[evaporation] sweep at %s: deprecated=%d skipped=%d\n",
		t.Format(time.RFC3339), deprecated, skipped)
	return deprecated, skipped
}

// runEvaporationSweep scans active-lifecycle memories, computes their effective
// importance, and deprecates those below cfg.EvictionThreshold (up to
// cfg.SweepBatchLimit). Returns (deprecatedCount, skippedCount, err).
func runEvaporationSweep(ctx context.Context, store memory.Store, cfg memory.EvaporationConfig, m *engrammetrics.Metrics) (int, int, error) {
	start := time.Now()
	if m != nil {
		m.EvaporationSweepTotal.Inc()
		defer func() { m.EvaporationSweepDuration.Observe(time.Since(start).Seconds()) }()
	}

	batchLimit := cfg.SweepBatchLimit
	if batchLimit <= 0 {
		batchLimit = 100
	}

	deprecated := 0
	skipped := 0
	var offset string

	for deprecated < batchLimit {
		mems, nextOffset, err := store.Scroll(ctx, memory.ScrollOptions{Limit: 100, Offset: offset})
		if err != nil {
			return deprecated, skipped, fmt.Errorf("scroll: %w", err)
		}

		for i := range mems {
			mem := &mems[i]
			// Only consider active-lifecycle memories. Empty status is treated
			// as active for backward compatibility.
			if mem.LifecycleStatus != "" && mem.LifecycleStatus != memory.LifecycleActive {
				continue
			}
			eff := memory.EffectiveImportance(mem, cfg, time.Now())
			if eff >= cfg.EvictionThreshold {
				continue
			}

			meta := make(map[string]any, len(mem.Metadata)+3)
			for k, v := range mem.Metadata {
				meta[k] = v
			}
			now := float64(time.Now().Unix())
			meta["deprecated_reason"] = "evaporation"
			meta["deprecated_at"] = now
			meta["effective_importance_at_deprecation"] = eff

			fields := map[string]any{
				"lifecycle_status": memory.LifecycleDeprecated,
				"metadata":         meta,
			}
			if err := store.Update(ctx, mem.ID, fields); err != nil {
				fmt.Fprintf(os.Stderr, "[evaporation] update failed for %s: %v\n", mem.ID, err)
				skipped++
				continue
			}
			deprecated++
			if m != nil {
				m.EvaporationDeprecatedTotal.WithLabelValues(string(mem.Type)).Inc()
			}
			if deprecated >= batchLimit {
				return deprecated, skipped, nil
			}
		}

		if nextOffset == "" {
			break
		}
		offset = nextOffset
	}

	return deprecated, skipped, nil
}
