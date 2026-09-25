package server

import (
	"context"
	"fmt"
	"log/slog"
	rand "math/rand/v2"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/reflection"
)

// DefaultReflectionInterval is the default period between reflection trigger
// evaluations performed by the in-process scheduler.
const DefaultReflectionInterval = 20 * time.Minute

// reflectionSchedulerInitialDelay is a short warmup wait before the first
// evaluation so a process restart does not have to wait a full interval before
// a due reflection can start.
const reflectionSchedulerInitialDelay = 30 * time.Second

// StartReflectionScheduler launches a background goroutine that periodically
// evaluates the reflection trigger and, when due, starts a run through the
// exact same single-flight entry point (reflectionRunner.start) used by the
// reflection_run MCP tool. It does NOT bypass or duplicate the 2h min-interval
// or the 3-per-day cap: those gates live inside eng.Check and eng.Run, and
// single-flight lives inside start().
//
// The goroutine respects ctx cancellation and stops cleanly when ctx is done.
// interval controls how often the trigger is evaluated (default: 20 minutes);
// pass a positive interval[0] to override. In addition to the periodic ticks,
// one evaluation runs after a short initial delay so a restart does not wait a
// full interval.
func (s *Server) StartReflectionScheduler(ctx context.Context, interval ...time.Duration) {
	iv := DefaultReflectionInterval
	if len(interval) > 0 && interval[0] > 0 {
		iv = interval[0]
	}

	go func() {
		// Short warmup evaluation before the first tick, cancellable via ctx.
		initial := time.NewTimer(reflectionSchedulerInitialDelay)
		select {
		case <-ctx.Done():
			initial.Stop()
			fmt.Fprintf(os.Stderr, "[reflection-scheduler] goroutine stopped\n")
			return
		case <-initial.C:
			if !s.jitterWait(ctx, iv) {
				fmt.Fprintf(os.Stderr, "[reflection-scheduler] goroutine stopped\n")
				return
			}
			s.evaluateAndMaybeRun(ctx)
		}

		ticker := time.NewTicker(iv)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(os.Stderr, "[reflection-scheduler] goroutine stopped\n")
				return
			case <-ticker.C:
				if !s.jitterWait(ctx, iv) {
					fmt.Fprintf(os.Stderr, "[reflection-scheduler] goroutine stopped\n")
					return
				}
				s.evaluateAndMaybeRun(ctx)
			}
		}
	}()

	fmt.Fprintf(os.Stderr, "[reflection-scheduler] started (interval: %s)\n", iv)
}

// defaultReflectionWindow is the quiet-hour window during which the scheduler
// is allowed to trigger reflection runs. Overridable via ENGRAM_REFLECTION_WINDOW.
const defaultReflectionWindow = "22:00-01:00"

// parseReflectionWindow parses a "HH:MM-HH:MM" spec into minutes-since-midnight
// for start and end. PURE, no globals. Errors on malformed input.
func parseReflectionWindow(spec string) (startMin, endMin int, err error) {
	parts := strings.Split(spec, "-")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid window %q: want HH:MM-HH:MM", spec)
	}
	startMin, err = parseHHMM(parts[0])
	if err != nil {
		return 0, 0, err
	}
	endMin, err = parseHHMM(parts[1])
	if err != nil {
		return 0, 0, err
	}
	return startMin, endMin, nil
}

// parseHHMM parses "HH:MM" into minutes-since-midnight, validating ranges.
func parseHHMM(s string) (int, error) {
	hm := strings.Split(strings.TrimSpace(s), ":")
	if len(hm) != 2 {
		return 0, fmt.Errorf("invalid time %q: want HH:MM", s)
	}
	h, err := strconv.Atoi(hm[0])
	if err != nil || h < 0 || h > 23 {
		return 0, fmt.Errorf("invalid hour in %q", s)
	}
	m, err := strconv.Atoi(hm[1])
	if err != nil || m < 0 || m > 59 {
		return 0, fmt.Errorf("invalid minute in %q", s)
	}
	return h*60 + m, nil
}

// inReflectionWindow reports whether now (its minutes-since-midnight) falls in
// [startMin, endMin). Supports midnight wrap: if start>end the window spans
// midnight. start==end is treated as always-open (full day). End-exclusive.
func inReflectionWindow(now time.Time, startMin, endMin int) bool {
	nowMin := now.Hour()*60 + now.Minute()
	if startMin == endMin {
		return true // always open
	}
	if startMin < endMin {
		return nowMin >= startMin && nowMin < endMin
	}
	// Wrap across midnight.
	return nowMin >= startMin || nowMin < endMin
}

// reflectionWindowLocation returns the timezone used for the quiet window.
func reflectionWindowLocation() *time.Location {
	loc, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		return time.FixedZone("CST", 8*3600)
	}
	return loc
}

// defaultReflectionJitter is the default upper bound for the per-evaluation
// random jitter, overridable via ENGRAM_REFLECTION_JITTER (Go duration).
const defaultReflectionJitter = 5 * time.Minute

// jitterMax computes the jitter upper bound from an env value and the tick
// interval. PURE. Default 5m; parse errors / negatives fall back to the
// default; the result is capped at interval/2 so jitter never approaches a full
// interval.
func jitterMax(envVal string, iv time.Duration) time.Duration {
	max := defaultReflectionJitter
	if envVal != "" {
		if d, err := time.ParseDuration(strings.TrimSpace(envVal)); err == nil && d >= 0 {
			max = d
		}
	}
	if cap := iv / 2; max > cap {
		max = cap
	}
	if max < 0 {
		return 0
	}
	return max
}

// jitterWait sleeps a random duration in [0, jitterMax] before an evaluation,
// cancellable via ctx. Returns false if ctx was cancelled during the wait so
// the caller can stop the scheduler goroutine.
func (s *Server) jitterWait(ctx context.Context, iv time.Duration) bool {
	max := jitterMax(os.Getenv("ENGRAM_REFLECTION_JITTER"), iv)
	if max <= 0 {
		return ctx.Err() == nil
	}
	d := time.Duration(rand.Int64N(int64(max) + 1))
	slog.Info("reflection scheduler: jitter wait", "jitter", d.String(), "max", max.String())
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// evaluateAndMaybeRun checks the reflection trigger once and, when it fires,
// starts a run via the shared single-flight runner.
func (s *Server) evaluateAndMaybeRun(ctx context.Context) {
	// Pause gate: a paused scheduler returns WITHOUT evaluating. Manual
	// reflection_run is unaffected (it never calls this method).
	if paused, reason := reflection.SchedulerPaused(); paused {
		slog.Info("reflection scheduler: paused", "reason", reason)
		return
	}
	// Quiet-window gate: restrict scheduler triggering to a configured hour
	// window (default 22:00-01:00 Asia/Shanghai). A parse error fails OPEN so a
	// typo can't permanently disable reflection.
	spec := os.Getenv("ENGRAM_REFLECTION_WINDOW")
	if spec == "" {
		spec = defaultReflectionWindow
	}
	tz := reflectionWindowLocation()
	if startMin, endMin, err := parseReflectionWindow(spec); err != nil {
		slog.Warn("reflection scheduler: invalid window spec, running anyway (fail-open)",
			"window", spec, "error", err.Error())
	} else {
		now := time.Now().In(tz)
		if !inReflectionWindow(now, startMin, endMin) {
			slog.Info("reflection scheduler: outside quiet window",
				"window", spec, "now", now.Format("15:04"))
			return
		}
	}

	eng := reflection.NewEngine(s.store, s.embedder, s.reflectionConfig())
	res, err := eng.Check(ctx)
	if err != nil {
		slog.Error("reflection scheduler: check failed", "error", err.Error())
		return
	}
	if !res.ShouldTrigger {
		slog.Info("reflection scheduler: not due", "skip_reason", res.SkipReason)
		return
	}

	// These closures intentionally mirror handleReflectionRun (server.go:1502)
	// so the scheduler and the reflection_run MCP tool share the exact same
	// code path: the shared entry point is reflectionRunner.start.
	defaultRun := func(rctx context.Context) (*reflection.RunResult, error) {
		rcfg := s.reflectionConfig()
		rcfg.DryRun = false
		eng := reflection.NewEngine(s.store, s.embedder, rcfg)
		return eng.Run(rctx)
	}
	recordMetrics := func(result *reflection.RunResult) {
		if s.metrics != nil && result.Triggered {
			s.metrics.ReflectionRuns.WithLabelValues(result.Mode, "default").Inc()
			s.metrics.ReflectionInsightsCreated.WithLabelValues(result.Mode, "high").Add(float64(result.LLMConfHighCount))
			s.metrics.ReflectionInsightsCreated.WithLabelValues(result.Mode, "mid").Add(float64(result.LLMConfMidCount))
			s.metrics.ReflectionInsightsCreated.WithLabelValues(result.Mode, "low").Add(float64(result.LLMConfLowCount))
		}
	}

	started, runID, _ := s.reflectionRunner.start(defaultRun, recordMetrics)
	if started {
		slog.Info("reflection scheduler started run", "run_id", runID)
		return
	}
	slog.Info("reflection scheduler: run already in flight / too soon", "run_id", runID)
}
