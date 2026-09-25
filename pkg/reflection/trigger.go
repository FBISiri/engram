package reflection

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/FBISiri/engram/pkg/statedir"
)

const (
	reflectionLastRunFile = "reflection_last_run"
	reflectionMaxPerDay   = 3
	reflectionDailyFile   = "reflection_daily_count"

	// Failure-aware backoff state (separate mechanism from LLM retry/backoff).
	reflectionLastAttemptFile  = "reflection_last_attempt"         // RFC3339 timestamp
	reflectionFailureCountFile = "reflection_consecutive_failures" // integer text

	// failureBackoffBase / failureBackoffCap parameterise computeFailureBackoff.
	failureBackoffBase = 40 * time.Minute
	failureBackoffCap  = 6 * time.Hour

	// Circuit-breaker knobs (separate from failure backoff): after a threshold
	// of consecutive failures the scheduler stops evaluating (open) until a
	// cooldown elapses, then allows ONE probe run.
	envBreakerThreshold     = "ENGRAM_REFLECTION_BREAKER_THRESHOLD"
	envBreakerCooldownH     = "ENGRAM_REFLECTION_BREAKER_COOLDOWN_H"
	defaultBreakerThreshold = 4
	defaultBreakerCooldownH = 24.0

	// reflectionPausedFile pauses ONLY the periodic scheduler when present.
	reflectionPausedFile = "reflection_paused"
)

// siriDirPath returns the Siri state dir, creating it if needed.
func siriDirPath() (string, error) {
	return statedir.Dir()
}

// CheckResult is the output of a trigger check.
type CheckResult struct {
	ShouldTrigger           bool    `json:"should_trigger"`
	SkipReason              string  `json:"skip_reason,omitempty"`
	UnreflectedCount        int     `json:"unreflected_count"`
	AccumulatedImportance   float64 `json:"accumulated_importance"`
	Threshold               float64 `json:"threshold"`
	HoursSinceLastRun       float64 `json:"hours_since_last_run"`
	RunsToday               int     `json:"runs_today"`
	ConsecutiveFailures     int     `json:"consecutive_failures"`
	FailureBackoffRemaining float64 `json:"failure_backoff_remaining_minutes,omitempty"`
	CircuitOpen             bool    `json:"circuit_open"`
}

// check evaluates whether reflection should run now.
// Returns (shouldTrigger, skipReason, unreflectedMems, accumulatedImportance, error).
func (e *Engine) check(ctx context.Context) (*CheckResult, []memory.Memory, error) {
	dir, err := siriDirPath()
	if err != nil {
		return nil, nil, err
	}

	result := &CheckResult{
		Threshold: e.cfg.Threshold,
	}

	// Read daily count early so CheckResult.RunsToday is always populated,
	// even when early gates block execution.
	dailyCount, err := readDailyCount(filepath.Join(dir, reflectionDailyFile))
	if err != nil {
		// Non-fatal: treat as 0.
		dailyCount = 0
	}
	result.RunsToday = dailyCount

	// Read consecutive-failure count early so ConsecutiveFailures is always
	// populated for observability even when a gate blocks execution.
	failCount, err := readFailureCount(filepath.Join(dir, reflectionFailureCountFile))
	if err != nil {
		failCount = 0
	}
	result.ConsecutiveFailures = failCount

	// Gate 1: Time interval check.
	lastRunPath := filepath.Join(dir, reflectionLastRunFile)
	lastRunTime, err := readTimestampFile(lastRunPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read last run: %w", err)
	}
	gate1Blocked := false
	if !lastRunTime.IsZero() {
		result.HoursSinceLastRun = time.Since(lastRunTime).Hours()
		minInterval := time.Duration(e.cfg.MinIntervalH * float64(time.Hour))
		if time.Since(lastRunTime) < minInterval && !e.cfg.Force {
			result.SkipReason = fmt.Sprintf("too soon: last run %.1fh ago (min interval %.1fh)",
				result.HoursSinceLastRun, e.cfg.MinIntervalH)
			gate1Blocked = true
		}
	}

	// Gate 2: Daily run count check (count already loaded above).
	gate2Blocked := dailyCount >= reflectionMaxPerDay && !e.cfg.Force

	// Gate 3: Importance accumulation — always computed so AccumulatedImportance
	// is accurate even when Gate 1 or Gate 2 blocks ShouldTrigger. Tests and
	// callers rely on this field for observability regardless of trigger outcome.
	unreflected, err := e.fetchUnreflected(ctx, 200)
	if err != nil {
		return nil, nil, fmt.Errorf("fetch unreflected: %w", err)
	}
	result.UnreflectedCount = len(unreflected)

	var total float64
	for _, m := range unreflected {
		total += m.Importance
	}
	result.AccumulatedImportance = total

	// Apply Gate 1 / Gate 2 early-return now that accumulated_importance is set.
	if gate1Blocked {
		return result, nil, nil
	}

	// Circuit-breaker gate: after a threshold of consecutive failures, stop
	// evaluating entirely (open) until a cooldown elapses, then allow ONE probe
	// run to fall through to the normal gates. Placed BEFORE the failure-backoff
	// gate so it wins. Force bypasses it exactly like the other gates.
	probing := false
	if !e.cfg.Force {
		threshold := breakerThreshold()
		cooldown := breakerCooldown()
		lastAttempt, _ := readTimestampFile(filepath.Join(dir, reflectionLastAttemptFile))
		open, probeAllowed := circuitState(failCount, threshold, lastAttempt, cooldown, time.Now())
		if open {
			result.CircuitOpen = true
			if !probeAllowed {
				result.SkipReason = fmt.Sprintf("circuit open: %d consecutive failures >= threshold %d; reset %s to 0 or wait cooldown %s",
					failCount, threshold, filepath.Join(dir, reflectionFailureCountFile), cooldown)
				return result, nil, nil
			}
			// Half-open: allow ONE probe run past the failure-backoff gate. A
			// failed probe re-stamps last_attempt (recordFailure), so the next
			// probe waits another cooldown; success resets the counter to 0.
			probing = true
		}
	}

	// Failure backoff gate: after a run failed (all-429 etc.), lengthen the
	// retry cadence so the scheduler stops hammering every interval. This is a
	// SEPARATE mechanism from the LLM transient retry/backoff. Force bypasses it
	// exactly like the other gates.
	if failCount > 0 && !e.cfg.Force && !probing {
		lastAttempt, _ := readTimestampFile(filepath.Join(dir, reflectionLastAttemptFile))
		backoff := computeFailureBackoff(failCount)
		if !lastAttempt.IsZero() && time.Since(lastAttempt) < backoff {
			remaining := backoff - time.Since(lastAttempt)
			result.FailureBackoffRemaining = remaining.Minutes()
			result.SkipReason = fmt.Sprintf("failure backoff: %d consecutive failures, waiting %s (%.1fm remaining)",
				failCount, backoff, remaining.Minutes())
			return result, nil, nil
		}
	}

	if gate2Blocked {
		result.SkipReason = fmt.Sprintf("daily limit reached: %d/%d runs today", dailyCount, reflectionMaxPerDay)
		return result, nil, nil
	}

	if total < e.cfg.Threshold {
		result.SkipReason = fmt.Sprintf("importance accumulation %.1f < threshold %.1f (%d unreflected memories)",
			total, e.cfg.Threshold, len(unreflected))
		return result, unreflected, nil
	}

	result.ShouldTrigger = true
	return result, unreflected, nil
}

// fetchUnreflected returns memories where reflected_at is empty (not yet reflected).
// Uses the Qdrant-indexed reflected_at field via IsEmpty filter for O(1) lookup (W16 Phase 3).
// Falls back to legacy metadata["reflected"] check for memories created before the W16 migration.
func (e *Engine) fetchUnreflected(ctx context.Context, limit int) ([]memory.Memory, error) {
	// Use indexed reflected_at IsEmpty filter — only returns memories where
	// reflected_at payload field does not exist or is null.
	all, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: limit,
		Filters: []memory.Filter{
			{
				Field: "reflected_at",
				Op:    memory.OpIsEmpty,
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("scroll unreflected: %w", err)
	}

	// Secondary filter: exclude memories that have legacy metadata["reflected"]=true
	// but were not yet migrated to the reflected_at field.
	var unreflected []memory.Memory
	for _, m := range all {
		if !isReflected(m) {
			unreflected = append(unreflected, m)
		}
	}
	return unreflected, nil
}

// isReflected returns true if the memory has been marked as reflected.
// Checks ReflectedAt > 0 first (W16 new field), then falls back to
// metadata["reflected"] bool (legacy V1 field) for backward compatibility.
func isReflected(m memory.Memory) bool {
	// V2: check ReflectedAt timestamp field (preferred).
	if m.ReflectedAt > 0 {
		return true
	}
	// V1 fallback: check metadata["reflected"] bool.
	if m.Metadata == nil {
		return false
	}
	v, ok := m.Metadata["reflected"]
	if !ok {
		return false
	}
	b, ok := v.(bool)
	return ok && b
}

// updateLastRun writes the current time to the last-run file and increments daily count.
func updateLastRun() error {
	dir, err := siriDirPath()
	if err != nil {
		return err
	}

	// Write last-run timestamp.
	lastRunPath := filepath.Join(dir, reflectionLastRunFile)
	if err := os.WriteFile(lastRunPath, []byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		return fmt.Errorf("write last run: %w", err)
	}

	// Increment daily count.
	dailyPath := filepath.Join(dir, reflectionDailyFile)
	count, _ := readDailyCount(dailyPath)
	count++
	if err := writeDailyCount(dailyPath, count); err != nil {
		return fmt.Errorf("write daily count: %w", err)
	}

	// Success clears any accumulated failure backoff (R3): reset consecutive
	// failure count to 0 and stamp last-attempt = now.
	if err := writeFailureCount(filepath.Join(dir, reflectionFailureCountFile), 0); err != nil {
		return fmt.Errorf("reset failure count: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, reflectionLastAttemptFile),
		[]byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		return fmt.Errorf("write last attempt: %w", err)
	}

	return nil
}

// readFailureCount reads the consecutive-failure count from file.
// Missing file → 0; unparseable → 0.
func readFailureCount(path string) (int, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	var n int
	if _, err := fmt.Sscanf(strings.TrimSpace(string(data)), "%d", &n); err != nil {
		return 0, nil
	}
	return n, nil
}

// writeFailureCount writes the consecutive-failure count to file.
func writeFailureCount(path string, n int) error {
	return os.WriteFile(path, []byte(fmt.Sprintf("%d\n", n)), 0644)
}

// recordFailure increments the consecutive-failure count and stamps the
// last-attempt time. Called when a reflection run produced no insights (all
// failed), so the failure-backoff gate lengthens the retry cadence.
func recordFailure() error {
	dir, err := siriDirPath()
	if err != nil {
		return err
	}
	countPath := filepath.Join(dir, reflectionFailureCountFile)
	n, _ := readFailureCount(countPath)
	n++
	if err := writeFailureCount(countPath, n); err != nil {
		return fmt.Errorf("write failure count: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, reflectionLastAttemptFile),
		[]byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		return fmt.Errorf("write last attempt: %w", err)
	}
	return nil
}

// finalizeFailureAccounting is the SINGLE unified exit point for failure-backoff
// accounting. Run/RunV2 defer it so EVERY error early-return path (Stage 1
// generateFocalQuestions failure, dialectic failure, "no insights produced",
// LLM call failure, unparseable insights, …) funnels through one place instead
// of scattered recordFailure() calls that the dominant 429 path bypassed.
//
// Classification:
//   - result==nil or !Triggered  → no-op. A gated skip (min-interval, daily
//     cap, below-threshold, no memories, check error) is neither success nor
//     failure and must not touch the counter.
//   - succeeded==true            → no-op. The run marked sources and
//     updateLastRun() already reset the counter to 0.
//   - DryRun                      → no-op. A diagnostic dry run (never the
//     scheduler, which sets DryRun=false) must not corrupt the real backoff
//     state even if its Stage 1 call hits a 429.
//   - Triggered, not succeeded, and produced ≥1 error → count it as a
//     consecutive failure and stamp last_attempt so the backoff gate lengthens
//     the cadence. A benign post-trigger skip that sets SkipReason but no Errors
//     (e.g. "no unreflected memories available") is NOT a failure.
func finalizeFailureAccounting(result *RunResult, succeeded bool) {
	if result == nil || !result.Triggered || succeeded || result.DryRun {
		return
	}
	if len(result.Errors) == 0 {
		return
	}
	if err := recordFailure(); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("record failure failed: %v", err))
		return
	}
	// Active alert layer: read the freshly-incremented count and emit a signal
	// with the next backoff and the most recent rate-limit snapshot so a degraded
	// run is visible without scraping counters.
	dir, derr := siriDirPath()
	if derr != nil {
		return
	}
	countPath := filepath.Join(dir, reflectionFailureCountFile)
	n, _ := readFailureCount(countPath)
	// Fast-trip: credential/billing errors (401/402/403) are non-transient by
	// nature; open the breaker immediately (bump count to >= threshold) instead
	// of climbing the exponential backoff ladder. Detection is string-based:
	// RunResult carries errors as strings only, so we match the exact prefix
	// "llm returned status <code>" that (*llm.StatusError).Error() produces.
	if code := creditErrorStatus(result.Errors); code != 0 {
		threshold := breakerThreshold()
		if n < threshold {
			n = threshold
			if werr := writeFailureCount(countPath, n); werr != nil {
				result.Errors = append(result.Errors, fmt.Sprintf("fast-trip write failed: %v", werr))
			}
		}
		log.Printf("[reflection][ALERT] fast-trip: llm status %d (credential/billing) — opening circuit at %d failures (threshold %d)",
			code, n, threshold)
	}
	log.Printf("[reflection][ALERT] reflection run failed: %d consecutive failures, next backoff %s (last_ratelimit=%s)",
		n, computeFailureBackoff(n), formatLastRateLimit())
}

// creditErrorStatus scans reflection error strings for a non-transient
// credential/billing LLM status (401/402/403). PURE. RunResult carries errors
// as strings, not wrapped error values, so it matches the exact prefix
// "llm returned status <code>" emitted by (*llm.StatusError).Error(). Returns
// the lowest credit-error status code found, or 0 when none match (429/5xx are
// transient and deliberately excluded so they keep climbing the backoff ladder).
func creditErrorStatus(errs []string) int {
	for _, code := range []int{401, 402, 403} {
		needle := fmt.Sprintf("llm returned status %d", code)
		for _, e := range errs {
			if strings.Contains(e, needle) {
				return code
			}
		}
	}
	return 0
}

// breakerThreshold reads ENGRAM_REFLECTION_BREAKER_THRESHOLD (default 4). Values
// < 1 or unparseable fall back to the default.
func breakerThreshold() int {
	v := strings.TrimSpace(os.Getenv(envBreakerThreshold))
	if v == "" {
		return defaultBreakerThreshold
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 1 {
		return defaultBreakerThreshold
	}
	return n
}

// breakerCooldown reads ENGRAM_REFLECTION_BREAKER_COOLDOWN_H (float hours,
// default 24). Negative or unparseable falls back to the default.
func breakerCooldown() time.Duration {
	v := strings.TrimSpace(os.Getenv(envBreakerCooldownH))
	if v == "" {
		return time.Duration(defaultBreakerCooldownH * float64(time.Hour))
	}
	h, err := strconv.ParseFloat(v, 64)
	if err != nil || h < 0 {
		return time.Duration(defaultBreakerCooldownH * float64(time.Hour))
	}
	return time.Duration(h * float64(time.Hour))
}

// circuitState reports the breaker state. PURE (no I/O). The circuit is OPEN
// when failCount >= threshold. When open, exactly one probe is allowed once the
// cooldown has elapsed since lastAttempt (a zero lastAttempt is treated as
// cooldown-elapsed so the breaker can never latch permanently). threshold < 1
// is clamped to 1.
func circuitState(failCount, threshold int, lastAttempt time.Time, cooldown time.Duration, now time.Time) (open bool, probeAllowed bool) {
	if threshold < 1 {
		threshold = 1
	}
	open = failCount >= threshold
	if !open {
		return false, false
	}
	if lastAttempt.IsZero() || now.Sub(lastAttempt) >= cooldown {
		probeAllowed = true
	}
	return open, probeAllowed
}

// SchedulerPaused reports whether the periodic reflection scheduler is paused,
// via either ENGRAM_REFLECTION_PAUSED=1|true or a <stateDir>/reflection_paused
// file (stateDir resolved exactly like siriDirPath). It affects ONLY the
// scheduler; the manual reflection_run MCP tool is unaffected. The second
// return value is a human-readable reason for logging.
func SchedulerPaused() (bool, string) {
	if v := strings.TrimSpace(os.Getenv("ENGRAM_REFLECTION_PAUSED")); v == "1" || strings.EqualFold(v, "true") {
		return true, "env ENGRAM_REFLECTION_PAUSED=" + v
	}
	dir, err := siriDirPath()
	if err != nil {
		return false, ""
	}
	p := filepath.Join(dir, reflectionPausedFile)
	if _, err := os.Stat(p); err == nil {
		return true, "file " + p
	}
	return false, ""
}

// computeFailureBackoff returns the failure-backoff duration for n consecutive
// failures. PURE function (no I/O). n<=0 → 0. Exponential: base 40m doubling
// each additional failure, capped at 6h. n=1→40m, n=2→80m, n=3→160m,
// n=4→320m, n>=5→capped 6h.
func computeFailureBackoff(n int) time.Duration {
	if n <= 0 {
		return 0
	}
	backoff := failureBackoffBase
	for i := 1; i < n; i++ {
		backoff *= 2
		if backoff >= failureBackoffCap {
			return failureBackoffCap
		}
	}
	if backoff > failureBackoffCap {
		return failureBackoffCap
	}
	return backoff
}

// readTimestampFile reads a RFC3339 timestamp from a file. Returns zero time if not found.
func readTimestampFile(path string) (time.Time, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, err
	}
	t, err := time.Parse(time.RFC3339, string(data))
	if err != nil {
		// Try mtime fallback.
		info, statErr := os.Stat(path)
		if statErr != nil {
			return time.Time{}, nil
		}
		return info.ModTime(), nil
	}
	return t, nil
}

// cstLocation is the Asia/Shanghai timezone used for daily count boundaries.
// Siri operates in CST (+8), so "today" must align with CST midnight, not UTC.
// Without this, runs at 7am CST (= 23:xx UTC previous day) are attributed to
// the previous UTC date and appear as runs_today=0 on the next UTC morning.
var cstLocation = func() *time.Location {
	loc, err := time.LoadLocation("Asia/Shanghai")
	if err != nil {
		return time.FixedZone("CST", 8*60*60)
	}
	return loc
}()

// readDailyCount reads today's reflection run count from file.
// Returns 0 if file doesn't exist or is from a previous day.
func readDailyCount(path string) (int, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}

	today := time.Now().In(cstLocation).Format("2006-01-02")
	// Format: "DATE COUNT\n"
	var date string
	var count int
	if _, err := fmt.Sscanf(string(data), "%s %d", &date, &count); err != nil {
		return 0, nil
	}
	if date != today {
		return 0, nil // New day, reset.
	}
	return count, nil
}

// filterExistingIDs returns only IDs from the provided list that still exist
// in the store right now. IDs not found (deleted by TTL expiry, consolidation,
// or manual cleanup since fetchUnreflected ran) are omitted and their count
// reported as orphanCount.
//
// This guards the mark-reflected step against the TOCTOU race: memories
// deleted between fetchUnreflected and store.Update can't be marked, so
// skipping them is correct — the memories are already gone.
//
// On SearchByIDs failure, all IDs are returned unchanged (fail-open) so a
// transient lookup error doesn't silently drop source marks.
func filterExistingIDs(ctx context.Context, store memory.Store, ids []string) (existing []string, orphanCount int, err error) {
	if len(ids) == 0 {
		return nil, 0, nil
	}
	found, err := store.SearchByIDs(ctx, ids)
	if err != nil {
		return ids, 0, err // fail-open: caller uses all IDs
	}
	foundSet := make(map[string]bool, len(found))
	for _, m := range found {
		foundSet[m.ID] = true
	}
	for _, id := range ids {
		if foundSet[id] {
			existing = append(existing, id)
		} else {
			orphanCount++
		}
	}
	return existing, orphanCount, nil
}

// writeDailyCount writes today's count to file.
func writeDailyCount(path string, count int) error {
	today := time.Now().In(cstLocation).Format("2006-01-02")
	return os.WriteFile(path, []byte(fmt.Sprintf("%s %d\n", today, count)), 0644)
}
