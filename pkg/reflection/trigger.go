package reflection

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

const (
	reflectionLastRunFile = "reflection_last_run"
	reflectionMaxPerDay   = 3
	reflectionDailyFile   = "reflection_daily_count"
)

// siriDirPath returns ~/.siri, creating it if needed.
func siriDirPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	dir := filepath.Join(home, ".siri")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create .siri dir: %w", err)
	}
	return dir, nil
}

// CheckResult is the output of a trigger check.
type CheckResult struct {
	ShouldTrigger       bool    `json:"should_trigger"`
	SkipReason          string  `json:"skip_reason,omitempty"`
	UnreflectedCount    int     `json:"unreflected_count"`
	AccumulatedImportance float64 `json:"accumulated_importance"`
	Threshold           float64 `json:"threshold"`
	HoursSinceLastRun   float64 `json:"hours_since_last_run"`
	RunsToday           int     `json:"runs_today"`
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

	// Gate 1: Time interval check.
	lastRunPath := filepath.Join(dir, reflectionLastRunFile)
	lastRunTime, err := readTimestampFile(lastRunPath)
	if err != nil {
		return nil, nil, fmt.Errorf("read last run: %w", err)
	}
	if !lastRunTime.IsZero() {
		result.HoursSinceLastRun = time.Since(lastRunTime).Hours()
		minInterval := time.Duration(e.cfg.MinIntervalH * float64(time.Hour))
		if time.Since(lastRunTime) < minInterval {
			result.SkipReason = fmt.Sprintf("too soon: last run %.1fh ago (min interval %.1fh)",
				result.HoursSinceLastRun, e.cfg.MinIntervalH)
			return result, nil, nil
		}
	}

	// Gate 2: Daily run count check.
	dailyCount, err := readDailyCount(filepath.Join(dir, reflectionDailyFile))
	if err != nil {
		// Non-fatal: treat as 0.
		dailyCount = 0
	}
	result.RunsToday = dailyCount
	if dailyCount >= reflectionMaxPerDay {
		result.SkipReason = fmt.Sprintf("daily limit reached: %d/%d runs today", dailyCount, reflectionMaxPerDay)
		return result, nil, nil
	}

	// Gate 3: Importance accumulation check.
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

	if total < e.cfg.Threshold {
		result.SkipReason = fmt.Sprintf("importance accumulation %.1f < threshold %.1f (%d unreflected memories)",
			total, e.cfg.Threshold, len(unreflected))
		return result, unreflected, nil
	}

	result.ShouldTrigger = true
	return result, unreflected, nil
}

// fetchUnreflected returns memories where metadata.reflected is not true.
// Uses two strategies: first tries metadata.reflected==false filter, then falls back
// to fetching all and filtering in-memory (handles missing field case).
func (e *Engine) fetchUnreflected(ctx context.Context, limit int) ([]memory.Memory, error) {
	// Strategy: Scroll all recent memories (Qdrant nested payload filter for
	// metadata.reflected may not be indexed, so we fetch and filter in-memory).
	// We fetch 3x limit to account for already-reflected ones.
	fetchLimit := limit * 3
	if fetchLimit > 600 {
		fetchLimit = 600
	}

	all, _, err := e.store.Scroll(ctx, memory.ScrollOptions{
		Limit: fetchLimit,
	})
	if err != nil {
		return nil, fmt.Errorf("scroll memories: %w", err)
	}

	// Filter: keep only those NOT marked as reflected.
	var unreflected []memory.Memory
	for _, m := range all {
		if !isReflected(m) {
			unreflected = append(unreflected, m)
			if len(unreflected) >= limit {
				break
			}
		}
	}
	return unreflected, nil
}

// isReflected returns true if the memory has been marked as reflected.
func isReflected(m memory.Memory) bool {
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

	return nil
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

// dailyCountEntry holds the date and count stored in the daily count file.
type dailyCountEntry struct {
	Date  string `json:"date"`
	Count int    `json:"count"`
}

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

	today := time.Now().UTC().Format("2006-01-02")
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

// writeDailyCount writes today's count to file.
func writeDailyCount(path string, count int) error {
	today := time.Now().UTC().Format("2006-01-02")
	return os.WriteFile(path, []byte(fmt.Sprintf("%s %d\n", today, count)), 0644)
}
