// Package dream implements the Dream Engine — Siri's autonomous memory
// consolidation and insight generation system.
package dream

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
)

const (
	siriDir          = ".siri"
	lastRunFile      = "dream_last_run"
	pidFile          = "dream.pid"

	gateTimeInterval    = 20 * time.Hour // Reduced from 24h for 20:00 daily trigger flexibility
	gateNewMemoriesMin  = 20             // Min new memories since last run (replaces session count)
	pidStaleTimeout     = 2 * time.Hour
)

// GateResult is the JSON output of dream-check.
type GateResult struct {
	ShouldRun               bool    `json:"should_run"`
	Reason                  string  `json:"reason"`
	LastRun                 string  `json:"last_run"`
	HoursSinceLastRun       float64 `json:"hours_since_last_run,omitempty"`
	NewMemoriesSinceLastRun int     `json:"new_memories_since_last_run"`
	Gate1Time               bool    `json:"gate1_time"`
	Gate2Memories           bool    `json:"gate2_memories"`
	Gate3Pid                bool    `json:"gate3_pid"`
}

// siriDirPath returns ~/.siri, creating it if needed.
func siriDirPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}
	dir := filepath.Join(home, siriDir)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("create %s: %w", dir, err)
	}
	return dir, nil
}

// CheckGates evaluates the Triple Gate and returns a JSON-serializable result.
// store is used for Gate 2 (count new memories since last run).
// If store is nil, Gate 2 is skipped (passes by default) — for backward compatibility.
func CheckGates(store memory.Store) (*GateResult, error) {
	dir, err := siriDirPath()
	if err != nil {
		return nil, err
	}

	result := &GateResult{}

	// Gate 1: Time interval >= 20h since last run.
	lastRun, err := readLastRun(filepath.Join(dir, lastRunFile))
	if err != nil {
		return nil, fmt.Errorf("gate1: %w", err)
	}
	if lastRun.IsZero() {
		result.LastRun = "never"
		result.HoursSinceLastRun = 0
	} else {
		result.LastRun = lastRun.UTC().Format(time.RFC3339)
		result.HoursSinceLastRun = time.Since(lastRun).Hours()
	}
	gate1 := lastRun.IsZero() || time.Since(lastRun) >= gateTimeInterval
	result.Gate1Time = gate1

	// Gate 2: New memories since last run >= gateNewMemoriesMin.
	var newMemCount int
	gate2 := true // default pass if no store provided
	if store != nil {
		newMemCount, err = countNewMemoriesSince(store, lastRun)
		if err != nil {
			// Non-fatal: log and treat as gate2 pass to avoid blocking Dream Engine
			fmt.Fprintf(os.Stderr, "warning: gate2 memory count failed: %v (treating as pass)\n", err)
			gate2 = true
		} else {
			gate2 = newMemCount >= gateNewMemoriesMin
		}
	}
	result.NewMemoriesSinceLastRun = newMemCount
	result.Gate2Memories = gate2

	// Gate 3: PID lock — no other dream-run is active.
	gate3, pidReason := checkPIDLock(filepath.Join(dir, pidFile))
	result.Gate3Pid = gate3

	// Evaluate.
	switch {
	case !gate1:
		elapsed := time.Since(lastRun).Truncate(time.Minute)
		result.Reason = fmt.Sprintf("gate1 fail: only %s since last run (need %s)", elapsed, gateTimeInterval)
	case !gate2:
		result.Reason = fmt.Sprintf("gate2 fail: only %d new memories since last run (need %d)", newMemCount, gateNewMemoriesMin)
	case !gate3:
		result.Reason = fmt.Sprintf("gate3 fail: %s", pidReason)
	default:
		result.ShouldRun = true
		result.Reason = "all gates passed"
	}

	return result, nil
}

// countNewMemoriesSince counts memories created after the given time.
// If lastRun is zero, counts all memories (first run).
func countNewMemoriesSince(store memory.Store, lastRun time.Time) (int, error) {
	ctx := context.Background()

	opts := memory.ScrollOptions{
		Limit: 500, // fetch up to 500 new memories for counting
	}

	if !lastRun.IsZero() {
		opts.Filters = []memory.Filter{
			{Field: "created_at", Op: memory.OpGte, Value: lastRun.Unix()},
		}
	}

	mems, _, err := store.Scroll(ctx, opts)
	if err != nil {
		return 0, fmt.Errorf("scroll memories: %w", err)
	}
	return len(mems), nil
}

// PrintGateResult prints the gate check result as JSON to stdout.
func PrintGateResult(r *GateResult) error {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// readLastRun reads the last-run timestamp file. Returns zero time if file doesn't exist.
func readLastRun(path string) (time.Time, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, err
	}
	// Try parsing as RFC3339 first (new format).
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(string(data)))
	if err == nil {
		return t, nil
	}
	// Fall back to file mtime for backward compatibility.
	info, err := os.Stat(path)
	if err != nil {
		return time.Time{}, err
	}
	return info.ModTime(), nil
}

// checkPIDLock returns (ok, reason). ok=true means no active lock.
func checkPIDLock(path string) (bool, string) {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return true, ""
	}
	if err != nil {
		return false, fmt.Sprintf("stat pid file: %v", err)
	}

	// Stale check: if file older than pidStaleTimeout, auto-remove.
	if time.Since(info.ModTime()) > pidStaleTimeout {
		os.Remove(path)
		return true, ""
	}

	// Check if PID is still alive.
	data, err := os.ReadFile(path)
	if err != nil {
		return false, fmt.Sprintf("read pid file: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		os.Remove(path) // corrupt
		return true, ""
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		os.Remove(path)
		return true, ""
	}
	// Signal 0 checks existence without killing.
	if err := proc.Signal(syscall.Signal(0)); err != nil {
		os.Remove(path) // process dead
		return true, ""
	}

	return false, fmt.Sprintf("dream-run already running (pid %d)", pid)
}

// WritePIDLock creates the PID lock file with the current process's PID.
func WritePIDLock() error {
	dir, err := siriDirPath()
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, pidFile), []byte(strconv.Itoa(os.Getpid())), 0644)
}

// ReleasePIDLock removes the PID lock file.
func ReleasePIDLock() {
	dir, err := siriDirPath()
	if err != nil {
		return
	}
	os.Remove(filepath.Join(dir, pidFile))
}

// UpdateRunTimestamp writes the current time to the last-run file.
// (Session counter removed — replaced by new-memories gate.)
func UpdateRunTimestamp() error {
	dir, err := siriDirPath()
	if err != nil {
		return err
	}
	// Write last-run file with RFC3339 timestamp content.
	if err := os.WriteFile(filepath.Join(dir, lastRunFile), []byte(time.Now().UTC().Format(time.RFC3339)), 0644); err != nil {
		return fmt.Errorf("update last run: %w", err)
	}
	return nil
}
