package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/mcp"

	"github.com/FBISiri/engram/pkg/reflection"
)

// reflectionRunner is a server-level single-flight runner for reflection
// cycles. LLM-backed reflection can take minutes, far longer than the MCP
// client bridge's hard 30s timeout, so runs are launched in a detached
// goroutine (with a Background-derived context) and observed via
// reflection_status / GET /health.
type reflectionRunner struct {
	mu         sync.Mutex
	running    bool
	runID      string
	startedAt  time.Time
	lastRunAt  time.Time
	lastRunID  string
	lastResult *reflection.RunResult
	lastError  string
	// runFn is a test hook; nil => use the engine-backed defaultRun passed to start.
	runFn func(ctx context.Context) (*reflection.RunResult, error)
}

// reflectionStatus is the serialized runner state shared verbatim by the
// reflection_status MCP tool and GET /health, so operators see identical JSON.
type reflectionStatus struct {
	Running              bool     `json:"running"`
	RunID                string   `json:"run_id"`
	StartedAt            string   `json:"started_at"`
	LastRunAt            string   `json:"last_run_at"`
	LastRunID            string   `json:"last_run_id"`
	LastDuration         string   `json:"last_duration"`
	InsightsCreated      int      `json:"insights_created"`
	InsightsDedupSkipped int      `json:"insights_dedup_skipped"`
	Triggered            bool     `json:"triggered"`
	SkipReason           string   `json:"skip_reason"`
	Mode                 string   `json:"mode"`
	Errors               []string `json:"errors"`
	LastError            string   `json:"last_error"`
}

// newRunID returns an 8-byte hex run identifier, falling back to a time-based
// value if the system RNG is unavailable.
func newRunID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("t%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// start launches a reflection run unless one is already in flight (single
// flight). If a run is already active it returns started=false along with the
// in-flight run's id and start time, WITHOUT launching anything. Otherwise it
// marks running, generates a run_id, and spawns a detached goroutine that
// executes runFn (or defaultRun when the test hook is nil) using a
// Background-derived context capped at 30 minutes — never the request context.
func (r *reflectionRunner) start(
	defaultRun func(ctx context.Context) (*reflection.RunResult, error),
	recordMetrics func(*reflection.RunResult),
) (started bool, runID string, startedAt time.Time) {
	r.mu.Lock()
	if r.running {
		id, at := r.runID, r.startedAt
		r.mu.Unlock()
		return false, id, at
	}
	r.running = true
	r.runID = newRunID()
	r.startedAt = time.Now()
	id, at := r.runID, r.startedAt
	fn := r.runFn
	r.mu.Unlock()

	if fn == nil {
		fn = defaultRun
	}

	slog.Info("reflection run started", "run_id", id)

	go func() {
		// Decouple from the request context: reflection may run for minutes.
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
		defer cancel()

		// Single-flight reset MUST always happen, even if fn panics; otherwise
		// running stays true forever and wedges the runner. Recover here so a
		// panic in fn cannot leak, and release the flight under the same mutex.
		defer func() {
			if p := recover(); p != nil {
				slog.Error("reflection run panicked", "run_id", id, "panic", p)
			}
			r.mu.Lock()
			r.running = false
			r.runID = ""
			r.mu.Unlock()
		}()

		result, err := fn(ctx)

		r.mu.Lock()
		r.lastResult = result
		if err != nil {
			r.lastError = err.Error()
		} else if result != nil && len(result.Errors) > 0 {
			r.lastError = summarizeReflectionErrors(result.Errors)
		} else {
			r.lastError = ""
		}
		r.lastRunAt = time.Now()
		r.lastRunID = id
		r.mu.Unlock()

		if err == nil && result != nil && recordMetrics != nil {
			recordMetrics(result)
		}

		created, dur := 0, ""
		if result != nil {
			created = result.InsightsCreated
			dur = result.Duration
		}
		errStr := ""
		if err != nil {
			errStr = err.Error()
		}
		slog.Info("reflection run finished",
			"run_id", id, "insights_created", created, "duration", dur, "error", errStr)
	}()

	return true, id, at
}

// summarizeReflectionErrors condenses a reflection run's Errors slice into a
// single LastError line: empty when none, the sole error when one, and
// "first (+N more)" when several.
func summarizeReflectionErrors(errs []string) string {
	if len(errs) == 0 {
		return ""
	}
	if len(errs) == 1 {
		return errs[0]
	}
	return fmt.Sprintf("%s (+%d more)", errs[0], len(errs)-1)
}

// status returns a snapshot of the runner state for serialization. Fields are
// zeroed/empty when no run has completed; Errors is always non-nil.
func (r *reflectionRunner) status() reflectionStatus {
	r.mu.Lock()
	defer r.mu.Unlock()

	st := reflectionStatus{
		Running:   r.running,
		RunID:     r.runID,
		LastRunID: r.lastRunID,
		LastError: r.lastError,
		Errors:    []string{},
	}
	if !r.startedAt.IsZero() {
		st.StartedAt = r.startedAt.Format(time.RFC3339)
	}
	if !r.lastRunAt.IsZero() {
		st.LastRunAt = r.lastRunAt.Format(time.RFC3339)
	}
	if r.lastResult != nil {
		st.LastDuration = r.lastResult.Duration
		st.InsightsCreated = r.lastResult.InsightsCreated
		st.InsightsDedupSkipped = r.lastResult.InsightsDedupSkipped
		st.Triggered = r.lastResult.Triggered
		st.SkipReason = r.lastResult.SkipReason
		st.Mode = r.lastResult.Mode
		if len(r.lastResult.Errors) > 0 {
			st.Errors = r.lastResult.Errors
		}
	}
	return st
}

// handleReflectionStatus implements the reflection_status tool. It reports the
// async runner's live and last-completed state so callers can poll after
// reflection_run returns immediately.
func (s *Server) handleReflectionStatus(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if _, isolated := isolatedCaller(ctx); isolated {
		return mcp.NewToolResultError("reflection_status is a global-scope action not permitted for isolated callers"), nil
	}
	data, err := json.Marshal(s.reflectionRunner.status())
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}
	return mcp.NewToolResultText(string(data)), nil
}
