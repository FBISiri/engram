package replay

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/FBISiri/engram/pkg/trajectory"
)

// Engine replays recorded queries against a running Engram server over HTTP.
//
// It is deliberately constructed around an *http.Client and a base URL so its
// request/response marshalling and the collection guard are unit-testable with
// httptest, without requiring a live server.
type Engine struct {
	BaseURL    string
	APIKey     string
	Collection string // MUST carry the engram_eval_ prefix (guarded)
	Client     *http.Client
}

// DefaultBaseURL / DefaultCollection are used when the environment is unset.
const (
	DefaultBaseURL    = "http://localhost:8080"
	DefaultCollection = "engram_eval_replay"
)

// NewEngineFromEnv builds an Engine from ENGRAM_URL / ENGRAM_API_KEY, falling
// back to sane defaults. The collection must still be provided by the caller.
func NewEngineFromEnv(collection string) *Engine {
	base := os.Getenv("ENGRAM_URL")
	if base == "" {
		base = DefaultBaseURL
	}
	return &Engine{
		BaseURL:    base,
		APIKey:     os.Getenv("ENGRAM_API_KEY"),
		Collection: collection,
		Client:     &http.Client{Timeout: 30 * time.Second},
	}
}

// RunOptions controls a replay run.
type RunOptions struct {
	// SnapshotID, if set, is restored (memory_reset action=restore) before the
	// replay begins, giving a deterministic baseline memory state. The engine
	// first captures the CURRENT state to a fresh snapshot and restores it on
	// exit so production data is never irreversibly overwritten.
	SnapshotID string
	// ConfigOverride, if non-nil, is applied via memory_apply_config before the
	// replay begins (A/B testing). It is reverted on exit by re-applying
	// BaselineConfig.
	ConfigOverride *MemoryConfig
	// BaselineConfig is the config to restore after a ConfigOverride run. The
	// server exposes no read-config API AND memory_apply_config is set-only
	// (cannot clear a field), so when ConfigOverride is set this MUST be a
	// non-zero config declaring the operator's current values; otherwise Run
	// fails closed rather than leave the override live on the global server.
	BaselineConfig *MemoryConfig
	// TopK is the FLOOR on results requested per query. The engine requests
	// max(len(RecordedResults), TopK) so recall is not artificially capped for
	// cases that recorded more than TopK hits. Default 10.
	TopK int
}

func (e *Engine) client() *http.Client {
	if e.Client != nil {
		return e.Client
	}
	return http.DefaultClient
}

// Run replays every case and returns the live results. It guards the target
// collection first (R10), ensures the eval collection is registered, and — for
// any global mutation (config override or snapshot restore) — captures the
// original state up front and DEFERS its restoration so it runs on every exit
// path (success or error). See design §4.2B step 5.
func (e *Engine) Run(ctx context.Context, cases []ReplayCase, opts RunOptions) ([]ReplayResult, error) {
	if _, err := guardCollection(e.Collection); err != nil {
		return nil, err
	}
	if err := e.ensureCollection(ctx); err != nil {
		return nil, fmt.Errorf("ensure eval collection: %w", err)
	}

	// Detached context so restores fire even if ctx is later cancelled.
	restoreCtx := context.WithoutCancel(ctx)

	if opts.SnapshotID != "" {
		origID, err := e.snapshotState(ctx)
		if err != nil {
			return nil, fmt.Errorf("capture baseline state: %w", err)
		}
		defer func() {
			if err := e.restoreState(restoreCtx, origID); err != nil {
				fmt.Fprintf(os.Stderr, "REPLAY: FAILED to restore original state snapshot %s: %v\n", origID, err)
			}
		}()
		if err := e.restoreState(ctx, opts.SnapshotID); err != nil {
			return nil, fmt.Errorf("restore snapshot %s: %w", opts.SnapshotID, err)
		}
	}

	var applied MemoryConfig
	if opts.ConfigOverride != nil {
		// memory_apply_config is SET-ONLY: it applies a field only when nonzero
		// and can never clear one. Restoring therefore requires an explicit,
		// concrete baseline; a zero MemoryConfig would be a silent no-op leaving
		// the override live on the GLOBAL server. Fail closed.
		if opts.BaselineConfig == nil || isZeroConfig(*opts.BaselineConfig) {
			return nil, fmt.Errorf("refusing: cannot safely restore config; supply --baseline-config with current values")
		}
		baseline := *opts.BaselineConfig
		defer func() {
			if err := e.applyConfig(restoreCtx, baseline); err != nil {
				fmt.Fprintf(os.Stderr, "REPLAY: FAILED to restore baseline config: %v\n", err)
			}
		}()
		if err := e.applyConfig(ctx, *opts.ConfigOverride); err != nil {
			return nil, fmt.Errorf("apply config: %w", err)
		}
		applied = *opts.ConfigOverride
	}

	floor := opts.TopK
	if floor <= 0 {
		floor = 10
	}

	results := make([]ReplayResult, 0, len(cases))
	for i := range cases {
		c := &cases[i]
		limit := floor
		if n := len(c.RecordedResults); n > limit {
			limit = n // don't cap recall for cases that recorded > floor hits
		}
		start := time.Now()
		live, err := e.search(ctx, c.Query, limit)
		if err != nil {
			return nil, fmt.Errorf("search %q: %w", c.Query, err)
		}
		results = append(results, ReplayResult{
			Case:          c,
			LiveResults:   live,
			LiveLatency:   time.Since(start).Milliseconds(),
			ConfigApplied: applied,
		})
	}
	return results, nil
}

// ensureCollection registers the eval collection so /memories/search accepts it
// (the endpoint validates the `collection` field against the server registry).
// 201 Created and 409 Conflict (already registered) are both success.
//
// assumed: eval collections are registered via POST /collections; the
// underlying store is single-collection (W20 Phase 2), so the collection field
// only satisfies validation and does not physically isolate the search.
func (e *Engine) ensureCollection(ctx context.Context) error {
	if _, err := guardCollection(e.Collection); err != nil {
		return err
	}
	body, _ := json.Marshal(map[string]any{"name": e.Collection})
	resp, err := e.do(ctx, http.MethodPost, "/collections", body)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusCreated || resp.StatusCode == http.StatusConflict {
		return nil
	}
	return fmt.Errorf("register collection status %d: %s", resp.StatusCode, string(data))
}

// search issues one recorded query against the live search endpoint.
func (e *Engine) search(ctx context.Context, query string, limit int) ([]trajectory.ResultItem, error) {
	if _, err := guardCollection(e.Collection); err != nil {
		return nil, err
	}
	body, _ := json.Marshal(map[string]any{
		"query":      query,
		"collection": e.Collection,
		"limit":      limit,
	})
	resp, err := e.do(ctx, http.MethodPost, "/memories/search", body)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search status %d: %s", resp.StatusCode, string(data))
	}
	// The endpoint returns a JSON array of objects carrying id/content/score.
	var items []trajectory.ResultItem
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, fmt.Errorf("decode search response: %w", err)
	}
	return items, nil
}

// applyConfig hot-reloads retrieval/update config via the memory_apply_config
// MCP tool over the streamable-HTTP JSON-RPC endpoint.
func (e *Engine) applyConfig(ctx context.Context, cfg MemoryConfig) error {
	cfgJSON, _ := json.Marshal(cfg)
	_, err := e.callMCPTool(ctx, "memory_apply_config", map[string]any{
		"config": string(cfgJSON),
	})
	return err
}

// snapshotState captures the current memory state to a fresh snapshot and
// returns its id (memory_reset action=snapshot).
func (e *Engine) snapshotState(ctx context.Context) (string, error) {
	txt, err := e.callMCPTool(ctx, "memory_reset", map[string]any{"action": "snapshot"})
	if err != nil {
		return "", err
	}
	var out struct {
		SnapshotID string `json:"snapshot_id"`
	}
	if err := json.Unmarshal([]byte(txt), &out); err != nil {
		return "", fmt.Errorf("parse snapshot result: %w", err)
	}
	if out.SnapshotID == "" {
		return "", fmt.Errorf("snapshot returned empty snapshot_id: %s", txt)
	}
	return out.SnapshotID, nil
}

// restoreState restores a snapshot via the memory_reset MCP tool.
func (e *Engine) restoreState(ctx context.Context, snapshotID string) error {
	_, err := e.callMCPTool(ctx, "memory_reset", map[string]any{
		"action":      "restore",
		"snapshot_id": snapshotID,
	})
	return err
}

// callMCPTool posts a JSON-RPC tools/call request to the MCP endpoint and
// returns the concatenated text content of the tool result.
func (e *Engine) callMCPTool(ctx context.Context, name string, args map[string]any) (string, error) {
	reqBody, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      1,
		"method":  "tools/call",
		"params":  map[string]any{"name": name, "arguments": args},
	})
	resp, err := e.do(ctx, http.MethodPost, "/mcp", reqBody)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s status %d: %s", name, resp.StatusCode, string(data))
	}
	var rpc struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
			IsError bool `json:"isError"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(data, &rpc); err != nil {
		return "", fmt.Errorf("decode %s response: %w", name, err)
	}
	if rpc.Error != nil {
		return "", fmt.Errorf("%s rpc error: %s", name, rpc.Error.Message)
	}
	var text string
	for _, c := range rpc.Result.Content {
		text += c.Text
	}
	// Tools report business failures via NewToolResultError -> HTTP 200 with
	// result.isError=true. Treat that as an error so failed apply/reset/restore
	// are never mistaken for success.
	if rpc.Result.IsError {
		return "", fmt.Errorf("%s tool error: %s", name, text)
	}
	return text, nil
}

// isZeroConfig reports whether every field of cfg is zero-valued (so applying
// it via memory_apply_config would be a no-op that cannot restore anything).
func isZeroConfig(c MemoryConfig) bool {
	return c == MemoryConfig{}
}

func (e *Engine) do(ctx context.Context, method, path string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, e.BaseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	return e.client().Do(req)
}
