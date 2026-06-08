// eval.go — ApplyConfig and Reset tools for ALMA-style evaluation loops.
//
// memory_apply_config  Hot-reload retrieval/update config at runtime.
// memory_reset         Snapshot the current memory state or restore from a snapshot.
//
// Both are MCP tools registered alongside the standard memory tools.
// Snapshot files are written as JSONL to the directory given by ENGRAM_SNAPSHOT_DIR
// (default: /data/engram/snapshots/).
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/FBISiri/engram/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
)

// =============================================================================
// Runtime config (hot-reloadable via memory_apply_config)
// =============================================================================

// RetrieveConfig holds hot-reloadable retrieval settings.
type RetrieveConfig struct {
	RecencyWeight      float64 `json:"recency_weight,omitempty"`
	ScoreThreshold     float64 `json:"score_threshold,omitempty"`
	TopK               int     `json:"top_k,omitempty"`
	QueryRewritePrompt string  `json:"query_rewrite_prompt,omitempty"`
}

// UpdateConfig holds hot-reloadable write settings.
type UpdateConfig struct {
	DedupeThreshold float64 `json:"dedupe_threshold,omitempty"`
	MaxEntries      int     `json:"max_entries,omitempty"`
	EvictionPolicy  string  `json:"eviction_policy,omitempty"`
}

// MemoryConfig bundles the two hot-reloadable sub-configs.
type MemoryConfig struct {
	RetrieveConfig RetrieveConfig `json:"retrieve_config"`
	UpdateConfig   UpdateConfig   `json:"update_config"`
}

// runtimeOverrides stores active config overrides under a RWMutex.
// nil pointer fields mean "use server default".
type runtimeOverrides struct {
	mu      sync.RWMutex
	weights *memory.ScoringWeights
	dedup   *float64
	topK    *int
}

func (o *runtimeOverrides) getWeights(def memory.ScoringWeights) memory.ScoringWeights {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if o.weights != nil {
		return *o.weights
	}
	return def
}

func (o *runtimeOverrides) getDedupThreshold(def float64) float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if o.dedup != nil {
		return *o.dedup
	}
	return def
}

// =============================================================================
// memory_apply_config handler
// =============================================================================

func (s *Server) handleApplyConfig(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	configJSON, err := request.RequireString("config")
	if err != nil {
		return mcp.NewToolResultError("config is required (JSON string)"), nil
	}

	var cfg MemoryConfig
	if err := json.Unmarshal([]byte(configJSON), &cfg); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("invalid config JSON: %v", err)), nil
	}

	s.overrides.mu.Lock()
	defer s.overrides.mu.Unlock()

	if cfg.RetrieveConfig.RecencyWeight != 0 {
		w := s.weights // copy defaults
		w.Recency = cfg.RetrieveConfig.RecencyWeight
		s.overrides.weights = &w
	}
	if cfg.RetrieveConfig.TopK > 0 {
		k := cfg.RetrieveConfig.TopK
		s.overrides.topK = &k
	}
	if cfg.UpdateConfig.DedupeThreshold > 0 {
		d := cfg.UpdateConfig.DedupeThreshold
		s.overrides.dedup = &d
	}

	type result struct {
		Status  string       `json:"status"`
		Applied MemoryConfig `json:"applied"`
	}
	data, _ := json.Marshal(result{Status: "applied", Applied: cfg})
	return mcp.NewToolResultText(string(data)), nil
}

// =============================================================================
// memory_reset handler (snapshot + restore)
// =============================================================================

// snapshotRecord is one memory entry stored in the snapshot JSONL.
type snapshotRecord struct {
	memory.Memory
}

func snapshotDir() string {
	if d := os.Getenv("ENGRAM_SNAPSHOT_DIR"); d != "" {
		return d
	}
	return "/data/engram/snapshots"
}

func (s *Server) handleReset(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	action := request.GetString("action", "snapshot")

	switch action {
	case "snapshot":
		return s.doSnapshot(ctx)
	case "restore":
		snapshotID, err := request.RequireString("snapshot_id")
		if err != nil {
			return mcp.NewToolResultError("snapshot_id is required for action=restore"), nil
		}
		return s.doRestore(ctx, snapshotID)
	default:
		return mcp.NewToolResultError(fmt.Sprintf("unknown action %q (use snapshot or restore)", action)), nil
	}
}

func (s *Server) doSnapshot(ctx context.Context) (*mcp.CallToolResult, error) {
	memories, _, err := s.store.Scroll(ctx, memory.ScrollOptions{Limit: 10000})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("scroll error: %v", err)), nil
	}

	dir := snapshotDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("create snapshot dir: %v", err)), nil
	}

	snapshotID := time.Now().UTC().Format("20060102-150405")
	path := filepath.Join(dir, snapshotID+".jsonl")

	f, err := os.Create(path)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("create snapshot file: %v", err)), nil
	}
	defer f.Close()

	for _, m := range memories {
		data, err := json.Marshal(snapshotRecord{m})
		if err == nil {
			_, _ = f.Write(append(data, '\n'))
		}
	}

	type result struct {
		Status     string `json:"status"`
		SnapshotID string `json:"snapshot_id"`
		Count      int    `json:"count"`
		Path       string `json:"path"`
	}
	data, _ := json.Marshal(result{
		Status:     "snapshot_created",
		SnapshotID: snapshotID,
		Count:      len(memories),
		Path:       path,
	})
	return mcp.NewToolResultText(string(data)), nil
}

func (s *Server) doRestore(ctx context.Context, snapshotID string) (*mcp.CallToolResult, error) {
	path := filepath.Join(snapshotDir(), snapshotID+".jsonl")
	f, err := os.Open(path)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("open snapshot %s: %v", snapshotID, err)), nil
	}
	defer f.Close()

	// Parse snapshot records
	var records []snapshotRecord
	dec := json.NewDecoder(f)
	for dec.More() {
		var r snapshotRecord
		if err := dec.Decode(&r); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("decode snapshot: %v", err)), nil
		}
		records = append(records, r)
	}

	// Delete all current memories
	current, _, err := s.store.Scroll(ctx, memory.ScrollOptions{Limit: 10000})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("scroll current: %v", err)), nil
	}
	if len(current) > 0 {
		ids := make([]string, len(current))
		for i, m := range current {
			ids[i] = m.ID
		}
		if _, err := s.store.Delete(ctx, ids); err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("delete current: %v", err)), nil
		}
	}

	// Re-embed and re-insert snapshot records
	inserted := 0
	failed := 0
	for _, r := range records {
		m := r.Memory
		vec, err := s.embedder.Embed(ctx, m.Content)
		if err != nil {
			failed++
			continue
		}
		if err := s.store.Insert(ctx, &m, vec); err != nil {
			failed++
			continue
		}
		inserted++
	}

	type result struct {
		Status   string `json:"status"`
		Restored int    `json:"restored"`
		Failed   int    `json:"failed,omitempty"`
		Deleted  int    `json:"deleted"`
	}
	data, _ := json.Marshal(result{
		Status:   "restored",
		Restored: inserted,
		Failed:   failed,
		Deleted:  len(current),
	})
	return mcp.NewToolResultText(string(data)), nil
}
