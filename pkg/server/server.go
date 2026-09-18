// Package server implements MCP and REST API transports for Engram.
package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"

	"github.com/FBISiri/engram/pkg/collection"
	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/embedding"
	"github.com/FBISiri/engram/pkg/memory"
	engrammetrics "github.com/FBISiri/engram/pkg/metrics"
	"github.com/FBISiri/engram/pkg/reflection"
	"github.com/FBISiri/engram/pkg/semconv"
	"github.com/FBISiri/engram/pkg/trajectory"
)

var tracer = otel.Tracer("engram.memory")

// strictProvenanceMsg is returned when a write omits source_type while
// ENGRAM_PROVENANCE_MODE=strict.
const strictProvenanceMsg = "source_type is required (ENGRAM_PROVENANCE_MODE=strict). Valid values: tool_output, reflection, web_search, user_input, calendar, document, unknown"

// strictProvenanceRejectMsg is returned when a write supplies an explicit
// source_type that is not permitted under ENGRAM_PROVENANCE_MODE=strict (the
// "unknown" sentinel, or a value outside ENGRAM_ALLOWED_PROVENANCES).
const strictProvenanceRejectMsg = "source_type not permitted (ENGRAM_PROVENANCE_MODE=strict): \"unknown\" and values outside ENGRAM_ALLOWED_PROVENANCES are rejected"

// Server wraps the MCP server with Engram's memory operations.
type Server struct {
	store             memory.Store
	embedder          embedding.Embedder
	weights           memory.ScoringWeights
	decay             memory.DecayConfig
	evapCfg           memory.EvaporationConfig
	mmrLambda         float64
	dedupThreshold    float64
	mcpServer         *mcpserver.MCPServer
	embedCache        memory.EmbedCache      // optional; set via SetEmbedCache
	metrics           *engrammetrics.Metrics // optional; set via SetMetrics
	traj              *trajectory.Logger     // optional; set via SetTrajectoryLogger
	overrides         runtimeOverrides       // hot-reloadable config
	cfg               *config.Config         // full loaded config
	rateLimiter       *RateLimiter           // A-MAC per-type write limiter
	importanceMonitor *ImportanceMonitor     // CP2 monitor; nil unless write checkpoints enabled
	contentChecker    *ContentChecker        // CP4 checker; nil unless write checkpoints enabled
	reflectionRunner  *reflectionRunner      // single-flight async reflection runner
}

// Metrics returns the prometheus metrics registered via SetMetrics (nil until set).
func (s *Server) Metrics() *engrammetrics.Metrics { return s.metrics }

// SetMetrics registers prometheus metrics so handlers can record latency.
func (s *Server) SetMetrics(m *engrammetrics.Metrics) {
	s.metrics = m
	if s.importanceMonitor != nil {
		s.importanceMonitor.SetMetrics(m)
	}
}

// evaporationConfig returns the server's effective evaporation config, honoring
// any runtime override applied via memory_apply_config.
func (s *Server) evaporationConfig() memory.EvaporationConfig {
	return s.overrides.getEvaporation(s.evapCfg)
}

// collectionStatter is satisfied by qdrant.MultiStore.
type collectionStatter interface {
	PerCollectionStats(ctx context.Context) map[string]uint64
}

// targetedUpdater is satisfied by qdrant.MultiStore.
// It allows updating only the collection that owns a point, avoiding fan-out
// NotFound WARNs when SetPayload targets all collections.
type targetedUpdater interface {
	UpdateInCollection(ctx context.Context, id string, fields map[string]any, sourceCollection string) error
}

// PerCollectionStats returns per-collection memory counts if the backing store
// supports it; returns nil otherwise.
func (s *Server) PerCollectionStats(ctx context.Context) map[string]uint64 {
	if cs, ok := s.store.(collectionStatter); ok {
		return cs.PerCollectionStats(ctx)
	}
	return nil
}

// SetEmbedCache registers a cache so its stats can be exposed via /metrics.
func (s *Server) SetEmbedCache(c memory.EmbedCache) {
	s.embedCache = c
}

// SetTrajectoryLogger registers a trajectory logger so search/update operations are logged.
func (s *Server) SetTrajectoryLogger(l *trajectory.Logger) {
	s.traj = l
}

// NewServer creates a new Engram MCP server with all tools registered.
func NewServer(store memory.Store, embedder embedding.Embedder, cfg *config.Config) *Server {
	s := &Server{
		store:            store,
		embedder:         embedder,
		weights:          cfg.Weights,
		decay:            cfg.Decay,
		evapCfg:          cfg.Evaporation,
		mmrLambda:        cfg.MMRLambda,
		dedupThreshold:   cfg.DedupThreshold,
		cfg:              cfg,
		reflectionRunner: &reflectionRunner{},
	}

	rateLimits := cfg.RateLimits
	if rateLimits == nil {
		rateLimits = config.DefaultRateLimits()
	}
	s.rateLimiter = NewRateLimiter(rateLimits)

	// Write Discipline Checkpoints (spec v1): allocate monitor/checker only when
	// the master switch is enabled — zero overhead otherwise (R11).
	if cfg.WriteCheckpointsEnabled {
		s.importanceMonitor = NewImportanceMonitor(cfg.CPImportanceWindow, cfg.CPImportanceThreshold)
		s.contentChecker = &ContentChecker{
			MinContentLength:  cfg.CPContentMinLength,
			MaxContentLength:  cfg.CPContentMaxLength,
			RequireTags:       cfg.CPRequireTags,
			RequireSourceType: cfg.CPRequireSourceType,
		}
	}

	s.mcpServer = mcpserver.NewMCPServer(
		"Engram",
		"0.1.0",
		mcpserver.WithToolCapabilities(false),
	)

	s.registerTools()
	return s
}

// GetMCPServer returns the underlying mcp-go server for testing or advanced use.
func (s *Server) GetMCPServer() *mcpserver.MCPServer {
	return s.mcpServer
}

// ServeStdio starts the server using stdio transport (for MCP clients).
func (s *Server) ServeStdio() error {
	return mcpserver.ServeStdio(s.mcpServer)
}

// registerTools adds all memory and reflection tools to the MCP server.
func (s *Server) registerTools() {
	// Tool 1: memory_search
	searchTool := mcp.NewTool("memory_search",
		mcp.WithDescription("Semantic search over stored memories. Returns scored results combining relevance, recency, and importance. Each result includes source_collection identifying which collection it came from."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query text for semantic similarity matching.")),
		mcp.WithNumber("limit", mcp.Description("Maximum number of results to return. Default: 5.")),
		mcp.WithArray("types", mcp.Description("Filter by memory types (identity, event, insight, directive)."), mcp.WithStringItems()),
		mcp.WithArray("tags", mcp.Description("Filter by tags. Memories must have at least one matching tag."), mcp.WithStringItems()),
		mcp.WithNumber("time_start", mcp.Description("Filter memories created after this Unix timestamp.")),
		mcp.WithNumber("time_end", mcp.Description("Filter memories created before this Unix timestamp.")),
		mcp.WithArray("collections", mcp.Description("Filter by collection names (e.g. engram_user, engram_reflection). Default: searches all collections (fan-out)."), mcp.WithStringItems()),
		mcp.WithArray("source_type", mcp.Description("Filter by provenance source_type (tool_output, reflection, web_search, user_input, calendar, document). Memories must match at least one."), mcp.WithStringItems()),
		mcp.WithString("task_id", mcp.Description("Optional event-loop task identifier. Recorded in the trajectory log to join retrievals to task outcomes (Memory Worth analysis).")),
	)
	s.mcpServer.AddTool(searchTool, s.handleSearch)

	// Tool 1b: memory_list — filter-only time-window listing (no vector search).
	listTool := mcp.NewTool("memory_list",
		mcp.WithDescription("List memories within a time window without semantic search (no embedding cost). Filter-only pass-through to the store's scroll query. Excludes superseded/expired entries. Results are sorted by created_at ascending. Each entry includes source_collection identifying which collection it came from."),
		mcp.WithNumber("time_start", mcp.Description("List memories created at or after this Unix timestamp.")),
		mcp.WithNumber("time_end", mcp.Description("List memories created at or before this Unix timestamp.")),
		mcp.WithArray("collections", mcp.Description("Filter by collection names (e.g. engram_user, engram_reflection). Default: all collections (fan-out)."), mcp.WithStringItems()),
		mcp.WithNumber("limit", mcp.Description("Maximum number of entries to return. Default: 50. Values above 100 are capped to 100.")),
	)
	s.mcpServer.AddTool(listTool, s.handleList)

	// Tool 2: memory_add
	addTool := mcp.NewTool("memory_add",
		mcp.WithDescription("Store a new memory. Automatically deduplicates against existing memories. Optionally pass task_id to join this write to an event-loop task outcome (Memory Worth analysis)."),
		mcp.WithString("content", mcp.Required(), mcp.Description("The memory content text.")),
		mcp.WithString("type", mcp.Description("Memory type."), mcp.Enum("identity", "event", "insight", "directive")),
		mcp.WithNumber("importance", mcp.Description("Importance score. Per-type defaults if omitted: identity=6, directive=7, insight=5, event=4. Per-type bounds (values outside are clamped): identity [7,9], directive [6,10], insight [5,8], event [3,7]. Omit to use per-type default.")),
		mcp.WithArray("tags", mcp.Description("Tags for classification."), mcp.WithStringItems()),
		mcp.WithString("source", mcp.Description("Source of the memory: user, agent, or system. Default: agent.")),
		mcp.WithString("source_type", mcp.Description("Fine-grained provenance of the memory. Stored in metadata.source_type."), mcp.Enum("tool_output", "reflection", "web_search", "user_input", "calendar", "document")),
		mcp.WithNumber("valid_until", mcp.Description("Optional expiration time as Unix timestamp. 0 or omitted = never expires.")),
		mcp.WithString("task_id", mcp.Description("Optional event-loop task identifier. Recorded in the candidate trajectory record to join this write to task outcomes (Memory Worth analysis / L3 valence reinforcement). Same semantics as memory_search's task_id.")),
	)
	s.mcpServer.AddTool(addTool, s.handleAdd)

	// Tool 3: memory_update
	updateTool := mcp.NewTool("memory_update",
		mcp.WithDescription("Update memories by semantic search. Finds old memories matching old_content, deletes them, and stores new_content. SAFETY: similarity_threshold must be >= 0.85 (3 prior mass-delete incidents at lower values)."),
		mcp.WithString("old_content", mcp.Required(), mcp.Description("Search query to find old memories to replace.")),
		mcp.WithString("new_content", mcp.Required(), mcp.Description("New memory content to store.")),
		mcp.WithString("type", mcp.Description("Memory type for the new memory."), mcp.Enum("identity", "event", "insight", "directive")),
		mcp.WithNumber("importance", mcp.Description("Importance score. Per-type defaults if omitted: identity=6, directive=7, insight=5, event=4. Per-type bounds (values outside are clamped): identity [7,9], directive [6,10], insight [5,8], event [3,7]. Omit to use per-type default.")),
		mcp.WithArray("tags", mcp.Description("Tags for the new memory."), mcp.WithStringItems()),
		mcp.WithNumber("similarity_threshold", mcp.Description("Minimum cosine similarity for deletion. Must be >= 0.85. Default: 0.7 (rejected — use 0.92 for single targeted replacement).")),
		mcp.WithString("source_type", mcp.Description("Fine-grained provenance for the new memory. Stored in metadata.source_type."), mcp.Enum("tool_output", "reflection", "web_search", "user_input", "calendar", "document")),
		mcp.WithNumber("valid_until", mcp.Description("Optional expiration time as Unix timestamp. 0 or omitted = inherit from old memory.")),
		mcp.WithBoolean("dry_run", mcp.Description("If true, preview which memories would be deleted/added without making changes. Default: false.")),
	)
	s.mcpServer.AddTool(updateTool, s.handleUpdate)

	// Tool 4: memory_delete
	deleteTool := mcp.NewTool("memory_delete",
		mcp.WithDescription("Delete memories by semantic search. Finds memories matching the query above the similarity threshold and removes them. SAFETY: limit > 1 requires similarity_threshold >= 0.85."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query to find memories to delete.")),
		mcp.WithNumber("similarity_threshold", mcp.Description("Minimum cosine similarity for deletion. Default: 0.7 (rejected when limit > 1 — use >= 0.85).")),
		mcp.WithNumber("limit", mcp.Description("Maximum number of memories to delete. Default: 20.")),
		mcp.WithBoolean("dry_run", mcp.Description("If true, preview which memories would be deleted without removing them. Default: false.")),
	)
	s.mcpServer.AddTool(deleteTool, s.handleDelete)

	// Tool 5: reflection_check
	reflectionCheckTool := mcp.NewTool("reflection_check",
		mcp.WithDescription("Check whether the Reflection Engine should run now. Returns trigger status, accumulated importance, unreflected memory count, and skip reason if not triggered."),
	)
	s.mcpServer.AddTool(reflectionCheckTool, s.handleReflectionCheck)

	// Tool 5b: reflection_status
	reflectionStatusTool := mcp.NewTool("reflection_status",
		mcp.WithDescription("Report the status of the asynchronous reflection runner: whether a run is in flight (running, run_id, started_at) and the last completed run's results (insights_created, duration, triggered, skip_reason, mode, errors). Poll this after reflection_run."),
	)
	s.mcpServer.AddTool(reflectionStatusTool, s.handleReflectionStatus)

	// Tool 6: reflection_run
	reflectionRunTool := mcp.NewTool("reflection_run",
		mcp.WithDescription("Run one Reflection Engine cycle. Synthesizes insights from unreflected memories using an LLM. Respects min-interval (2h) and daily limit (3x/day). Runs asynchronously: returns immediately with a run_id (or already_running if a run is in flight) — poll reflection_status for progress/results. dry_run=true is a synchronous preview that returns the RunResult directly."),
		mcp.WithBoolean("dry_run", mcp.Description("If true, simulate the run without writing any changes. Default: false.")),
	)
	s.mcpServer.AddTool(reflectionRunTool, s.handleReflectionRun)

	// Tool 7: memory_apply_config — hot-reload retrieval/update config (for ALMA eval loops).
	applyConfigTool := mcp.NewTool("memory_apply_config",
		mcp.WithDescription("Hot-reload memory retrieval/update config without restarting the server. Used by ALMA evaluation loops to iterate on MemoryConfig. Config changes take effect immediately."),
		mcp.WithString("config", mcp.Required(), mcp.Description(`JSON string with MemoryConfig. Example: {"retrieve_config":{"recency_weight":0.3},"update_config":{"dedupe_threshold":0.90}}`)),
	)
	s.mcpServer.AddTool(applyConfigTool, s.handleApplyConfig)

	// Tool 8: memory_reset — snapshot current state or restore from snapshot (for ALMA eval loops).
	resetTool := mcp.NewTool("memory_reset",
		mcp.WithDescription(`Snapshot or restore memory state for ALMA evaluation loops.
action=snapshot  Create a JSONL snapshot of all current memories. Returns snapshot_id.
action=restore   Delete all current memories and restore from snapshot_id (re-embeds content). Use in evaluation environments only.`),
		mcp.WithString("action", mcp.Required(), mcp.Description("snapshot or restore")),
		mcp.WithString("snapshot_id", mcp.Description("Required for action=restore. The snapshot_id returned by a previous snapshot call.")),
	)
	s.mcpServer.AddTool(resetTool, s.handleReset)
}

// =============================================================================
// Tool Handlers
// =============================================================================

// isolatedCaller reports whether the caller in ctx is identity-isolated and,
// if so, returns its own collection name. Isolated callers (e.g. pigo) may only
// read/write their own collection and are barred from global-scope actions.
// Non-isolated callers get ("", false) and retain today's fan-out behaviour.
func isolatedCaller(ctx context.Context) (own string, isolated bool) {
	ct := CallerTypeFromContext(ctx)
	if collection.IsIsolatedCallerType(ct) {
		return collection.DefaultRegistry.Resolve(ct), true
	}
	return "", false
}

// handleSearch implements the memory.search tool.
func (s *Server) handleSearch(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ctx, span := tracer.Start(ctx, "engram.memory.search")
	defer span.End()
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpSearchMemory)...)
	start := time.Now()
	if s.metrics != nil {
		defer func() { s.metrics.SearchDuration.Observe(time.Since(start).Seconds()) }()
	}

	// Parse parameters
	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError("query is required"), nil
	}

	limit := request.GetInt("limit", 5)
	if limit < 1 {
		limit = 1
	}
	if limit > 100 {
		limit = 100
	}

	taskID := request.GetString("task_id", "")

	// Build filters
	var filters []memory.Filter
	var tagsCount int

	// Type filter
	if types := getStringSlice(request, "types"); len(types) > 0 {
		filters = append(filters, memory.Filter{
			Field: "type",
			Op:    memory.OpIn,
			Value: types,
		})
	}

	// Tag filter
	if tags := getStringSlice(request, "tags"); len(tags) > 0 {
		tagsCount = len(tags)
		filters = append(filters, memory.Filter{
			Field: "tags",
			Op:    memory.OpIn,
			Value: tags,
		})
	}

	// Time range filters
	if timeStart := request.GetFloat("time_start", 0); timeStart > 0 {
		filters = append(filters, memory.Filter{
			Field: "created_at",
			Op:    memory.OpGte,
			Value: timeStart,
		})
	}
	if timeEnd := request.GetFloat("time_end", 0); timeEnd > 0 {
		filters = append(filters, memory.Filter{
			Field: "created_at",
			Op:    memory.OpLte,
			Value: timeEnd,
		})
	}

	// C1 provenance filter: match memories whose metadata.source_type is in the set.
	if sourceTypes := getStringSlice(request, "source_type"); len(sourceTypes) > 0 {
		for _, v := range sourceTypes {
			if memory.ValidateSourceType(v) != nil {
				return mcp.NewToolResultError(fmt.Sprintf("invalid source_type: %s", v)), nil
			}
		}
		filters = append(filters, memory.Filter{
			Field: "metadata.source_type",
			Op:    memory.OpIn,
			Value: sourceTypes,
		})
	}

	// Collections filter: validate names and filter by stored collection field.
	// ISOLATION: an isolated caller (e.g. pigo) may ONLY read its own
	// collection; a self-declared collections arg can never widen scope.
	if own, isolated := isolatedCaller(ctx); isolated {
		filters = append(filters, memory.Filter{
			Field: "collection",
			Op:    memory.OpIn,
			Value: []string{own},
		})
	} else if cols := getStringSlice(request, "collections"); len(cols) > 0 {
		for _, col := range cols {
			if _, ok := collection.DefaultRegistry.Get(col); !ok {
				return mcp.NewToolResultError(fmt.Sprintf("unknown collection: %s", col)), nil
			}
		}
		filters = append(filters, memory.Filter{
			Field: "collection",
			Op:    memory.OpIn,
			Value: cols,
		})
	}

	span.SetAttributes(
		attribute.Int("query.length", len(query)),
		attribute.Int("tags.count", tagsCount),
		attribute.Int("limit", limit),
		attribute.String("embedder.provider", fmt.Sprintf("%T", s.embedder)),
	)

	// Embed query
	embedStart := time.Now()
	vec, err := s.embedder.Embed(ctx, query)
	if s.metrics != nil {
		s.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "embedding error")
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	// Over-fetch 3x from store for scoring + MMR
	fetchLimit := limit * 3
	if fetchLimit < 10 {
		fetchLimit = 10
	}

	results, err := s.store.Search(ctx, vec, memory.SearchOptions{
		Limit:   fetchLimit,
		Filters: filters,
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "search error")
		return mcp.NewToolResultError(fmt.Sprintf("search error: %v", err)), nil
	}

	if len(results) == 0 {
		span.SetAttributes(
			attribute.Int("result.count", 0),
			attribute.Int64("latency_ms", time.Since(start).Milliseconds()),
		)
		return mcp.NewToolResultText("[]"), nil
	}

	if s.metrics != nil {
		s.metrics.SearchTopScore.WithLabelValues(results[0].Collection).Observe(results[0].Score)
	}

	// Apply 3-component scoring + MMR rerank (shared with REST search).
	weights := s.overrides.getWeights(s.weights)
	evapCfg := s.evaporationConfig()
	results = rerankResults(results, weights, s.decay, evapCfg, s.mmrLambda, limit)

	span.SetAttributes(
		attribute.Int("result.count", len(results)),
		attribute.Int64("latency_ms", time.Since(start).Milliseconds()),
	)

	// Async update access_count and last_accessed_at for returned memories.
	items := make([]accessUpdate, len(results))
	for i, r := range results {
		items[i] = accessUpdate{ID: r.ID, AccessCount: r.AccessCount, Collection: r.Collection}
	}
	var tu targetedUpdater
	if t, ok := s.store.(targetedUpdater); ok {
		tu = t
	}
	asyncUpdateAccessCounts(s.store, tu, items, "", false)

	// Format output
	type searchResult struct {
		ID                  string         `json:"id"`
		Type                string         `json:"type"`
		Content             string         `json:"content"`
		Source              string         `json:"source"`
		Importance          float64        `json:"importance"`
		EffectiveImportance float64        `json:"effective_importance"`
		Tags                []string       `json:"tags"`
		CreatedAt           float64        `json:"created_at"`
		UpdatedAt           float64        `json:"updated_at"`
		Score               float64        `json:"score"`
		ValidUntil          float64        `json:"valid_until,omitempty"`
		AccessCount         int64          `json:"access_count"`
		LastAccessedAt      float64        `json:"last_accessed_at,omitempty"`
		Metadata            map[string]any `json:"metadata,omitempty"`
		SourceCollection    string         `json:"source_collection"`
		SourceType          string         `json:"source_type,omitempty"`
	}

	output := make([]searchResult, len(results))
	for i, r := range results {
		effImp := memory.EffectiveImportance(&results[i].Memory, evapCfg, time.Now())
		if s.metrics != nil {
			s.metrics.EvaporationEffectiveImportance.WithLabelValues(string(r.Type)).Observe(effImp)
		}
		output[i] = searchResult{
			ID:                  r.ID,
			Type:                string(r.Type),
			Content:             r.Content,
			Source:              r.Source,
			Importance:          r.Importance,
			EffectiveImportance: effImp,
			Tags:                r.Tags,
			CreatedAt:           r.CreatedAt,
			UpdatedAt:           r.UpdatedAt,
			Score:               r.Score,
			ValidUntil:          r.ValidUntil,
			AccessCount:         r.AccessCount,
			LastAccessedAt:      r.LastAccessedAt,
			Metadata:            r.Metadata,
			SourceCollection:    collectionOrFallback(r.Collection, collection.CollectionUser),
			SourceType:          sourceTypeFromMetadata(r.Metadata),
		}
	}

	data, err := json.Marshal(output)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}

	// Trajectory logging (async, non-blocking)
	if s.traj != nil {
		items := make([]trajectory.ResultItem, len(output))
		for i, r := range output {
			items[i] = trajectory.ResultItem{ID: r.ID, Content: r.Content, Score: r.Score}
		}
		s.traj.Log(trajectory.Record{
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Operation: "retrieve",
			Query:     query,
			Results:   items,
			Strategy:  "semantic_search",
			LatencyMs: time.Since(start).Milliseconds(),
			Caller:    CallerTypeFromContext(ctx),
			TaskID:    taskID,
		})
	}

	return mcp.NewToolResultText(string(data)), nil
}

// handleList implements the memory_list tool: a filter-only, time-window
// listing that passes straight through to Store.Scroll (no vector search, no
// embedding cost). Superseded/expired entries are excluded by Scroll. Results
// are sorted by created_at ascending and truncated to limit.
func (s *Server) handleList(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ctx, span := tracer.Start(ctx, "engram.memory.list")
	defer span.End()
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpListMemory)...)

	limit := request.GetInt("limit", memory.DefaultListLimit)
	if limit <= 0 {
		limit = memory.DefaultListLimit
	}
	if limit > 100 {
		limit = 100
	}

	// Resolve collection scope. ISOLATION: an isolated caller (e.g. pigo) may
	// ONLY read its own collection; a self-declared collections arg can never
	// widen scope.
	var collections []string
	if own, isolated := isolatedCaller(ctx); isolated {
		collections = []string{own}
	} else if cols := getStringSlice(request, "collections"); len(cols) > 0 {
		for _, col := range cols {
			if _, ok := collection.DefaultRegistry.Get(col); !ok {
				return mcp.NewToolResultError(fmt.Sprintf("unknown collection: %s", col)), nil
			}
		}
		collections = cols
	}

	opts := memory.ListMemoriesOptions{
		TimeStart:   request.GetFloat("time_start", 0),
		TimeEnd:     request.GetFloat("time_end", 0),
		Collections: collections,
		Limit:       limit,
	}

	mems, err := memory.ListMemories(ctx, s.store, opts)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "scroll error")
		return mcp.NewToolResultError(fmt.Sprintf("scroll error: %v", err)), nil
	}

	type listResult struct {
		ID               string   `json:"id"`
		Content          string   `json:"content"`
		Type             string   `json:"type"`
		Importance       float64  `json:"importance"`
		CreatedAt        float64  `json:"created_at"`
		Tags             []string `json:"tags"`
		SourceCollection string   `json:"source_collection"`
	}

	output := make([]listResult, len(mems))
	for i, m := range mems {
		output[i] = listResult{
			ID:               m.ID,
			Content:          m.Content,
			Type:             string(m.Type),
			Importance:       m.Importance,
			CreatedAt:        m.CreatedAt,
			Tags:             m.Tags,
			SourceCollection: collectionOrFallback(m.Collection, collection.CollectionUser),
		}
	}

	data, err := json.Marshal(output)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}
	return mcp.NewToolResultText(string(data)), nil
}

// handleAdd implements the memory.add tool.
func (s *Server) handleAdd(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	addStart := time.Now()
	ctx, span := tracer.Start(ctx, "engram.memory.add")
	defer span.End()
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpUpsertMemory)...)

	content, err := request.RequireString("content")
	if err != nil {
		return mcp.NewToolResultError("content is required"), nil
	}

	memType := memory.MemoryType(request.GetString("type", "event"))
	if !memory.ValidTypes[memType] {
		return mcp.NewToolResultError(fmt.Sprintf("invalid memory type: %s", memType)), nil
	}

	sourceType := request.GetString("source_type", "")

	var importance float64
	var importanceClamped bool
	if s.amacEnabled() {
		importance, importanceClamped = s.amacImportance(memType, request.GetFloat("importance", 0))
		// R6 governance: a directive from Frank (user_input) gets a +1 boost.
		if memType == memory.TypeDirective && sourceType == "user_input" {
			importance++
			if importance > 10 {
				importance = 10
			}
		}
	} else {
		importance = request.GetFloat("importance", 5.0)
		if importance < 1 {
			importance = 1
		}
		if importance > 10 {
			importance = 10
		}
	}

	source := request.GetString("source", "agent")
	tags := getStringSlice(request, "tags")
	validUntil := request.GetFloat("valid_until", 0)
	taskID := request.GetString("task_id", "")

	span.SetAttributes(
		attribute.Int("content.length", len(content)),
		attribute.Int("tags.count", len(tags)),
		attribute.String("type", string(memType)),
		attribute.Float64("importance", importance),
	)

	// Auto-compute TTL if caller didn't explicitly set valid_until.
	// This uses the TTL matrix: type × importance band → duration.
	ttlCfg := memory.DefaultTTLConfig()
	computedValidUntil := memory.ComputeValidUntil(ttlCfg, memType, importance, tags, validUntil)

	// Create the memory
	opts := []memory.Option{
		memory.WithType(memType),
		memory.WithImportance(importance),
		memory.WithSource(source),
		memory.WithTags(tags...),
	}
	if computedValidUntil > 0 {
		opts = append(opts, memory.WithValidUntil(computedValidUntil))
	}
	mem := memory.New(content, opts...)
	mem.Collection = CollectionFromContext(ctx)

	// C1 provenance: soft-require source_type (shared helper). sourceType is
	// also consumed by checkDedup below for source_type-aware merging (C2).
	if mem.Metadata == nil {
		mem.Metadata = map[string]any{}
	}
	if err := s.applyProvenance(mem.Metadata, sourceType, sourceType != "", "memory_add"); err != nil {
		return mcp.NewToolResultError(provenanceMCPMsg(err)), nil
	}

	// D4: pre-admission candidate flow recording. Register a single deferred
	// trajectory write that fires once per call at function exit, tagged with
	// the admission decision reached by the gates below (rate-limit / embed /
	// dedup / insert). Async + non-blocking; replaces the old per-branch
	// update logs.
	admissionDecision := "admitted"
	gateDetails := ""
	var dedupTopScore float64
	if s.traj != nil {
		defer func() {
			s.traj.Log(trajectory.Record{
				Timestamp:         time.Now().UTC().Format(time.RFC3339),
				Operation:         "candidate",
				Content:           content,
				Type:              string(memType),
				Importance:        importance,
				SourceType:        sourceType,
				Tags:              tags,
				AdmissionDecision: admissionDecision,
				GateDetails:       gateDetails,
				DedupTopScore:     dedupTopScore,
				LatencyMs:         time.Since(addStart).Milliseconds(),
				Caller:            CallerTypeFromContext(ctx),
				TaskID:            taskID,
			})
			if s.metrics != nil {
				s.metrics.AdmissionTotal.WithLabelValues(string(memType), admissionDecision).Inc()
			}
		}()
	}

	// A-MAC (R2/R6): annotate importance clamp + identity governance flag, then
	// enforce the per-type write-frequency limit before spending an embedding.
	if s.amacEnabled() {
		if importanceClamped {
			mem.Metadata["importance_clamped"] = true
		}
		if memType == memory.TypeIdentity && sourceType == "reflection" {
			mem.Metadata["needs_review"] = true
		}
		if err := s.rateLimiter.Allow(mem.Collection, memType); err != nil {
			var rle *RateLimitError
			if errors.As(err, &rle) {
				admissionDecision = "rate_limited"
				gateDetails = fmt.Sprintf("rate_limited: retry_after=%ds", rle.RetryAfterSeconds)
				data, _ := json.Marshal(map[string]any{
					"status":              "rate_limited",
					"type":                string(memType),
					"retry_after_seconds": rle.RetryAfterSeconds,
					"message":             rle.Error(),
				})
				return mcp.NewToolResultError(string(data)), nil
			}
		}
	}

	// Embed content
	embedStart := time.Now()
	vec, err := s.embedder.Embed(ctx, content)
	if s.metrics != nil {
		s.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "embedding error")
		admissionDecision = "error"
		gateDetails = fmt.Sprintf("embed_error: %v", err)
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	// Check for duplicates (child span)
	dedupResult, dupErr := s.checkDedup(ctx, vec, content, sourceType, memType)
	if dupErr != nil {
		span.RecordError(dupErr)
		span.SetStatus(codes.Error, "dedup check error")
		log.Printf("[ERROR] engram memory_add: dedup check failed: %v", dupErr)
		admissionDecision = "error"
		gateDetails = fmt.Sprintf("dedup_error: %v", dupErr)
		return mcp.NewToolResultError(fmt.Sprintf("dedup check error: %v", dupErr)), nil
	}
	dedupTopScore = dedupResult.TopScore
	if dedupResult.DupFound {
		span.SetAttributes(attribute.Bool("dedup.hit", true))
		if s.metrics != nil {
			s.metrics.DedupHits.WithLabelValues(mem.Collection, dedupTypeLabel(dedupResult.Threshold)).Inc()
			s.metrics.MemoryOps.WithLabelValues("add", mem.Collection, sourceType).Inc()
		}
		admissionDecision = "dedup_rejected"
		var dup struct {
			Existing struct {
				ID    string  `json:"id"`
				Score float64 `json:"score"`
			} `json:"existing"`
		}
		_ = json.Unmarshal(dedupResult.DupData, &dup)
		gateDetails = fmt.Sprintf("dedup score=%.4f against id=%s", dup.Existing.Score, dup.Existing.ID)
		return mcp.NewToolResultText(string(dedupResult.DupData)), nil
	}

	span.SetAttributes(attribute.Bool("dedup.hit", false))

	// Insert
	if err := s.store.Insert(ctx, mem, vec); err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "insert error")
		admissionDecision = "error"
		gateDetails = fmt.Sprintf("insert_error: %v", err)
		return mcp.NewToolResultError(fmt.Sprintf("insert error: %v", err)), nil
	}

	// Return the created memory
	type addResult struct {
		Status     string         `json:"status"`
		Memory     *memory.Memory `json:"memory"`
		Advisories []Advisory     `json:"advisories,omitempty"`
	}
	// Write Discipline Checkpoints (advisory-only, non-blocking): CP1→CP4.
	var advisories []Advisory
	if s.writeCheckpointsEnabled() {
		if a := s.cp1DedupAdvisory(dedupResult.Candidates, memType); a != nil {
			advisories = append(advisories, *a)
			if s.metrics != nil {
				s.metrics.CheckpointTotal.WithLabelValues("cp1", "advisory").Inc()
			}
		} else if s.metrics != nil {
			s.metrics.CheckpointTotal.WithLabelValues("cp1", "clean").Inc()
		}
		if a := s.cp2Importance(mem.Collection, importance); a != nil {
			advisories = append(advisories, *a)
			if s.metrics != nil {
				s.metrics.CheckpointTotal.WithLabelValues("cp2", "advisory").Inc()
			}
		} else if s.metrics != nil {
			s.metrics.CheckpointTotal.WithLabelValues("cp2", "clean").Inc()
		}
		if a := s.cp3RateLimitAdvisory(mem.Collection, memType); a != nil {
			advisories = append(advisories, *a)
		}
		advisories = append(advisories, s.cp4Content(content, tags, sourceType)...)
		for _, a := range advisories {
			logAdvisory(a)
		}
	}
	result := addResult{
		Status:     "created",
		Memory:     mem,
		Advisories: advisories,
	}
	data, err := json.Marshal(result)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}

	// Trajectory logging happens via the deferred candidate record registered
	// above (D4); admissionDecision remains "admitted" on this success path.

	if s.metrics != nil {
		s.metrics.MemoryOps.WithLabelValues("add", mem.Collection, sourceType).Inc()
	}

	return mcp.NewToolResultText(string(data)), nil
}

// DedupResult carries the outcome of a dedup check. When DupFound is true,
// DupData holds the JSON response for the duplicate case. Candidates holds the
// top-3 candidates from the dedup search (reused by CP1 for the near-duplicate
// advisory — zero extra search cost).
type DedupResult struct {
	DupFound   bool
	DupData    []byte
	Candidates []memory.ScoredMemory
	// TopScore is the highest dedup-search similarity score (0 when no
	// candidates); carried out so callers can record it on the trajectory.
	TopScore float64
	// Threshold is the dedup similarity threshold actually used for this
	// decision (from resolveDedupThreshold). Carried out so callers label the
	// dedup-hit metric with the real threshold instead of a hardcoded value.
	Threshold float64
}

// dedupTypeLabel formats the dedup_type metric label deterministically from the
// threshold actually used for the decision. The value is always
// server_side_NNN where NNN = round(threshold*100) zero-padded to 3 digits
// (e.g. 0.90 -> "server_side_090", 0.92 -> "server_side_092", 0.78 ->
// "server_side_078"). Fixing to an integer percent bounds label cardinality.
func dedupTypeLabel(threshold float64) string {
	return fmt.Sprintf("server_side_%03d", int(math.Round(threshold*100)))
}

// checkDedup runs deduplication check as a child span. Returns a *DedupResult;
// DupFound is true when a content duplicate was found.
// incomingSourceType is the source_type of the memory being added; it drives
// source_type-aware provenance merging (C2) when a content duplicate is found.
func (s *Server) checkDedup(ctx context.Context, vec []float32, content string, incomingSourceType string, memType memory.MemoryType) (*DedupResult, error) {
	ctx, span := tracer.Start(ctx, "engram.memory.dedup_check")
	defer span.End()
	// dedup_check is a sub-step of the upsert flow, so it carries upsert_memory.
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpUpsertMemory)...)
	start := time.Now()

	threshold := s.resolveDedupThreshold(memType)
	span.SetAttributes(
		attribute.Int("query.length", len(content)),
		attribute.Float64("threshold", threshold),
	)

	resolvedCol := CollectionFromContext(ctx)
	dupeResults, err := s.store.Search(ctx, vec, memory.SearchOptions{
		Limit:   3,
		Filters: []memory.Filter{{Field: "collection", Op: memory.OpIn, Value: []string{resolvedCol}}},
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, "dedup search error")
		span.SetAttributes(
			attribute.String("decision", "add"),
			attribute.Float64("top_score", 0),
			attribute.Int64("latency_ms", time.Since(start).Milliseconds()),
		)
		return nil, err
	}

	topScore := 0.0
	if len(dupeResults) > 0 {
		topScore = dupeResults[0].Score
		if s.metrics != nil {
			s.metrics.SearchTopScore.WithLabelValues(resolvedCol).Observe(topScore)
		}
	}

	dup := memory.IsDuplicate(dupeResults, threshold)

	decision := "add"
	if dup != nil {
		decision = "skip"
	}

	span.SetAttributes(
		attribute.Float64("top_score", topScore),
		attribute.String("decision", decision),
		attribute.Int64("latency_ms", time.Since(start).Milliseconds()),
	)

	if dup != nil {
		type dupResult struct {
			Status   string `json:"status"`
			Message  string `json:"message"`
			Existing struct {
				ID                 string  `json:"id"`
				Content            string  `json:"content"`
				Score              float64 `json:"score"`
				SourceType         string  `json:"source_type,omitempty"`
				ProvenanceMerged   bool    `json:"provenance_merged,omitempty"`
				IncomingSourceType string  `json:"incoming_source_type,omitempty"`
			} `json:"existing"`
		}
		result := dupResult{
			Status:  "duplicate",
			Message: "A very similar memory already exists. Skipped.",
		}
		result.Existing.ID = dup.ID
		result.Existing.Content = dup.Content
		result.Existing.Score = dup.Score

		existingST := sourceTypeFromMeta(dup.Metadata)
		result.Existing.SourceType = existingST

		switch {
		case incomingSourceType == "" || incomingSourceType == string(memory.SourceTypeUnknown):
			// C2 §5.2: unknown/empty incoming adds no provenance value — standard dedup.
		case existingST == "" || existingST == string(memory.SourceTypeUnknown):
			// C2 §5.3: legacy existing memory — opportunistic backfill of source_type.
			// This is initial classification, not a merge, so no provenance_history.
			md := cloneMetadata(dup.Metadata)
			md["source_type"] = incomingSourceType
			if err := s.store.Update(ctx, dup.ID, map[string]any{
				"metadata":   md,
				"updated_at": float64(time.Now().Unix()),
			}); err == nil {
				result.Existing.SourceType = incomingSourceType
			}
		case incomingSourceType != existingST:
			// C2 §4.3: different source_type — merge provenance.
			primary, merged, err := s.provenanceMerge(ctx, dup, incomingSourceType)
			if err == nil && merged {
				result.Status = "duplicate_provenance_merged"
				result.Message = fmt.Sprintf(
					"Content duplicate found. Provenance merged: added %q to existing memory (primary: %q).",
					incomingSourceType, primary)
				result.Existing.SourceType = primary
				result.Existing.ProvenanceMerged = true
				result.Existing.IncomingSourceType = incomingSourceType
			}
		}

		data, _ := json.Marshal(result)
		return &DedupResult{DupFound: true, DupData: data, Candidates: dupeResults, TopScore: topScore, Threshold: threshold}, nil
	}

	return &DedupResult{DupFound: false, Candidates: dupeResults, TopScore: topScore, Threshold: threshold}, nil
}

// provenanceMerge updates an existing memory's provenance metadata when a
// content-duplicate arrives with a different source_type (C2 §4.4). The
// existing content is not modified; only metadata.provenance_history and the
// primary metadata.source_type (promoted to highest trust) are updated via a
// payload-only store.Update (no re-embedding). Returns the new primary
// source_type and whether a merge was performed.
func (s *Server) provenanceMerge(ctx context.Context, existing *memory.ScoredMemory, incomingSourceType string) (string, bool, error) {
	ctx, span := tracer.Start(ctx, "engram.memory.provenance_merge")
	defer span.End()
	// provenance_merge is a sub-step of the upsert flow, so it carries upsert_memory.
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpUpsertMemory)...)

	existingST := sourceTypeFromMeta(existing.Metadata)
	history := getProvenanceHistory(existing.Metadata)

	// Idempotency: this source is already recorded (C2 §5.1 / test Idempotent).
	if memory.HasSourceType(history, incomingSourceType) {
		span.SetAttributes(attribute.String("decision", "provenance_merge_skip_idempotent"))
		return "", false, nil
	}
	// Safety cap (C2 §5.4): don't grow provenance_history unboundedly.
	if len(history) >= memory.MaxProvenanceHistory {
		span.SetAttributes(attribute.String("decision", "provenance_merge_skip_cap"))
		return "", false, nil
	}

	history = append(history, memory.ProvenanceEntry{
		SourceType:   incomingSourceType,
		MergedAt:     time.Now().Unix(),
		ContentScore: existing.Score,
	})

	sources := make([]string, 0, len(history)+1)
	if existingST != "" {
		sources = append(sources, existingST)
	}
	for _, h := range history {
		sources = append(sources, h.SourceType)
	}
	primary := memory.HighestTrustSource(sources)

	md := cloneMetadata(existing.Metadata)
	md["provenance_history"] = memory.ProvenanceHistoryToAny(history)
	md["source_type"] = primary

	fields := map[string]any{
		"metadata":   md,
		"updated_at": float64(time.Now().Unix()),
	}

	span.SetAttributes(
		attribute.String("decision", "provenance_merge"),
		attribute.String("existing_source_type", existingST),
		attribute.String("incoming_source_type", incomingSourceType),
		attribute.String("primary_source_type", primary),
		attribute.Bool("trust_promoted", primary != existingST),
		attribute.Float64("top_score", existing.Score),
	)

	if err := s.store.Update(ctx, existing.ID, fields); err != nil {
		span.RecordError(err)
		return "", false, err
	}
	return primary, true, nil
}

// cloneMetadata returns a shallow copy of a metadata map (nil -> new empty map)
// so callers can RMW a nested "metadata" payload without mutating the source or
// dropping sibling keys. Mirrors qdrant SetPayload replacing the key wholesale.
func cloneMetadata(m map[string]any) map[string]any {
	out := make(map[string]any, len(m)+2)
	for k, v := range m {
		out[k] = v
	}
	return out
}

// sourceTypeFromMeta extracts the source_type string from a metadata map.
// Returns "" when absent.
func sourceTypeFromMeta(metadata map[string]any) string {
	if metadata == nil {
		return ""
	}
	st, _ := metadata["source_type"].(string)
	return st
}

// getProvenanceHistory extracts and deserializes the provenance_history from a
// metadata map. Handles both the in-process []memory.ProvenanceEntry form and
// the JSON round-tripped []any-of-map form (as returned by the store).
func getProvenanceHistory(metadata map[string]any) []memory.ProvenanceEntry {
	if metadata == nil {
		return nil
	}
	raw, ok := metadata["provenance_history"]
	if !ok || raw == nil {
		return nil
	}
	switch v := raw.(type) {
	case []memory.ProvenanceEntry:
		return v
	case []any:
		out := make([]memory.ProvenanceEntry, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				continue
			}
			var e memory.ProvenanceEntry
			e.SourceType, _ = m["source_type"].(string)
			switch t := m["merged_at"].(type) {
			case float64:
				e.MergedAt = int64(t)
			case int64:
				e.MergedAt = t
			case int:
				e.MergedAt = int64(t)
			}
			e.ContentScore, _ = m["content_score"].(float64)
			out = append(out, e)
		}
		return out
	default:
		return nil
	}
}

// handleUpdate implements the memory.update tool.
func (s *Server) handleUpdate(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	oldContent, err := request.RequireString("old_content")
	if err != nil {
		return mcp.NewToolResultError("old_content is required"), nil
	}
	newContent, err := request.RequireString("new_content")
	if err != nil {
		return mcp.NewToolResultError("new_content is required"), nil
	}

	threshold := request.GetFloat("similarity_threshold", 0.7)
	dryRun := false
	if args := request.GetArguments(); args != nil {
		if v, ok := args["dry_run"]; ok {
			if b, ok := v.(bool); ok {
				dryRun = b
			}
		}
	}
	memType := memory.MemoryType(request.GetString("type", "event"))
	importance := request.GetFloat("importance", 5.0)
	if importance < 1 {
		importance = 1
	}
	if importance > 10 {
		importance = 10
	}
	tags := getStringSlice(request, "tags")
	validUntil := request.GetFloat("valid_until", 0)

	// Safety guardrail: memory_update always searches up to 20 candidates; a low
	// threshold has caused 3 mass-delete incidents. Require >= 0.85.
	if threshold < 0.85 {
		return mcp.NewToolResultError(fmt.Sprintf(
			"memory_update: similarity_threshold %.2f is unsafe (3 prior mass-delete incidents). "+
				"Use threshold >= 0.85 (recommended: 0.92 for single targeted replacement). "+
				"For batch operations use memory_delete + memory_add instead.",
			threshold,
		)), nil
	}

	// Step 1: Embed both contents upfront in a single batch call
	embedStart := time.Now()
	vecs, err := s.embedder.EmbedBatch(ctx, []string{oldContent, newContent})
	if s.metrics != nil {
		s.metrics.EmbedDuration.Observe(time.Since(embedStart).Seconds())
	}
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}
	oldVec, newVec := vecs[0], vecs[1]

	// ISOLATION: scope the candidate search to the isolated caller's own
	// collection so cross-collection targets are invisible (mirrors REST 403).
	var updFilters []memory.Filter
	if own, isolated := isolatedCaller(ctx); isolated {
		updFilters = append(updFilters, memory.Filter{Field: "collection", Op: memory.OpIn, Value: []string{own}})
	}

	searchResults, err := s.store.Search(ctx, oldVec, memory.SearchOptions{
		Limit:   20,
		Filters: updFilters,
	})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("search error: %v", err)), nil
	}

	// Filter by similarity threshold
	var toDelete []string
	var deletedMemories []memory.ScoredMemory
	for _, r := range searchResults {
		if r.Score >= threshold {
			toDelete = append(toDelete, r.ID)
			deletedMemories = append(deletedMemories, r)
		}
	}

	// Format deleted items for response
	type deletedItem struct {
		ID         string  `json:"id"`
		Content    string  `json:"content"`
		Score      float64 `json:"score"`
		ValidUntil float64 `json:"valid_until,omitempty"`
	}
	deleted := make([]deletedItem, len(deletedMemories))
	for i, d := range deletedMemories {
		deleted[i] = deletedItem{
			ID:         d.ID,
			Content:    d.Content,
			Score:      d.Score,
			ValidUntil: d.ValidUntil,
		}
	}

	// dry_run: return preview without making any changes
	if dryRun {
		type dryRunResult struct {
			Status           string        `json:"status"`
			WouldDeleteCount int           `json:"would_delete_count"`
			WouldDelete      []deletedItem `json:"would_delete"`
			NewContent       string        `json:"new_content"`
		}
		result := dryRunResult{
			Status:           "dry_run",
			WouldDeleteCount: len(deleted),
			WouldDelete:      deleted,
			NewContent:       newContent,
		}
		data, _ := json.Marshal(result)
		return mcp.NewToolResultText(string(data)), nil
	}

	// Step 2: Delete matching memories
	deletedCount := 0
	if len(toDelete) > 0 {
		deletedCount, err = s.store.Delete(ctx, toDelete)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("delete error: %v", err)), nil
		}
	}

	// Step 3: Create new memory
	// Inherit valid_until from old memory if user didn't explicitly set a new value
	if validUntil == 0 && len(deletedMemories) > 0 {
		validUntil = deletedMemories[0].ValidUntil
	}
	// Auto-compute TTL if still unset (neither explicit nor inherited)
	ttlCfg := memory.DefaultTTLConfig()
	computedValidUntil := memory.ComputeValidUntil(ttlCfg, memType, importance, tags, validUntil)
	opts := []memory.Option{
		memory.WithType(memType),
		memory.WithImportance(importance),
		memory.WithTags(tags...),
	}
	if computedValidUntil > 0 {
		opts = append(opts, memory.WithValidUntil(computedValidUntil))
	}
	mem := memory.New(newContent, opts...)
	mem.Collection = CollectionFromContext(ctx)

	// C1 provenance (shared helper).
	sourceTypeU := request.GetString("source_type", "")
	if mem.Metadata == nil {
		mem.Metadata = map[string]any{}
	}
	if err := s.applyProvenance(mem.Metadata, sourceTypeU, sourceTypeU != "", "memory_update"); err != nil {
		return mcp.NewToolResultError(provenanceMCPMsg(err)), nil
	}

	// R6 governance: when replacing an identity/directive, record the old
	// content on the new memory for audit (best-effort; only the top match).
	if s.amacEnabled() && (memType == memory.TypeIdentity || memType == memory.TypeDirective) && len(deletedMemories) > 0 {
		mem.Metadata["superseded_content"] = deletedMemories[0].Content
	}

	if err := s.store.Insert(ctx, mem, newVec); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("insert error: %v", err)), nil
	}

	if s.metrics != nil {
		s.metrics.MemoryOps.WithLabelValues("update", mem.Collection, sourceTypeU).Inc()
	}

	type updateResult struct {
		Status       string         `json:"status"`
		DeletedCount int            `json:"deleted_count"`
		Deleted      []deletedItem  `json:"deleted"`
		NewMemory    *memory.Memory `json:"new_memory"`
		Advisories   []Advisory     `json:"advisories,omitempty"`
	}

	// Write Discipline Checkpoints (R9): CP2 (importance trend) and CP4
	// (content quality) apply on update; CP1/CP3 do not.
	var advisories []Advisory
	if s.writeCheckpointsEnabled() {
		if a := s.cp2Importance(mem.Collection, importance); a != nil {
			advisories = append(advisories, *a)
		}
		advisories = append(advisories, s.cp4Content(newContent, tags, sourceTypeU)...)
		for _, a := range advisories {
			logAdvisory(a)
		}
	}

	result := updateResult{
		Status:       "updated",
		DeletedCount: deletedCount,
		Deleted:      deleted,
		NewMemory:    mem,
		Advisories:   advisories,
	}

	data, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(data)), nil
}

// handleDelete implements the memory.delete tool.
func (s *Server) handleDelete(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	ctx, span := tracer.Start(ctx, "engram.memory.delete")
	defer span.End()
	span.SetAttributes(semconv.GenAIAttrs(semconv.OpDeleteMemory)...)

	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError("query is required"), nil
	}

	threshold := request.GetFloat("similarity_threshold", 0.7)
	limit := request.GetInt("limit", 20)
	dryRun := false
	if args := request.GetArguments(); args != nil {
		if v, ok := args["dry_run"]; ok {
			if b, ok := v.(bool); ok {
				dryRun = b
			}
		}
	}
	if limit < 1 {
		limit = 1
	}

	// Safety guardrail: batch deletes (limit > 1) with a low threshold have caused
	// mass-delete incidents. Require threshold >= 0.85 when deleting multiple memories.
	if limit > 1 && threshold < 0.85 {
		return mcp.NewToolResultError(fmt.Sprintf(
			"memory_delete: limit=%d with similarity_threshold=%.2f is unsafe (3 prior mass-delete incidents). "+
				"Use threshold >= 0.85, or set limit=1 for single targeted deletion.",
			limit, threshold,
		)), nil
	}

	// Embed query
	vec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	// Search for matching memories.
	// ISOLATION: scope the candidate search to the isolated caller's own
	// collection so cross-collection targets cannot be deleted (mirrors REST 403).
	var delFilters []memory.Filter
	if own, isolated := isolatedCaller(ctx); isolated {
		delFilters = append(delFilters, memory.Filter{Field: "collection", Op: memory.OpIn, Value: []string{own}})
	}
	results, err := s.store.Search(ctx, vec, memory.SearchOptions{
		Limit:   limit,
		Filters: delFilters,
	})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("search error: %v", err)), nil
	}

	// Filter by similarity threshold
	var toDelete []string
	type deletedItem struct {
		ID         string  `json:"id"`
		Content    string  `json:"content"`
		Score      float64 `json:"score"`
		ValidUntil float64 `json:"valid_until,omitempty"`
	}
	var deletedItems []deletedItem

	for _, r := range results {
		if r.Score >= threshold {
			toDelete = append(toDelete, r.ID)
			deletedItems = append(deletedItems, deletedItem{
				ID:         r.ID,
				Content:    r.Content,
				Score:      r.Score,
				ValidUntil: r.ValidUntil,
			})
		}
	}

	if len(toDelete) == 0 {
		type deleteResult struct {
			Status       string        `json:"status"`
			DeletedCount int           `json:"deleted_count"`
			Deleted      []deletedItem `json:"deleted"`
		}
		result := deleteResult{
			Status:       "no_matches",
			DeletedCount: 0,
			Deleted:      []deletedItem{},
		}
		data, _ := json.Marshal(result)
		return mcp.NewToolResultText(string(data)), nil
	}

	// dry_run: return preview without deleting
	if dryRun {
		type dryRunResult struct {
			Status           string        `json:"status"`
			WouldDeleteCount int           `json:"would_delete_count"`
			WouldDelete      []deletedItem `json:"would_delete"`
		}
		result := dryRunResult{
			Status:           "dry_run",
			WouldDeleteCount: len(deletedItems),
			WouldDelete:      deletedItems,
		}
		data, _ := json.Marshal(result)
		return mcp.NewToolResultText(string(data)), nil
	}

	// Delete
	deletedCount, err := s.store.Delete(ctx, toDelete)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("delete error: %v", err)), nil
	}

	if s.metrics != nil {
		s.metrics.MemoryOps.WithLabelValues("delete", CollectionFromContext(ctx), "unknown").Inc()
	}

	type deleteResult struct {
		Status       string        `json:"status"`
		DeletedCount int           `json:"deleted_count"`
		Deleted      []deletedItem `json:"deleted"`
	}
	result := deleteResult{
		Status:       "deleted",
		DeletedCount: deletedCount,
		Deleted:      deletedItems,
	}
	data, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(data)), nil
}

// =============================================================================
// Reflection Handlers
// =============================================================================

// handleReflectionCheck implements the reflection_check tool.
// It evaluates whether the Reflection Engine should run now without executing it.
func (s *Server) handleReflectionCheck(ctx context.Context, _ mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if _, isolated := isolatedCaller(ctx); isolated {
		return mcp.NewToolResultError("reflection_check is a global-scope action not permitted for isolated callers"), nil
	}
	eng := reflection.NewEngine(s.store, s.embedder, s.reflectionConfig())
	result, err := eng.Check(ctx)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("reflection check error: %v", err)), nil
	}
	data, err := json.Marshal(result)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}
	return mcp.NewToolResultText(string(data)), nil
}

// handleReflectionRun implements the reflection_run tool.
// It executes one reflection cycle, respecting rate limits and dry_run mode.
func (s *Server) handleReflectionRun(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if _, isolated := isolatedCaller(ctx); isolated {
		return mcp.NewToolResultError("reflection_run is a global-scope action not permitted for isolated callers"), nil
	}
	dryRun := false
	if args := request.GetArguments(); args != nil {
		if v, ok := args["dry_run"]; ok {
			if b, ok := v.(bool); ok {
				dryRun = b
			}
		}
	}

	cfg := s.reflectionConfig()
	cfg.DryRun = dryRun

	// dry_run stays synchronous: a preview must return the RunResult directly.
	if dryRun {
		eng := reflection.NewEngine(s.store, s.embedder, cfg)
		result, err := eng.Run(ctx)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("reflection run error: %v", err)), nil
		}
		data, err := json.Marshal(result)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
		}
		return mcp.NewToolResultText(string(data)), nil
	}

	// Real runs are launched asynchronously via the single-flight runner so the
	// 30s MCP bridge timeout cannot kill a minutes-long LLM cycle.
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

	started, runID, startedAt := s.reflectionRunner.start(defaultRun, recordMetrics)
	if !started {
		resp := map[string]any{
			"already_running": true,
			"run_id":          runID,
			"started_at":      startedAt.Format(time.RFC3339),
		}
		data, _ := json.Marshal(resp)
		return mcp.NewToolResultText(string(data)), nil
	}
	resp := map[string]any{
		"started": true,
		"run_id":  runID,
		"dry_run": false,
	}
	data, _ := json.Marshal(resp)
	return mcp.NewToolResultText(string(data)), nil
}

// =============================================================================
// Helpers
// =============================================================================

// handleReflectionRunEvent implements the reflection_run_event tool (W17 v1.1
// batch 2). Event-driven single-event reflection triggered by task failures
// or user corrections — bypasses accumulator thresholds and daily quotas.
//
// registration; retained pending wiring of the reflection_run_event MCP tool.
// Not dead — RunSingleEvent is exercised by reflection engine tests.
//
//nolint:unused // Staged in commit 8dd5200 (W17 v1.1 batch 2) ahead of tool
func (s *Server) handleReflectionRunEvent(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	if _, isolated := isolatedCaller(ctx); isolated {
		return mcp.NewToolResultError("reflection_run_event is a global-scope action not permitted for isolated callers"), nil
	}
	cause, err := request.RequireString("cause")
	if err != nil {
		return mcp.NewToolResultError("cause is required"), nil
	}
	summary, err := request.RequireString("summary")
	if err != nil {
		return mcp.NewToolResultError("summary is required"), nil
	}

	// Validate cause.
	var tc reflection.TriggerCause
	switch cause {
	case string(reflection.TriggerTaskFailure):
		tc = reflection.TriggerTaskFailure
	case string(reflection.TriggerUserCorrection):
		tc = reflection.TriggerUserCorrection
	case string(reflection.TriggerExternalEvent):
		tc = reflection.TriggerExternalEvent
	default:
		return mcp.NewToolResultError(fmt.Sprintf("invalid cause %q (allowed: task_failure, user_correction, external_event)", cause)), nil
	}

	evidenceIDs := getStringSlice(request, "evidence_ids")
	extraTags := getStringSlice(request, "extra_tags")

	importance := 0.0
	if args := request.GetArguments(); args != nil {
		if v, ok := args["importance"]; ok {
			if f, ok := v.(float64); ok {
				importance = f
			}
		}
	}

	dryRun := false
	if args := request.GetArguments(); args != nil {
		if v, ok := args["dry_run"]; ok {
			if b, ok := v.(bool); ok {
				dryRun = b
			}
		}
	}

	cfg := s.reflectionConfig()
	cfg.DryRun = dryRun

	eng := reflection.NewEngine(s.store, s.embedder, cfg)
	result, err := eng.RunSingleEvent(ctx, reflection.SingleEventInput{
		Cause:       tc,
		Summary:     summary,
		EvidenceIDs: evidenceIDs,
		Importance:  importance,
		ExtraTags:   extraTags,
	})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("reflection run_event error: %v", err)), nil
	}
	data, err := json.Marshal(result)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}
	return mcp.NewToolResultText(string(data)), nil
}

// collectionOrFallback returns col if non-empty, otherwise fallback.
// Used to surface per-memory collection labels while remaining backward-compatible
// with memories written before the collection field was introduced.
func collectionOrFallback(col, fallback string) string {
	if col != "" {
		return col
	}
	return fallback
}

// reflectionConfig builds the reflection engine config from the loaded server
// config, wiring provenance (ENGRAM_PROVENANCE_MODE) into ProvenanceFilter so
// that strict mode reaches evidence filtering and write-back enforcement,
// instead of silently downgrading to default via the legacy fields.
func (s *Server) reflectionConfig() reflection.Config {
	c := reflection.DefaultConfig()
	if s.cfg != nil {
		c.RequireProvenance = s.cfg.RequireProvenance   //nolint:staticcheck // backward compat: migration to ProvenanceFilter tracked separately
		c.AllowedProvenances = s.cfg.AllowedProvenances //nolint:staticcheck // backward compat: migration to ProvenanceFilter tracked separately
		c.ProvenanceFilter = s.cfg.ProvenanceFilterConfig()
		if s.cfg.ReflectionMode != "" {
			c.Mode = s.cfg.ReflectionMode
		}
		if s.cfg.DialecticTimeout > 0 {
			c.DialecticTimeout = s.cfg.DialecticTimeout
		}
	}
	return c
}

// strictProvenanceRejects reports whether an explicit source_type must be
// rejected under ENGRAM_PROVENANCE_MODE=strict: the "unknown" sentinel is never
// acceptable, and any value outside a configured non-empty allow-list is
// rejected (so it cannot be persisted and silently filtered at read time).
func (s *Server) strictProvenanceRejects(sourceType string) bool {
	if s.cfg == nil || s.cfg.ProvenanceMode != "strict" {
		return false
	}
	if sourceType == string(memory.DefaultSourceType) {
		return true
	}
	if len(s.cfg.AllowedProvenances) > 0 && !containsString(s.cfg.AllowedProvenances, sourceType) {
		return true
	}
	return false
}

func containsString(list []string, v string) bool {
	for _, s := range list {
		if s == v {
			return true
		}
	}
	return false
}

// sourceTypeFromMetadata extracts source_type from a metadata map, returning ""
// if absent or not a string.
func sourceTypeFromMetadata(md map[string]any) string {
	if md == nil {
		return ""
	}
	if st, ok := md["source_type"].(string); ok {
		return st
	}
	return ""
}

// getStringSlice extracts a []string from the request arguments.
func getStringSlice(request mcp.CallToolRequest, key string) []string {
	args := request.GetArguments()
	if args == nil {
		return nil
	}
	val, ok := args[key]
	if !ok {
		return nil
	}
	arr, ok := val.([]interface{})
	if !ok {
		return nil
	}
	result := make([]string, 0, len(arr))
	for _, item := range arr {
		if s, ok := item.(string); ok {
			result = append(result, s)
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

// =============================================================================
// A-MAC helpers (Adaptive Memory Admission Control, spec v1)
// =============================================================================

// amacEnabled reports whether A-MAC admission control is active.
func (s *Server) amacEnabled() bool {
	return s.cfg != nil && s.cfg.AMACEnabled
}

// importanceDefaults returns the configured A-MAC per-type importance defaults,
// falling back to package defaults when the config lacks them (e.g. in tests).
func (s *Server) importanceDefaults() map[memory.MemoryType]float64 {
	if s.cfg != nil && s.cfg.ImportanceDefaults != nil {
		return s.cfg.ImportanceDefaults
	}
	return config.DefaultImportanceDefaults()
}

// importanceBounds returns the configured A-MAC per-type [min,max] bounds,
// falling back to package defaults when absent.
func (s *Server) importanceBounds() map[memory.MemoryType][2]float64 {
	if s.cfg != nil && s.cfg.ImportanceBounds != nil {
		return s.cfg.ImportanceBounds
	}
	return config.DefaultImportanceBounds()
}

// amacImportance applies the A-MAC per-type importance default (when the caller
// omitted importance, i.e. provided <= 0) and clamps into the per-type bounds.
// Returns the resolved importance and whether a clamp occurred.
func (s *Server) amacImportance(memType memory.MemoryType, provided float64) (float64, bool) {
	imp := provided
	if imp <= 0 {
		imp = s.importanceDefaults()[memType]
	}
	b := s.importanceBounds()[memType]
	lo, hi := b[0], b[1]
	clamped := false
	if lo > 0 && imp < lo {
		imp, clamped = lo, true
	}
	if hi > 0 && imp > hi {
		imp, clamped = hi, true
	}
	return imp, clamped
}

// resolveDedupThreshold returns the dedup threshold for memType. A runtime
// per-type override (memory_apply_config) wins; otherwise when A-MAC is enabled
// the per-type config/default applies; otherwise the legacy global threshold
// (with its runtime override) is used.
func (s *Server) resolveDedupThreshold(memType memory.MemoryType) float64 {
	// R4/§4.3: ALL A-MAC behavior (per-type config AND the memory_apply_config
	// runtime override) is gated by the flag as a unit. With A-MAC off, this
	// returns exactly the legacy global threshold (incl. its runtime override).
	if !s.amacEnabled() {
		return s.overrides.getDedupThreshold(s.dedupThreshold)
	}
	if v, ok := s.overrides.getTypeDedupThreshold(memType); ok {
		return v
	}
	if s.cfg.DedupThresholds != nil {
		if v, ok := s.cfg.DedupThresholds[memType]; ok && v > 0 {
			return v
		}
	}
	return memory.DedupThresholdForType(memType)
}
