// Package server implements MCP and REST API transports for Engram.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"

	"github.com/mark3labs/mcp-go/mcp"
	mcpserver "github.com/mark3labs/mcp-go/server"

	"github.com/anthropics/engram/pkg/config"
	"github.com/anthropics/engram/pkg/embedding"
	"github.com/anthropics/engram/pkg/memory"
)

// Server wraps the MCP server with Engram's memory operations.
type Server struct {
	store          memory.Store
	embedder       embedding.Embedder
	weights        memory.ScoringWeights
	decay          memory.DecayConfig
	mmrLambda      float64
	dedupThreshold float64
	mcpServer      *mcpserver.MCPServer
}

// NewServer creates a new Engram MCP server with all tools registered.
func NewServer(store memory.Store, embedder embedding.Embedder, cfg *config.Config) *Server {
	s := &Server{
		store:          store,
		embedder:       embedder,
		weights:        cfg.Weights,
		decay:          cfg.Decay,
		mmrLambda:      cfg.MMRLambda,
		dedupThreshold: cfg.DedupThreshold,
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

// registerTools adds all 4 memory tools to the MCP server.
func (s *Server) registerTools() {
	// Tool 1: memory_search
	searchTool := mcp.NewTool("memory_search",
		mcp.WithDescription("Semantic search over stored memories. Returns scored results combining relevance, recency, and importance."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query text for semantic similarity matching.")),
		mcp.WithNumber("limit", mcp.Description("Maximum number of results to return. Default: 5.")),
		mcp.WithArray("types", mcp.Description("Filter by memory types (identity, event, insight, directive)."), mcp.WithStringItems()),
		mcp.WithArray("tags", mcp.Description("Filter by tags. Memories must have at least one matching tag."), mcp.WithStringItems()),
		mcp.WithNumber("time_start", mcp.Description("Filter memories created after this Unix timestamp.")),
		mcp.WithNumber("time_end", mcp.Description("Filter memories created before this Unix timestamp.")),
	)
	s.mcpServer.AddTool(searchTool, s.handleSearch)

	// Tool 2: memory_add
	addTool := mcp.NewTool("memory_add",
		mcp.WithDescription("Store a new memory. Automatically deduplicates against existing memories."),
		mcp.WithString("content", mcp.Required(), mcp.Description("The memory content text.")),
		mcp.WithString("type", mcp.Description("Memory type."), mcp.Enum("identity", "event", "insight", "directive")),
		mcp.WithNumber("importance", mcp.Description("Importance score from 1-10. Default: 5.")),
		mcp.WithArray("tags", mcp.Description("Tags for classification."), mcp.WithStringItems()),
		mcp.WithString("source", mcp.Description("Source of the memory: user, agent, or system. Default: agent.")),
	)
	s.mcpServer.AddTool(addTool, s.handleAdd)

	// Tool 3: memory_update
	updateTool := mcp.NewTool("memory_update",
		mcp.WithDescription("Update memories by semantic search. Finds old memories matching old_content, deletes them, and stores new_content."),
		mcp.WithString("old_content", mcp.Required(), mcp.Description("Search query to find old memories to replace.")),
		mcp.WithString("new_content", mcp.Required(), mcp.Description("New memory content to store.")),
		mcp.WithString("type", mcp.Description("Memory type for the new memory."), mcp.Enum("identity", "event", "insight", "directive")),
		mcp.WithNumber("importance", mcp.Description("Importance score for the new memory (1-10). Default: 5.")),
		mcp.WithArray("tags", mcp.Description("Tags for the new memory."), mcp.WithStringItems()),
		mcp.WithNumber("similarity_threshold", mcp.Description("Minimum cosine similarity for deletion. Default: 0.7.")),
	)
	s.mcpServer.AddTool(updateTool, s.handleUpdate)

	// Tool 4: memory_delete
	deleteTool := mcp.NewTool("memory_delete",
		mcp.WithDescription("Delete memories by semantic search. Finds memories matching the query above the similarity threshold and removes them."),
		mcp.WithString("query", mcp.Required(), mcp.Description("Search query to find memories to delete.")),
		mcp.WithNumber("similarity_threshold", mcp.Description("Minimum cosine similarity for deletion. Default: 0.7.")),
		mcp.WithNumber("limit", mcp.Description("Maximum number of memories to delete. Default: 20.")),
	)
	s.mcpServer.AddTool(deleteTool, s.handleDelete)
}

// =============================================================================
// Tool Handlers
// =============================================================================

// handleSearch implements the memory.search tool.
func (s *Server) handleSearch(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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

	// Build filters
	var filters []memory.Filter

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

	// Embed query
	vec, err := s.embedder.Embed(ctx, query)
	if err != nil {
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
		return mcp.NewToolResultError(fmt.Sprintf("search error: %v", err)), nil
	}

	if len(results) == 0 {
		return mcp.NewToolResultText("[]"), nil
	}

	// Apply 3-component scoring (the store returns raw cosine similarity)
	for i := range results {
		results[i].Score = memory.Score(&results[i].Memory, results[i].Score, s.weights, s.decay)
	}

	// Sort by final score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// For MMR, we need the vectors. Re-fetch them by searching with vectors included.
	// Since we don't have vectors in the result, we skip MMR if we can't get vectors.
	// For now, just truncate to limit.
	if len(results) > limit {
		results = results[:limit]
	}

	// Format output
	type searchResult struct {
		ID         string         `json:"id"`
		Type       string         `json:"type"`
		Content    string         `json:"content"`
		Source     string         `json:"source"`
		Importance float64        `json:"importance"`
		Tags       []string       `json:"tags"`
		CreatedAt  float64        `json:"created_at"`
		Score      float64        `json:"score"`
		Metadata   map[string]any `json:"metadata,omitempty"`
	}

	output := make([]searchResult, len(results))
	for i, r := range results {
		output[i] = searchResult{
			ID:         r.ID,
			Type:       string(r.Type),
			Content:    r.Content,
			Source:     r.Source,
			Importance: r.Importance,
			Tags:       r.Tags,
			CreatedAt:  r.CreatedAt,
			Score:      r.Score,
			Metadata:   r.Metadata,
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
	content, err := request.RequireString("content")
	if err != nil {
		return mcp.NewToolResultError("content is required"), nil
	}

	memType := memory.MemoryType(request.GetString("type", "event"))
	if !memory.ValidTypes[memType] {
		return mcp.NewToolResultError(fmt.Sprintf("invalid memory type: %s", memType)), nil
	}

	importance := request.GetFloat("importance", 5.0)
	if importance < 1 {
		importance = 1
	}
	if importance > 10 {
		importance = 10
	}

	source := request.GetString("source", "agent")
	tags := getStringSlice(request, "tags")

	// Create the memory
	mem := memory.New(content,
		memory.WithType(memType),
		memory.WithImportance(importance),
		memory.WithSource(source),
		memory.WithTags(tags...),
	)

	// Embed content
	vec, err := s.embedder.Embed(ctx, content)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	// Check for duplicates
	dupeResults, err := s.store.Search(ctx, vec, memory.SearchOptions{
		Limit: 3,
	})
	if err == nil {
		if dup := memory.IsDuplicate(dupeResults, s.dedupThreshold); dup != nil {
			type dupResult struct {
				Status   string `json:"status"`
				Message  string `json:"message"`
				Existing struct {
					ID      string  `json:"id"`
					Content string  `json:"content"`
					Score   float64 `json:"score"`
				} `json:"existing"`
			}
			result := dupResult{
				Status:  "duplicate",
				Message: "A very similar memory already exists. Skipped.",
			}
			result.Existing.ID = dup.ID
			result.Existing.Content = dup.Content
			result.Existing.Score = dup.Score
			data, _ := json.Marshal(result)
			return mcp.NewToolResultText(string(data)), nil
		}
	}

	// Insert
	if err := s.store.Insert(ctx, mem, vec); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("insert error: %v", err)), nil
	}

	// Return the created memory
	type addResult struct {
		Status string         `json:"status"`
		Memory *memory.Memory `json:"memory"`
	}
	result := addResult{
		Status: "created",
		Memory: mem,
	}
	data, err := json.Marshal(result)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("json marshal error: %v", err)), nil
	}

	return mcp.NewToolResultText(string(data)), nil
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
	memType := memory.MemoryType(request.GetString("type", "event"))
	importance := request.GetFloat("importance", 5.0)
	if importance < 1 {
		importance = 1
	}
	if importance > 10 {
		importance = 10
	}
	tags := getStringSlice(request, "tags")

	// Step 1: Embed old_content and find matching memories
	oldVec, err := s.embedder.Embed(ctx, oldContent)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	searchResults, err := s.store.Search(ctx, oldVec, memory.SearchOptions{
		Limit: 20,
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

	// Step 2: Delete matching memories
	deletedCount := 0
	if len(toDelete) > 0 {
		deletedCount, err = s.store.Delete(ctx, toDelete)
		if err != nil {
			return mcp.NewToolResultError(fmt.Sprintf("delete error: %v", err)), nil
		}
	}

	// Step 3: Create new memory
	mem := memory.New(newContent,
		memory.WithType(memType),
		memory.WithImportance(importance),
		memory.WithTags(tags...),
	)

	newVec, err := s.embedder.Embed(ctx, newContent)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding new content error: %v", err)), nil
	}

	if err := s.store.Insert(ctx, mem, newVec); err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("insert error: %v", err)), nil
	}

	// Format deleted items for response
	type deletedItem struct {
		ID      string  `json:"id"`
		Content string  `json:"content"`
		Score   float64 `json:"score"`
	}
	deleted := make([]deletedItem, len(deletedMemories))
	for i, d := range deletedMemories {
		deleted[i] = deletedItem{
			ID:      d.ID,
			Content: d.Content,
			Score:   d.Score,
		}
	}

	type updateResult struct {
		Status       string         `json:"status"`
		DeletedCount int            `json:"deleted_count"`
		Deleted      []deletedItem  `json:"deleted"`
		NewMemory    *memory.Memory `json:"new_memory"`
	}

	result := updateResult{
		Status:       "updated",
		DeletedCount: deletedCount,
		Deleted:      deleted,
		NewMemory:    mem,
	}

	data, _ := json.Marshal(result)
	return mcp.NewToolResultText(string(data)), nil
}

// handleDelete implements the memory.delete tool.
func (s *Server) handleDelete(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	query, err := request.RequireString("query")
	if err != nil {
		return mcp.NewToolResultError("query is required"), nil
	}

	threshold := request.GetFloat("similarity_threshold", 0.7)
	limit := request.GetInt("limit", 20)
	if limit < 1 {
		limit = 1
	}

	// Embed query
	vec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("embedding error: %v", err)), nil
	}

	// Search for matching memories
	results, err := s.store.Search(ctx, vec, memory.SearchOptions{
		Limit: limit,
	})
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("search error: %v", err)), nil
	}

	// Filter by similarity threshold
	var toDelete []string
	type deletedItem struct {
		ID      string  `json:"id"`
		Content string  `json:"content"`
		Score   float64 `json:"score"`
	}
	var deletedItems []deletedItem

	for _, r := range results {
		if r.Score >= threshold {
			toDelete = append(toDelete, r.ID)
			deletedItems = append(deletedItems, deletedItem{
				ID:      r.ID,
				Content: r.Content,
				Score:   r.Score,
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

	// Delete
	deletedCount, err := s.store.Delete(ctx, toDelete)
	if err != nil {
		return mcp.NewToolResultError(fmt.Sprintf("delete error: %v", err)), nil
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
// Helpers
// =============================================================================

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
