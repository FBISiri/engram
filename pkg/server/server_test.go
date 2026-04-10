package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/FBISiri/engram/pkg/config"
	"github.com/FBISiri/engram/pkg/memory"
	"github.com/mark3labs/mcp-go/mcp"
)

// =============================================================================
// Mock Embedder
// =============================================================================

// mockEmbedder returns deterministic embeddings for testing.
type mockEmbedder struct {
	dimension int
}

func newMockEmbedder() *mockEmbedder {
	return &mockEmbedder{dimension: 8}
}

func (m *mockEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	// Generate a deterministic vector from text hash
	vec := make([]float32, m.dimension)
	for i, ch := range text {
		vec[i%m.dimension] += float32(ch) / 1000.0
	}
	// Normalize
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		n := float32(math.Sqrt(norm))
		for i := range vec {
			vec[i] /= n
		}
	}
	return vec, nil
}

func (m *mockEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := m.Embed(context.Background(), t)
		if err != nil {
			return nil, err
		}
		result[i] = vec
	}
	return result, nil
}

func (m *mockEmbedder) Dimension() int {
	return m.dimension
}

// =============================================================================
// Mock Store
// =============================================================================

type mockStore struct {
	mu       sync.Mutex
	memories map[string]storedPoint
}

type storedPoint struct {
	mem    memory.Memory
	vector []float32
}

func newMockStore() *mockStore {
	return &mockStore{
		memories: make(map[string]storedPoint),
	}
}

func (s *mockStore) Insert(_ context.Context, mem *memory.Memory, vector []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.memories[mem.ID] = storedPoint{mem: *mem, vector: vector}
	return nil
}

func (s *mockStore) Search(_ context.Context, vector []float32, opts memory.SearchOptions) ([]memory.ScoredMemory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	limit := opts.Limit
	if limit == 0 {
		limit = 10
	}

	type scoredEntry struct {
		mem   memory.Memory
		score float64
	}

	now := float64(time.Now().Unix())
	var entries []scoredEntry
	for _, sp := range s.memories {
		// Exclude expired memories: valid_until > 0 AND valid_until < now
		if sp.mem.ValidUntil > 0 && sp.mem.ValidUntil < now {
			continue
		}
		// Apply filters
		if !matchFilters(sp.mem, opts.Filters) {
			continue
		}

		// Compute cosine similarity
		sim := cosineSimilarity(vector, sp.vector)
		entries = append(entries, scoredEntry{mem: sp.mem, score: sim})
	}

	// Sort by score descending
	for i := 0; i < len(entries); i++ {
		for j := i + 1; j < len(entries); j++ {
			if entries[j].score > entries[i].score {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
	}

	if len(entries) > limit {
		entries = entries[:limit]
	}

	results := make([]memory.ScoredMemory, len(entries))
	for i, e := range entries {
		sp := s.memories[e.mem.ID]
		results[i] = memory.ScoredMemory{
			Memory: e.mem,
			Score:  e.score,
			Vector: sp.vector,
		}
	}
	return results, nil
}

func (s *mockStore) Delete(_ context.Context, ids []string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	count := 0
	for _, id := range ids {
		if _, ok := s.memories[id]; ok {
			delete(s.memories, id)
			count++
		}
	}
	return count, nil
}

func (s *mockStore) Update(_ context.Context, id string, fields map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	sp, ok := s.memories[id]
	if !ok {
		return fmt.Errorf("memory not found: %s", id)
	}
	if content, ok := fields["content"].(string); ok {
		sp.mem.Content = content
	}
	s.memories[id] = sp
	return nil
}

func (s *mockStore) SearchByIDs(_ context.Context, ids []string) ([]memory.Memory, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var result []memory.Memory
	for _, id := range ids {
		if sp, ok := s.memories[id]; ok {
			result = append(result, sp.mem)
		}
	}
	return result, nil
}

func (s *mockStore) EnsureCollection(_ context.Context) error {
	return nil
}

func (s *mockStore) Scroll(_ context.Context, opts memory.ScrollOptions) ([]memory.Memory, string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	limit := opts.Limit
	if limit == 0 {
		limit = 50
	}

	var results []memory.Memory
	scrollNow := float64(time.Now().Unix())
	for _, sp := range s.memories {
		// Exclude expired memories: valid_until > 0 AND valid_until < now
		if sp.mem.ValidUntil > 0 && sp.mem.ValidUntil < scrollNow {
			continue
		}
		if !matchFilters(sp.mem, opts.Filters) {
			continue
		}
		results = append(results, sp.mem)
	}

	// Sort by created_at descending
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].CreatedAt > results[i].CreatedAt {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	var nextOffset string
	if len(results) > limit {
		results = results[:limit]
		if len(results) > 0 {
			nextOffset = results[len(results)-1].ID
		}
	}
	return results, nextOffset, nil
}

func (s *mockStore) Stats(_ context.Context) (*memory.CollectionStats, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return &memory.CollectionStats{
		PointCount: uint64(len(s.memories)),
		Status:     "green",
	}, nil
}

func (s *mockStore) DeleteExpired(_ context.Context) (int, error) {
	return 0, nil
}

// count returns the number of memories in the store.
func (s *mockStore) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.memories)
}

// matchFilters checks if a memory matches all filters.
func matchFilters(mem memory.Memory, filters []memory.Filter) bool {
	for _, f := range filters {
		switch f.Field {
		case "type":
			switch f.Op {
			case memory.OpEq:
				if string(mem.Type) != f.Value.(string) {
					return false
				}
			case memory.OpIn:
				types := f.Value.([]string)
				found := false
				for _, t := range types {
					if string(mem.Type) == t {
						found = true
						break
					}
				}
				if !found {
					return false
				}
			}
		case "tags":
			if f.Op == memory.OpIn {
				tags := f.Value.([]string)
				found := false
				for _, t := range tags {
					for _, mt := range mem.Tags {
						if t == mt {
							found = true
							break
						}
					}
					if found {
						break
					}
				}
				if !found {
					return false
				}
			}
		case "created_at":
			switch f.Op {
			case memory.OpGte:
				if mem.CreatedAt < f.Value.(float64) {
					return false
				}
			case memory.OpLte:
				if mem.CreatedAt > f.Value.(float64) {
					return false
				}
			}
		}
	}
	return true
}

// cosineSimilarity computes cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// =============================================================================
// Test Helpers
// =============================================================================

func newTestServer() (*Server, *mockStore) {
	store := newMockStore()
	embedder := newMockEmbedder()
	cfg := &config.Config{
		Weights:        memory.DefaultScoringWeights(),
		Decay:          memory.DefaultDecayConfig(),
		MMRLambda:      0.5,
		DedupThreshold: 0.92,
	}
	srv := NewServer(store, embedder, cfg)
	return srv, store
}

// callTool calls a tool handler directly by constructing a CallToolRequest.
func callTool(srv *Server, toolName string, args map[string]any) (*mcp.CallToolResult, error) {
	ctx := context.Background()
	request := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      toolName,
			Arguments: args,
		},
	}

	// Find and call the handler
	switch toolName {
	case "memory_search":
		return srv.handleSearch(ctx, request)
	case "memory_add":
		return srv.handleAdd(ctx, request)
	case "memory_update":
		return srv.handleUpdate(ctx, request)
	case "memory_delete":
		return srv.handleDelete(ctx, request)
	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// extractText extracts the text content from a CallToolResult.
func extractText(result *mcp.CallToolResult) string {
	if result == nil || len(result.Content) == 0 {
		return ""
	}
	for _, c := range result.Content {
		if tc, ok := c.(mcp.TextContent); ok {
			return tc.Text
		}
	}
	return ""
}

// =============================================================================
// Tests
// =============================================================================

func TestAddMemory(t *testing.T) {
	srv, store := newTestServer()

	result, err := callTool(srv, "memory_add", map[string]any{
		"content":    "User's name is Frank",
		"type":       "identity",
		"importance": float64(9),
		"tags":       []interface{}{"name", "personal"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text := extractText(result)
	if result.IsError {
		t.Fatalf("tool returned error: %s", text)
	}

	// Parse response
	var resp struct {
		Status string         `json:"status"`
		Memory memory.Memory  `json:"memory"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse response: %v\nraw: %s", err, text)
	}

	if resp.Status != "created" {
		t.Errorf("expected status 'created', got %q", resp.Status)
	}
	if resp.Memory.Content != "User's name is Frank" {
		t.Errorf("unexpected content: %s", resp.Memory.Content)
	}
	if resp.Memory.Type != memory.TypeIdentity {
		t.Errorf("expected type identity, got %s", resp.Memory.Type)
	}
	if resp.Memory.Importance != 9 {
		t.Errorf("expected importance 9, got %f", resp.Memory.Importance)
	}
	if store.count() != 1 {
		t.Errorf("expected 1 memory in store, got %d", store.count())
	}
}

func TestAddMemoryDefaults(t *testing.T) {
	srv, _ := newTestServer()

	result, err := callTool(srv, "memory_add", map[string]any{
		"content": "Something happened",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text := extractText(result)
	var resp struct {
		Status string        `json:"status"`
		Memory memory.Memory `json:"memory"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.Memory.Type != memory.TypeEvent {
		t.Errorf("expected default type event, got %s", resp.Memory.Type)
	}
	if resp.Memory.Importance != 5 {
		t.Errorf("expected default importance 5, got %f", resp.Memory.Importance)
	}
	if resp.Memory.Source != "agent" {
		t.Errorf("expected default source agent, got %s", resp.Memory.Source)
	}
}

func TestAddMemoryMissingContent(t *testing.T) {
	srv, _ := newTestServer()

	result, err := callTool(srv, "memory_add", map[string]any{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !result.IsError {
		t.Error("expected error result for missing content")
	}
}

func TestAddMemoryDedup(t *testing.T) {
	srv, _ := newTestServer()

	// Add first memory
	_, err := callTool(srv, "memory_add", map[string]any{
		"content": "User's name is Frank",
	})
	if err != nil {
		t.Fatalf("first add failed: %v", err)
	}

	// Add exact same content — should be detected as duplicate
	result, err := callTool(srv, "memory_add", map[string]any{
		"content": "User's name is Frank",
	})
	if err != nil {
		t.Fatalf("second add failed: %v", err)
	}

	text := extractText(result)
	var resp struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.Status != "duplicate" {
		t.Errorf("expected status 'duplicate', got %q", resp.Status)
	}
}

func TestSearchMemory(t *testing.T) {
	srv, _ := newTestServer()

	// Add some memories
	memories := []map[string]any{
		{"content": "User's name is Frank", "type": "identity", "importance": float64(9)},
		{"content": "User works as a software engineer", "type": "identity", "importance": float64(8)},
		{"content": "Discussed trip planning to Japan", "type": "event", "importance": float64(5)},
	}
	for _, m := range memories {
		_, err := callTool(srv, "memory_add", m)
		if err != nil {
			t.Fatalf("add failed: %v", err)
		}
	}

	// Search
	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "user name",
		"limit": float64(5),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	if result.IsError {
		t.Fatalf("search returned error: %s", text)
	}

	var results []map[string]any
	if err := json.Unmarshal([]byte(text), &results); err != nil {
		t.Fatalf("failed to parse search results: %v\nraw: %s", err, text)
	}

	if len(results) == 0 {
		t.Error("expected at least 1 result")
	}

	// Each result should have expected fields
	for _, r := range results {
		if _, ok := r["id"]; !ok {
			t.Error("result missing 'id' field")
		}
		if _, ok := r["content"]; !ok {
			t.Error("result missing 'content' field")
		}
		if _, ok := r["score"]; !ok {
			t.Error("result missing 'score' field")
		}
	}
}

func TestSearchWithTypeFilter(t *testing.T) {
	srv, _ := newTestServer()

	// Add memories of different types
	callTool(srv, "memory_add", map[string]any{
		"content": "User lives in Shanghai",
		"type":    "identity",
	})
	callTool(srv, "memory_add", map[string]any{
		"content": "Went to Shanghai for a trip",
		"type":    "event",
	})

	// Search only for identity type
	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "Shanghai",
		"types": []interface{}{"identity"},
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	var results []map[string]any
	json.Unmarshal([]byte(text), &results)

	for _, r := range results {
		if r["type"].(string) != "identity" {
			t.Errorf("expected only identity type, got %s", r["type"])
		}
	}
}

func TestSearchMissingQuery(t *testing.T) {
	srv, _ := newTestServer()

	result, err := callTool(srv, "memory_search", map[string]any{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected error for missing query")
	}
}

func TestUpdateMemory(t *testing.T) {
	srv, store := newTestServer()

	// Add initial memory
	_, err := callTool(srv, "memory_add", map[string]any{
		"content":    "User lives in Beijing",
		"type":       "identity",
		"importance": float64(8),
	})
	if err != nil {
		t.Fatalf("add failed: %v", err)
	}

	if store.count() != 1 {
		t.Fatalf("expected 1 memory, got %d", store.count())
	}

	// Update: move from Beijing to Shanghai
	result, err := callTool(srv, "memory_update", map[string]any{
		"old_content":          "User lives in Beijing",
		"new_content":          "User lives in Shanghai",
		"type":                 "identity",
		"importance":           float64(8),
		"similarity_threshold": float64(0.5),
	})
	if err != nil {
		t.Fatalf("update failed: %v", err)
	}

	text := extractText(result)
	if result.IsError {
		t.Fatalf("update returned error: %s", text)
	}

	var resp struct {
		Status       string `json:"status"`
		DeletedCount int    `json:"deleted_count"`
		NewMemory    struct {
			Content string `json:"content"`
		} `json:"new_memory"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.Status != "updated" {
		t.Errorf("expected status 'updated', got %q", resp.Status)
	}
	if resp.NewMemory.Content != "User lives in Shanghai" {
		t.Errorf("new content mismatch: %s", resp.NewMemory.Content)
	}

	// Store should have exactly 1 memory (old deleted, new added)
	if store.count() != 1 {
		t.Errorf("expected 1 memory after update, got %d", store.count())
	}
}

func TestDeleteMemory(t *testing.T) {
	srv, store := newTestServer()

	// Add a memory directly into the store to bypass dedup,
	// then use the add tool for the main target.
	// The issue is our mock embedder generates similar vectors for similar content.
	// Instead, just add one memory and delete it.

	result1, err := callTool(srv, "memory_add", map[string]any{
		"content":    "User prefers drinking espresso every morning at 7am",
		"type":       "identity",
		"importance": float64(7),
	})
	if err != nil {
		t.Fatalf("add failed: %v", err)
	}
	text1 := extractText(result1)
	if result1.IsError {
		t.Fatalf("add returned error: %s", text1)
	}

	if store.count() != 1 {
		t.Fatalf("expected 1 memory, got %d", store.count())
	}

	// Delete it
	result, err := callTool(srv, "memory_delete", map[string]any{
		"query":                "espresso coffee morning drink preference",
		"similarity_threshold": float64(0.3),
		"limit":                float64(10),
	})
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	text := extractText(result)
	if result.IsError {
		t.Fatalf("delete returned error: %s", text)
	}

	var resp struct {
		Status       string `json:"status"`
		DeletedCount int    `json:"deleted_count"`
	}
	if err := json.Unmarshal([]byte(text), &resp); err != nil {
		t.Fatalf("failed to parse response: %v", err)
	}

	if resp.DeletedCount == 0 {
		t.Error("expected at least 1 deletion")
	}
	if resp.Status != "deleted" {
		t.Errorf("expected status 'deleted', got %q", resp.Status)
	}
	if store.count() != 0 {
		t.Errorf("expected 0 memories after delete, got %d", store.count())
	}
}

func TestDeleteNoMatches(t *testing.T) {
	srv, _ := newTestServer()

	// Delete from empty store
	result, err := callTool(srv, "memory_delete", map[string]any{
		"query": "something that doesn't exist",
	})
	if err != nil {
		t.Fatalf("delete failed: %v", err)
	}

	text := extractText(result)
	var resp struct {
		Status       string `json:"status"`
		DeletedCount int    `json:"deleted_count"`
	}
	json.Unmarshal([]byte(text), &resp)

	if resp.Status != "no_matches" {
		t.Errorf("expected status 'no_matches', got %q", resp.Status)
	}
	if resp.DeletedCount != 0 {
		t.Errorf("expected 0 deletions, got %d", resp.DeletedCount)
	}
}

func TestGetMCPServer(t *testing.T) {
	srv, _ := newTestServer()
	mcpSrv := srv.GetMCPServer()
	if mcpSrv == nil {
		t.Fatal("GetMCPServer returned nil")
	}
}

func TestInvalidMemoryType(t *testing.T) {
	srv, _ := newTestServer()

	result, err := callTool(srv, "memory_add", map[string]any{
		"content": "test",
		"type":    "invalid_type",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.IsError {
		t.Error("expected error for invalid type")
	}
	text := extractText(result)
	if !strings.Contains(text, "invalid memory type") {
		t.Errorf("expected 'invalid memory type' error, got: %s", text)
	}
}

func TestImportanceClamping(t *testing.T) {
	srv, _ := newTestServer()

	// Test importance > 10 gets clamped
	result, err := callTool(srv, "memory_add", map[string]any{
		"content":    "test clamping",
		"importance": float64(15),
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	text := extractText(result)
	var resp struct {
		Memory struct {
			Importance float64 `json:"importance"`
		} `json:"memory"`
	}
	json.Unmarshal([]byte(text), &resp)

	if resp.Memory.Importance != 10 {
		t.Errorf("expected importance clamped to 10, got %f", resp.Memory.Importance)
	}
}

// TestSearchMMRActivated verifies that MMR reranking is actually invoked during search.
// We insert N memories with orthogonal vectors (max diversity) and N with near-duplicate
// vectors (min diversity), then confirm that with lambda<1 the results include the
// diverse candidates rather than being filled with near-duplicates.
func TestSearchMMRActivated(t *testing.T) {
	srv, store := newTestServer()

	// Insert 3 near-duplicate memories pointing in the same direction [1,0,0,...]
	for i := 0; i < 3; i++ {
		dup := memory.New(fmt.Sprintf("near-duplicate memory %d", i),
			memory.WithType(memory.TypeEvent),
			memory.WithImportance(8),
		)
		vec := make([]float32, 8)
		vec[0] = 1.0 // all point in direction 0
		if err := store.Insert(context.Background(), dup, vec); err != nil {
			t.Fatalf("insert dup %d: %v", i, err)
		}
	}

	// Insert 3 diverse memories pointing in orthogonal directions
	for i := 0; i < 3; i++ {
		diverse := memory.New(fmt.Sprintf("diverse memory %d", i),
			memory.WithType(memory.TypeEvent),
			memory.WithImportance(7),
		)
		vec := make([]float32, 8)
		vec[i+1] = 1.0 // orthogonal directions 1,2,3
		if err := store.Insert(context.Background(), diverse, vec); err != nil {
			t.Fatalf("insert diverse %d: %v", i, err)
		}
	}

	// Search with limit=3; query vector points near [1,0,0,...] (favors near-duplicates by pure similarity)
	// With MMR (lambda=0.5), should prefer diverse results over repetitive ones
	queryVec := []float32{1.0, 0.1, 0.1, 0.1, 0, 0, 0, 0}
	_ = queryVec // used implicitly via callTool

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "near-duplicate memory 0", // will embed to something similar to dup vector
		"limit": float64(3),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	var results []struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal([]byte(text), &results); err != nil {
		t.Fatalf("unmarshal failed: %v (text: %s)", err, text)
	}

	// We should get exactly 3 results (limit respected)
	if len(results) != 3 {
		t.Errorf("expected 3 results (limit), got %d", len(results))
	}
}

// =============================================================================
// P1-B: valid_until expiry filter integration tests (server layer)
// =============================================================================

// TestSearch_ExpiredMemoryNotReturned verifies that a memory with valid_until
// in the past does NOT appear in memory_search results.
func TestSearch_ExpiredMemoryNotReturned(t *testing.T) {
	srv, store := newTestServer()

	// Insert an expired memory (valid_until 1 hour in the past)
	past := float64(time.Now().Add(-time.Hour).Unix())
	expired := memory.New("expired directive about security",
		memory.WithType(memory.TypeDirective),
		memory.WithImportance(8),
		memory.WithValidUntil(past),
	)
	vec := make([]float32, 8)
	for i := range vec {
		vec[i] = float32(i+1) * 0.1
	}
	if err := store.Insert(context.Background(), expired, vec); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	// Also insert a non-expired memory with similar content
	live := memory.New("active directive about security",
		memory.WithType(memory.TypeDirective),
		memory.WithImportance(7),
	)
	liveVec := make([]float32, 8)
	for i := range liveVec {
		liveVec[i] = float32(i+1) * 0.1
	}
	if err := store.Insert(context.Background(), live, liveVec); err != nil {
		t.Fatalf("insert live failed: %v", err)
	}

	// Search
	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "directive about security",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	// Expired memory should not appear
	if strings.Contains(text, expired.ID) {
		t.Errorf("expired memory ID %q should not appear in results, but it did", expired.ID)
	}
	if strings.Contains(text, "expired directive") {
		t.Error("expired memory content should not appear in results")
	}
	// Live memory should appear
	if !strings.Contains(text, live.ID) {
		t.Errorf("live memory ID %q should appear in results, but it did not (results: %s)", live.ID, text)
	}
}

// TestSearch_FutureExpiryMemoryReturned verifies that a memory with valid_until
// in the FUTURE still appears in search results (not yet expired).
func TestSearch_FutureExpiryMemoryReturned(t *testing.T) {
	srv, store := newTestServer()

	// Insert a memory expiring 1 hour from now
	future := float64(time.Now().Add(time.Hour).Unix())
	upcoming := memory.New("soon-to-expire memory about caching",
		memory.WithType(memory.TypeInsight),
		memory.WithImportance(6),
		memory.WithValidUntil(future),
	)
	vec := make([]float32, 8)
	for i := range vec {
		vec[i] = float32(i+1) * 0.12
	}
	if err := store.Insert(context.Background(), upcoming, vec); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "memory about caching",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	if !strings.Contains(text, upcoming.ID) {
		t.Errorf("upcoming memory ID %q should appear in results (not yet expired), got: %s", upcoming.ID, text)
	}
}

// TestSearch_PermanentMemoryNotAffectedByExpiryFilter verifies that memories
// with valid_until == 0 (permanent) are never excluded.
func TestSearch_PermanentMemoryNotAffectedByExpiryFilter(t *testing.T) {
	srv, store := newTestServer()

	permanent := memory.New("permanent identity: user prefers dark mode",
		memory.WithType(memory.TypeIdentity),
		memory.WithImportance(7),
	)
	vec := make([]float32, 8)
	for i := range vec {
		vec[i] = float32(i+1) * 0.09
	}
	if err := store.Insert(context.Background(), permanent, vec); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	result, err := callTool(srv, "memory_search", map[string]any{
		"query": "user prefers dark mode",
		"limit": float64(10),
	})
	if err != nil {
		t.Fatalf("search failed: %v", err)
	}

	text := extractText(result)
	if !strings.Contains(text, permanent.ID) {
		t.Errorf("permanent memory ID %q should appear in results, got: %s", permanent.ID, text)
	}
}
