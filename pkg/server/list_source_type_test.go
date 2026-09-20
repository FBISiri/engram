package server

// Negative-control test for R1: the memory_list handler must project
// source_type (from metadata) onto every record, matching memory_search
// semantics via the shared sourceTypeFromMetadata helper. A record stored
// with a known source_type MUST come back through the LIST handler carrying
// that exact value — not merely a present key.

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
)

func TestList_ReturnsSourceType(t *testing.T) {
	srv, _ := newTestServer()

	// Store a memory with a known source_type through the MCP add path,
	// which persists source_type into metadata.
	if _, err := callTool(srv, "memory_add", map[string]any{
		"content":     "Reflection: the deploy pipeline stabilized after the rollback fix",
		"source_type": "reflection",
	}); err != nil {
		t.Fatalf("add failed: %v", err)
	}

	// Retrieve via the memory_list HANDLER (filter-only Scroll path).
	req := mcp.CallToolRequest{
		Params: mcp.CallToolParams{
			Name:      "memory_list",
			Arguments: map[string]any{"limit": float64(10)},
		},
	}
	result, err := srv.handleList(context.Background(), req)
	if err != nil {
		t.Fatalf("handleList failed: %v", err)
	}
	if result.IsError {
		t.Fatalf("handleList returned error: %s", extractText(result))
	}

	var records []struct {
		Content    string         `json:"content"`
		SourceType string         `json:"source_type"`
		Metadata   map[string]any `json:"metadata"`
	}
	if err := json.Unmarshal([]byte(extractText(result)), &records); err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d: %s", len(records), extractText(result))
	}
	if records[0].SourceType != "reflection" {
		t.Errorf("expected source_type=reflection, got %q: %s", records[0].SourceType, extractText(result))
	}
	// R2: metadata is also carried through on the Scroll/ListMemories path.
	if got, _ := records[0].Metadata["source_type"].(string); got != "reflection" {
		t.Errorf("expected metadata.source_type=reflection, got %v", records[0].Metadata["source_type"])
	}
}
