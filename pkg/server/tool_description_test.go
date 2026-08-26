// tool_description_test.go — regression guard for MCP tool parameter descriptions.
//
// H3: the memory_add / memory_update "importance" parameter description must
// reflect the per-type A-MAC defaults/bounds, NOT a hardcoded "1-10. Default: 5."
package server

import (
	"context"
	"strings"
	"testing"

	"github.com/mark3labs/mcp-go/mcp"
)

// importanceDescription lists tools/list and returns the "importance" parameter
// description advertised for the named tool.
func importanceDescription(t *testing.T, toolName string) string {
	t.Helper()
	ts, _ := buildMCPTestServer(t)
	c := newMCPClient(t, ts.URL, testPigoKey)
	tools, err := c.ListTools(context.Background(), mcp.ListToolsRequest{})
	if err != nil {
		t.Fatalf("tools/list: %v", err)
	}
	for _, tl := range tools.Tools {
		if tl.Name != toolName {
			continue
		}
		prop, ok := tl.InputSchema.Properties["importance"].(map[string]any)
		if !ok {
			t.Fatalf("%s: importance property missing or wrong shape: %#v", toolName, tl.InputSchema.Properties["importance"])
		}
		desc, _ := prop["description"].(string)
		return desc
	}
	t.Fatalf("%s not advertised in tools/list", toolName)
	return ""
}

func TestMemoryAdd_ImportanceDescription_PerType(t *testing.T) {
	desc := importanceDescription(t, "memory_add")
	if !strings.Contains(desc, "Per-type defaults") {
		t.Fatalf("memory_add importance description missing per-type defaults; got %q", desc)
	}
	if strings.Contains(desc, "Default: 5") {
		t.Fatalf("memory_add importance description regressed to hardcoded default; got %q", desc)
	}
}

func TestMemoryUpdate_ImportanceDescription_PerType(t *testing.T) {
	desc := importanceDescription(t, "memory_update")
	if !strings.Contains(desc, "Per-type defaults") {
		t.Fatalf("memory_update importance description missing per-type defaults; got %q", desc)
	}
}
